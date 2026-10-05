import 'dart:async';
import 'dart:collection';
import 'dart:math' as math;

import 'alignment.dart';
import 'calibration.dart';
import 'fusion_filter.dart';
import 'geo.dart';
import 'gnss_monitor.dart';
import 'map_matcher.dart';
import 'road_graph.dart';
import 'signal.dart';
import 'vec3.dart';

export 'alignment.dart' show AlignmentState;
export 'calibration.dart' show Calibration, MountAngles;
export 'gnss_monitor.dart' show GnssFix, GnssHealth;

enum NavMode { gnssIns, degraded, deadReckoning }

/// One IMU sample in the phone frame.
class ImuSample {
  const ImuSample({
    required this.t,
    required this.accel,
    required this.gyro,
    this.linear = Vec3.zero,
    this.mag = Vec3.zero,
  });

  final double t; // seconds
  final Vec3 accel; // m/s², includes gravity
  final Vec3 linear; // m/s², gravity removed (Android linear acceleration); zero if unavailable
  final Vec3 gyro; // rad/s
  final Vec3 mag; // microtesla; zero if unavailable
}

/// What happened when GNSS came back after a blackout.
class RecoveryStats {
  const RecoveryStats({
    required this.errorM,
    required this.distanceM,
    required this.seconds,
    required this.mapAided,
  });

  final double errorM;
  final double distanceM;
  final double seconds;
  final bool mapAided;

  double get driftPct => distanceM > 1 ? 100.0 * errorM / distanceM : 0;
}

typedef GeoPoint = ({double lat, double lon});

/// Snapshot of the navigation solution for the UI.
class NavState {
  const NavState({
    required this.t,
    required this.lat,
    required this.lon,
    required this.headingDeg,
    required this.speedMs,
    required this.mode,
    required this.gnssHealth,
    required this.hasFix,
    required this.fixAgeS,
    required this.fixAccuracyM,
    required this.uncertaintyM,
    required this.headingSigmaDeg,
    required this.speedSigmaMs,
    required this.blackoutSeconds,
    required this.blackoutDistanceM,
    required this.mapAided,
    required this.trail,
    required this.gnssTrack,
    required this.freeDrPosition,
    required this.lastRecovery,
    required this.alignment,
    required this.calibration,
    required this.mount,
    required this.gyroBiasDps,
    required this.accelBiasMs2,
    required this.fwdAccelHistory,
    required this.yawRateHistory,
    required this.vibrationRms,
    required this.shockCount,
    required this.roadSegments,
    required this.simulated,
    required this.acceptedFixes,
    required this.rejectedFixes,
  });

  final double t;
  final double lat, lon;
  final double headingDeg;
  final double speedMs;
  final NavMode mode;
  final GnssHealth gnssHealth;
  final bool hasFix;
  final double fixAgeS;
  final double fixAccuracyM;

  /// 2-sigma horizontal position uncertainty from the fusion filter (m).
  final double uncertaintyM;
  final double headingSigmaDeg;
  final double speedSigmaMs;
  final double blackoutSeconds;
  final double blackoutDistanceM;
  final bool mapAided;
  final List<GeoPoint> trail;
  final List<GeoPoint> gnssTrack;
  final GeoPoint? freeDrPosition;
  final RecoveryStats? lastRecovery;
  final AlignmentState alignment;
  final Calibration? calibration;
  final MountAngles? mount;
  final double gyroBiasDps;
  final double accelBiasMs2;
  final List<double> fwdAccelHistory; // m/s², newest last
  final List<double> yawRateHistory; // deg/s, newest last
  final double vibrationRms; // m/s²
  final int shockCount;
  final int roadSegments;
  final bool simulated;
  final int acceptedFixes;
  final int rejectedFixes;

  bool get isDeadReckoning => mode == NavMode.deadReckoning;
}

/// Orchestrates alignment, calibration, the GNSS+INS fusion filter, the GNSS
/// deficit handler and map-aided dead reckoning. Feed it IMU samples and GNSS
/// fixes; read [states].
class NavigationEngine {
  NavigationEngine({
    GnssMonitor? monitor,
    FusionFilter? filter,
    this.emitInterval = 0.1,
    this.gravityTau = 1.0,
    this.imuRate = 20.0,
    this.useCompass = false,
    this.accelTau = 0.4,
    this.outageDeadband = 0.02,
    this.quietAccelThreshold = 0.3,
    this.residualTau = 8.0,
    this.shockThreshold = 8.0,
    this.historySeconds = 5.0,
  })  : monitor = monitor ?? GnssMonitor(),
        filter = filter ?? FusionFilter(),
        _aligner = InVehicleAligner(samplingRate: imuRate),
        _calibrator = BlackoutCalibrator(windowSeconds: 30, minSeconds: 8, fs: 10);

  final GnssMonitor monitor;
  final FusionFilter filter;
  final double emitInterval;
  final double gravityTau;
  final double imuRate;

  /// Use the calibrated compass as a weak heading measurement during outages.
  /// Off by default: in-vehicle and in-tunnel magnetometers are unreliable.
  bool useCompass;
  final double accelTau; // low-pass on the forward acceleration (s)
  final double outageDeadband; // m/s², soft dead-band on accel during outages
  final double quietAccelThreshold; // m/s², below this an outage stretch counts as constant speed
  final double residualTau; // s, time constant for re-learning the residual bias in quiet stretches
  final double shockThreshold; // m/s² |linear accel| treated as a pothole shock
  final double historySeconds;

  /// Offline road network used for map-aided dead reckoning (optional).
  RoadGraph? roadGraph;

  /// Demo switch: pretend GNSS is denied.
  bool simulateBlackout = false;

  final InVehicleAligner _aligner;
  final BlackoutCalibrator _calibrator;
  final _StationaryDetector _stationary = _StationaryDetector();
  final StreamController<NavState> _ctrl = StreamController<NavState>.broadcast();

  Stream<NavState> get states => _ctrl.stream;
  NavState? current;

  // GNSS bookkeeping
  GnssFix? _lastFix;
  double? _lastMovingCourseRad;
  double _lastAcceptedFixT = double.negativeInfinity;
  int _lastDisplacement = 0;
  double? _lat0, _lon0; // ENU origin of the fusion filter
  final ListQueue<GeoPoint> _gnssTrack = ListQueue<GeoPoint>();

  // IMU bookkeeping
  double? _lastImuT;
  Vec3 _grav = Vec3.zero;
  bool _gravInit = false;
  Vec3 _ghat = Vec3.unitZ;
  double _aLp = 0;
  double _residualBias = 0;
  double _linLp = 0;
  double _vibMs = 0;
  int _shockCount = 0;
  final ListQueue<double> _fwdAccelHist = ListQueue<double>();
  final ListQueue<double> _yawRateHist = ListQueue<double>();

  // Calibration products in use
  Calibration? _cal;
  Vec3? _fwd;
  double _fwdCorr = 0;
  double _yawSign = Calibration.defaultYawSign;

  // Mode / outage
  NavMode _mode = NavMode.degraded;
  MapAidedReckoner? _mapDr;
  double _blackoutStart = 0;
  double _blackoutDistance = 0;
  List<GeoPoint> _trail = <GeoPoint>[];
  RecoveryStats? _lastRecovery;
  GeoPoint? _display;
  double _headingRad = 0;
  double _lastEmit = -1;

  NavMode get mode => _mode;
  Calibration? get calibration => _cal;
  Vec3? get forwardAxis => _fwd;

  void dispose() => _ctrl.close();

  // ── GNSS ─────────────────────────────────────────────────────────────

  void onGnss(GnssFix f) {
    _lastFix = f;
    final health = monitor.assess(f, f.t, simulateDenied: simulateBlackout);

    if (health != GnssHealth.denied) {
      final speed = f.speedMs < 0.5 ? 0.0 : f.speedMs;
      if (f.speedMs > 1.0) _lastMovingCourseRad = degToRad(f.courseDeg);

      if (!filter.initialized) {
        _lat0 = f.lat;
        _lon0 = f.lon;
        filter.init(
          e: 0,
          n: 0,
          heading: f.speedMs > 2.0 ? degToRad(f.courseDeg) : null,
          speed: speed,
          gyroBias: _cal?.fromData == true ? _cal!.yawBias : 0,
          accelBias: _cal?.hasForwardAxis == true ? _cal!.accelBias : 0,
          posSigma: math.max(3.0, f.accuracyM),
        );
        _lastAcceptedFixT = f.t;
        _mode = health == GnssHealth.good ? NavMode.gnssIns : NavMode.degraded;
        _pushGnssTrack(f);
      } else {
        final enu = latLonToEnu(f.lat, f.lon, _lat0!, _lon0!);
        // Measure the dead-reckoning error against the first fix that comes
        // back, before the filter absorbs it.
        double? recoveryErr;
        if (_mode == NavMode.deadReckoning && _display != null) {
          recoveryErr = distanceM(_display!.lat, _display!.lon, f.lat, f.lon);
        }
        final sigma = health == GnssHealth.good
            ? math.max(3.0, f.accuracyM)
            : math.max(10.0, 2.0 * f.accuracyM);
        final dE = enu.e - filter.east, dN = enu.n - filter.north;
        final acceptedNow = filter.updatePosition(enu.e, enu.n, sigma);
        if (acceptedNow) {
          _rejectedInnovations.clear();
          filter.updateSpeed(speed, health == GnssHealth.good ? 0.3 : 0.6);
          if (f.speedMs > 2.0) {
            filter.updateCourse(degToRad(f.courseDeg), 0.06 + 0.5 / f.speedMs);
          }
          _lastAcceptedFixT = f.t;
          _pushGnssTrack(f);
          if (_mode == NavMode.deadReckoning) _exitBlackout(f, recoveryErr ?? 0);
        } else if (_consistentRejections(dE, dN)) {
          // Three rejected fixes in a row that agree with each other: that is
          // the GNSS being right and the filter being lost (long outage
          // drift, or the gate was over-confident). Snap to the fix.
          filter.reinitPosition(enu.e, enu.n, sigma);
          _rejectedInnovations.clear();
          _lastAcceptedFixT = f.t;
          _pushGnssTrack(f);
          if (_mode == NavMode.deadReckoning) _exitBlackout(f, recoveryErr ?? 0);
        }
      }
    }

    if (health == GnssHealth.good) {
      final c = _calibrator.compute();
      if (c != null) {
        _cal = c;
        _applyCalibration(c);
      }
    }
    _emit(f.t, force: true);
  }

  final List<(double, double)> _rejectedInnovations = <(double, double)>[];

  /// Records a rejected fix's innovation vector; true when the last three
  /// rejections agree with each other to within [consistencyM] metres
  /// (a steady GNSS offset relative to the filter, i.e. real GNSS), as
  /// opposed to scattered multipath jumps.
  bool _consistentRejections(double dE, double dN, {double consistencyM = 25.0}) {
    _rejectedInnovations.add((dE, dN));
    while (_rejectedInnovations.length > 3) {
      _rejectedInnovations.removeAt(0);
    }
    if (_rejectedInnovations.length < 3) return false;
    for (var i = 0; i < 3; i++) {
      for (var j = i + 1; j < 3; j++) {
        final a = _rejectedInnovations[i], b = _rejectedInnovations[j];
        final d = math.sqrt((a.$1 - b.$1) * (a.$1 - b.$1) + (a.$2 - b.$2) * (a.$2 - b.$2));
        if (d > consistencyM) return false;
      }
    }
    return true;
  }

  void _pushGnssTrack(GnssFix f) {
    _gnssTrack.addLast((lat: f.lat, lon: f.lon));
    while (_gnssTrack.length > 90) {
      _gnssTrack.removeFirst();
    }
  }

  /// Adopt a new forward axis only on better evidence, or on confident
  /// evidence of a clearly different axis (the phone was moved). A window
  /// with little kinematic signal must never replace a good axis. The
  /// accelerometer bias belongs to the axis, so it is reset when the axis
  /// changes substantially.
  void _applyCalibration(Calibration c) {
    if (!c.hasForwardAxis) return;
    final f = c.fwd!;
    _yawSign = c.yawSign;
    if (_fwd == null) {
      _fwd = f;
      _fwdCorr = c.forwardCorr;
      filter.resetAccelBias(c.accelBias, 0.3);
      return;
    }
    final angle = math.acos(_fwd!.dot(f).clamp(-1.0, 1.0));
    final different = angle > 15 * math.pi / 180;
    if (c.forwardCorr >= _fwdCorr + 0.03 || (different && c.forwardCorr > 0.9)) {
      _fwd = f;
      _fwdCorr = c.forwardCorr;
      if (different) filter.resetAccelBias(c.accelBias, 0.3);
    } else if (!different && c.forwardCorr >= _fwdCorr - 0.05) {
      // Comparable evidence, same axis: track it slowly so the estimate
      // keeps improving with more data.
      _fwd = (_fwd! * 0.8 + f * 0.2).normalized;
    }
  }

  void _onDisplacement() {
    _fwd = null;
    _fwdCorr = 0;
    _cal = null;
    _calibrator.clear();
    filter.resetAccelBias(0, 0.3);
  }

  // ── IMU ──────────────────────────────────────────────────────────────

  void onImu(ImuSample s) {
    final last = _lastImuT;
    final dt = last == null ? 1.0 / imuRate : (s.t - last).clamp(0.001, 0.25);
    _lastImuT = s.t;

    // Gravity estimate: Android linear acceleration when available, slow
    // low-pass of the raw accelerometer otherwise.
    final cand = s.linear.norm2 > 1e-6 ? s.accel - s.linear : s.accel;
    if (!_gravInit) {
      _grav = cand;
      _gravInit = true;
    } else {
      _grav = _grav + (cand - _grav) * (dt / (gravityTau + dt));
    }
    if (_grav.norm > 1e-6) _ghat = _grav.normalized;
    final ghat = _ghat;
    final lin = s.accel - _grav;

    // Vibration level and pothole shocks (the "vibration filter").
    final linNorm = lin.norm;
    _linLp += (linNorm - _linLp) * (dt / (0.5 + dt));
    final hp = linNorm - _linLp;
    _vibMs += (hp * hp - _vibMs) * (dt / (1.0 + dt));
    final shock = linNorm > shockThreshold;
    if (shock) _shockCount++;

    final stationary = _stationary.update(s.t, linNorm, s.gyro.norm);
    final health = monitor.assess(_lastFix, s.t, simulateDenied: simulateBlackout);

    _aligner.processFrame(s.accel, s.gyro, gpsSpeed: health == GnssHealth.good ? _lastFix?.speedMs : null);
    if (_aligner.displacementCount != _lastDisplacement) {
      _lastDisplacement = _aligner.displacementCount;
      _onDisplacement();
    }

    final az = phoneAzimuth(s.mag, ghat);

    // Calibration window while GNSS is healthy.
    final fix = _lastFix;
    if (health == GnssHealth.good && fix != null) {
      final course = fix.speedMs > 1.0
          ? degToRad(fix.courseDeg)
          : (_lastMovingCourseRad ?? degToRad(fix.courseDeg));
      _calibrator.add(CalSample(
        t: s.t,
        lin: lin,
        gyro: s.gyro,
        ghat: ghat,
        speed: fix.speedMs,
        courseRad: course,
        azimuthRad: az,
        fixT: fix.t,
      ));
    }

    final sinceAccepted = s.t - _lastAcceptedFixT;
    final outage = simulateBlackout || sinceAccepted > monitor.staleSeconds;

    // Aligned IMU inputs for the filter. While GNSS is healthy the raw
    // forward acceleration is used (white noise averages out in the
    // integration, and a low-pass lag would be mis-learned as a bias during
    // long accelerations); in an outage the low-passed value feeds the
    // dead-band that stops noise from random-walking the speed.
    final omega = _yawSign * s.gyro.dot(ghat);
    final turning = filter.initialized && (omega - filter.gyroBias).abs() > 0.08; // ~4.5 deg/s
    double? aFwd;
    var aRaw = 0.0;
    final fwd = _fwd;
    if (fwd != null) {
      aRaw = lin.dot(fwd).clamp(-8.0, 8.0);
      _aLp += (aRaw - _aLp) * (dt / (accelTau + dt));
      if (outage) {
        // Constant-speed prior in quiet stretches (what the IO-VNBD study
        // found inside tunnels): whatever small acceleration remains after
        // bias removal while nothing is happening is residual bias, and is
        // re-learned there so it cannot random-walk the speed. Real
        // acceleration events pass through untouched.
        final aCorr = _aLp - filter.accelBias;
        if (aCorr.abs() < quietAccelThreshold && !turning) {
          _residualBias += (aCorr - _residualBias) * (dt / (residualTau + dt));
        }
        aFwd = shock ? null : _aLp - _residualBias;
      } else {
        _residualBias = 0;
        aFwd = shock ? null : aRaw; // a shock carries no speed information
      }
    }

    if (filter.initialized) {
      // The accelerometer bias is only learned during gentle, straight
      // driving: in turns and hard accelerations any residual axis error
      // leaks proportional errors into the forward channel.
      final hardAccel = aRaw.abs() > 1.0;
      filter.predict(
        omega: omega,
        aFwd: aFwd,
        dt: dt,
        deadband: outage ? outageDeadband : 0,
        learnAccelBias: !turning && !hardAccel,
        outageSeconds: outage ? math.max(0.0, sinceAccepted - monitor.staleSeconds) : 0,
      );
      if (stationary) filter.updateZupt();
      if (useCompass && outage && az != null && _cal != null && _cal!.compassUsable) {
        filter.updateCourse(wrapPi(az + _cal!.azOffset), 0.2);
      }
    }

    _pushHistory(_fwdAccelHist, fwd == null ? 0 : _aLp);
    _pushHistory(_yawRateHist, radToDeg(omega - filter.gyroBias));

    // ── deficit handler ────────────────────────────────────────────────
    if (filter.initialized) {
      if (_mode != NavMode.deadReckoning && outage) {
        _enterBlackout(s.t);
      } else if (_mode != NavMode.deadReckoning) {
        _mode = (health == GnssHealth.degraded || filter.rejectedInRow > 0)
            ? NavMode.degraded
            : NavMode.gnssIns;
      }
    }

    // ── dead reckoning bookkeeping ─────────────────────────────────────
    if (_mode == NavMode.deadReckoning) {
      _blackoutDistance += filter.speed * dt;
      var m = _mapDr;
      if (m != null) {
        m.step(omega - filter.gyroBias, filter.speed, dt);
        if (m.deadEnd) {
          final g = roadGraph!;
          final again = MapAidedReckoner(g);
          m = again.anchor(filter.east, filter.north, filter.heading) ? again : null;
          _mapDr = m;
        }
      }
    }

    _updateDisplay();
    if (_mode == NavMode.deadReckoning && _display != null) {
      final lastT = _trail.isEmpty ? null : _trail.last;
      if (lastT == null || distanceM(lastT.lat, lastT.lon, _display!.lat, _display!.lon) >= 3.0) {
        _trail.add(_display!);
      }
    }
    _emit(s.t);
  }

  void _pushHistory(ListQueue<double> q, double v) {
    q.addLast(v);
    final max = (historySeconds * imuRate).round();
    while (q.length > max) {
      q.removeFirst();
    }
  }

  void _updateDisplay() {
    if (!filter.initialized || _lat0 == null) return;
    final m = _mapDr;
    if (_mode == NavMode.deadReckoning && m != null) {
      final p = m.latLon;
      _display = (lat: p.lat, lon: p.lon);
      _headingRad = wrapPi(m.heading);
    } else {
      final p = enuToLatLon(filter.east, filter.north, _lat0!, _lon0!);
      _display = (lat: p.lat, lon: p.lon);
      _headingRad = filter.heading;
    }
  }

  // ── blackout transitions ─────────────────────────────────────────────

  void _enterBlackout(double t) {
    _mapDr = null;
    final g = roadGraph;
    if (g != null && _lat0 != null) {
      // Filter ENU and graph ENU have different origins.
      final p = enuToLatLon(filter.east, filter.north, _lat0!, _lon0!);
      final enu = latLonToEnu(p.lat, p.lon, g.lat0, g.lon0);
      final m = MapAidedReckoner(g);
      if (m.anchor(enu.e, enu.n, filter.heading)) _mapDr = m;
    }
    _blackoutStart = t;
    _blackoutDistance = 0;
    _mode = NavMode.deadReckoning;
    _updateDisplay();
    _trail = <GeoPoint>[?_display];
  }

  void _exitBlackout(GnssFix f, double errorM) {
    _lastRecovery = RecoveryStats(
      errorM: errorM,
      distanceM: _blackoutDistance,
      seconds: f.t - _blackoutStart,
      mapAided: _mapDr != null,
    );
    _mode = NavMode.gnssIns;
    _mapDr = null;
    _trail = <GeoPoint>[];
    _updateDisplay();
  }

  // ── output ───────────────────────────────────────────────────────────

  void _emit(double t, {bool force = false}) {
    if (!force && _lastEmit >= 0 && t - _lastEmit < emitInterval) return;
    _lastEmit = t;
    final mapAided = _mode == NavMode.deadReckoning && _mapDr != null;
    GeoPoint? ghost;
    if (mapAided && _lat0 != null) {
      final p = enuToLatLon(filter.east, filter.north, _lat0!, _lon0!);
      ghost = (lat: p.lat, lon: p.lon);
    }
    final fix = _lastFix;
    final st = NavState(
      t: t,
      lat: _display?.lat ?? fix?.lat ?? 0,
      lon: _display?.lon ?? fix?.lon ?? 0,
      headingDeg: wrapDeg(radToDeg(_headingRad)),
      speedMs: filter.initialized ? filter.speed : 0,
      mode: _mode,
      gnssHealth: monitor.assess(fix, t, simulateDenied: simulateBlackout),
      hasFix: fix != null,
      fixAgeS: fix == null ? double.infinity : t - fix.t,
      fixAccuracyM: fix?.accuracyM ?? double.infinity,
      uncertaintyM: filter.initialized ? 2 * filter.posSigma : double.infinity,
      headingSigmaDeg: filter.initialized ? radToDeg(filter.headingSigma) : 180,
      speedSigmaMs: filter.initialized ? filter.speedSigma : 0,
      blackoutSeconds: _mode == NavMode.deadReckoning ? t - _blackoutStart : 0,
      blackoutDistanceM: _mode == NavMode.deadReckoning ? _blackoutDistance : 0,
      mapAided: mapAided,
      trail: List<GeoPoint>.unmodifiable(_trail),
      gnssTrack: List<GeoPoint>.unmodifiable(_gnssTrack),
      freeDrPosition: ghost,
      lastRecovery: _lastRecovery,
      alignment: _aligner.state,
      calibration: _cal,
      mount: _fwd == null ? null : MountAngles.from(up: _ghat, fwd: _fwd!),
      gyroBiasDps: radToDeg(filter.gyroBias),
      accelBiasMs2: filter.accelBias,
      fwdAccelHistory: List<double>.unmodifiable(_fwdAccelHist),
      yawRateHistory: List<double>.unmodifiable(_yawRateHist),
      vibrationRms: math.sqrt(math.max(0, _vibMs)),
      shockCount: _shockCount,
      roadSegments: roadGraph?.edges.length ?? 0,
      simulated: simulateBlackout,
      acceptedFixes: filter.accepted,
      rejectedFixes: filter.rejected,
    );
    current = st;
    if (!_ctrl.isClosed) _ctrl.add(st);
  }

  /// Phone compass azimuth (radians clockwise from magnetic north) of the
  /// phone's +Y axis, from the magnetometer and the up direction.
  static double? phoneAzimuth(Vec3 mag, Vec3 up) {
    if (mag.norm2 < 1e-6) return null;
    final mh = mag.projectOntoPlane(up);
    final yh = Vec3.unitY.projectOntoPlane(up);
    if (mh.norm < 1e-3 || yh.norm < 1e-3) return null;
    final a = mh.normalized, b = yh.normalized;
    return math.atan2(-(a.cross(b)).dot(up), a.dot(b));
  }
}

/// Conservative zero-velocity detector: quiet accelerometer and gyro for a
/// sustained period, released at once by sustained acceleration.
class _StationaryDetector {
  final ListQueue<double> _t = ListQueue<double>();
  final ListQueue<double> _lin = ListQueue<double>();
  final ListQueue<double> _gyr = ListQueue<double>();
  double _since = double.nan;

  // Tunables: real road vibration at speed keeps the |linear accel| variance
  // well above 0.03 (m/s^2)^2; engine idling at a stop sits below it on a
  // dashboard mount. Verify on the target phone before the finale.
  double window = 1.0;
  double holdSeconds = 2.0;
  double accelVarThresh = 0.03;
  double accelMeanThresh = 0.6;
  double gyroMeanThresh = 0.02;

  bool update(double t, double linNorm, double gyroNorm) {
    _t.addLast(t);
    _lin.addLast(linNorm);
    _gyr.addLast(gyroNorm);
    while (_t.isNotEmpty && t - _t.first > window) {
      _t.removeFirst();
      _lin.removeFirst();
      _gyr.removeFirst();
    }
    if (_lin.length < 5) return false;
    final lin = _lin.toList();
    final quiet = variance(lin) < accelVarThresh &&
        mean(lin) < accelMeanThresh &&
        mean(_gyr.toList()) < gyroMeanThresh;
    if (!quiet) {
      _since = double.nan;
      return false;
    }
    if (_since.isNaN) _since = t;
    return t - _since >= holdSeconds;
  }
}
