import 'dart:async';
import 'dart:collection';
import 'dart:math' as math;

import 'alignment.dart';
import 'calibration.dart';
import 'dead_reckoning.dart';
import 'geo.dart';
import 'gnss_monitor.dart';
import 'map_matcher.dart';
import 'road_graph.dart';
import 'signal.dart';
import 'vec3.dart';

export 'alignment.dart' show AlignmentState;
export 'calibration.dart' show Calibration;
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
  final Vec3 accel; // m/s^2, includes gravity
  final Vec3 linear; // m/s^2, gravity removed (Android linear acceleration); zero if unavailable
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
    required this.blackoutSeconds,
    required this.blackoutDistanceM,
    required this.mapAided,
    required this.alignment,
    required this.calibration,
    required this.trail,
    required this.freeDrPosition,
    required this.lastRecovery,
    required this.roadSegments,
    required this.simulated,
    required this.hasFix,
  });

  final double t;
  final double lat, lon;
  final double headingDeg;
  final double speedMs;
  final NavMode mode;
  final GnssHealth gnssHealth;
  final double blackoutSeconds;
  final double blackoutDistanceM;
  final bool mapAided;
  final AlignmentState alignment;
  final Calibration? calibration;
  final List<({double lat, double lon})> trail;
  final ({double lat, double lon})? freeDrPosition;
  final RecoveryStats? lastRecovery;
  final int roadSegments;
  final bool simulated;
  final bool hasFix;

  bool get isDeadReckoning => mode == NavMode.deadReckoning;
}

/// Orchestrates alignment, calibration, the GNSS deficit handler, free and
/// map-aided dead reckoning. Feed it IMU samples and GNSS fixes; read
/// [states].
class NavigationEngine {
  NavigationEngine({
    GnssMonitor? monitor,
    this.goodFixesToRecover = 2,
    this.emitInterval = 0.1,
    this.gravityTau = 1.0,
    this.imuRate = 20.0,
    this.compassGain = 0.0,
  })  : monitor = monitor ?? GnssMonitor(),
        _aligner = InVehicleAligner(samplingRate: imuRate),
        _calibrator = BlackoutCalibrator(windowSeconds: 30, minSeconds: 8, fs: 10);

  final GnssMonitor monitor;
  final int goodFixesToRecover;
  final double emitInterval;
  final double gravityTau;
  final double imuRate;

  /// Weak compass pull applied to the free-inertial heading when the compass
  /// offset calibrated cleanly. Off by default: in-vehicle and in-tunnel
  /// magnetometers are unreliable (steel, electrics), and the desktop results
  /// were obtained with gyro-only heading.
  final double compassGain;

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
  GnssFix? _lastGoodFix;
  double? _lastMovingCourseRad;
  int _goodStreak = 0;
  int _lastDisplacement = 0;

  // IMU bookkeeping
  double? _lastImuT;
  Vec3 _grav = Vec3.zero;
  bool _gravInit = false;
  Vec3 _ghat = Vec3.unitZ;

  // Solution
  NavMode _mode = NavMode.degraded;
  double _lat = 0, _lon = 0, _headingRad = 0, _speed = 0;
  Calibration? _cal;
  DeadReckoner? _shadow; // propagates from the last good fix while GNSS is healthy
  DeadReckoner? _freeDr;
  MapAidedReckoner? _mapDr;
  double _blackoutStart = 0;
  List<({double lat, double lon})> _trail = <({double lat, double lon})>[];
  RecoveryStats? _lastRecovery;
  double _lastEmit = -1;

  NavMode get mode => _mode;
  Calibration? get calibration => _cal;

  void dispose() => _ctrl.close();

  // ── GNSS ─────────────────────────────────────────────────────────────

  void onGnss(GnssFix f) {
    _lastFix = f;
    final health = monitor.assess(f, f.t, simulateDenied: simulateBlackout);
    if (health == GnssHealth.good) {
      _goodStreak++;
      _lastGoodFix = f;
      if (f.speedMs > 1.0) _lastMovingCourseRad = degToRad(f.courseDeg);
      if (_mode == NavMode.deadReckoning && _goodStreak >= goodFixesToRecover) {
        _exitBlackout(f);
      }
    } else {
      _goodStreak = 0;
    }

    if (_mode != NavMode.deadReckoning) {
      _lat = f.lat;
      _lon = f.lon;
      _speed = f.speedMs;
      if (f.speedMs > 1.0) _headingRad = degToRad(f.courseDeg);
      if (health == GnssHealth.good) {
        _mode = NavMode.gnssIns;
        _cal = _calibrator.compute() ?? Calibration.fallback(_ghat);
        _shadow = _seedReckoner(f, _cal!);
      } else if (health == GnssHealth.degraded) {
        _mode = NavMode.degraded;
      }
    }
    _emit(f.t, force: true);
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

    final stationary = _stationary.update(s.t, lin.norm, s.gyro.norm);
    final health = monitor.assess(_lastFix, s.t, simulateDenied: simulateBlackout);

    _aligner.processFrame(s.accel, s.gyro,
        gpsSpeed: health == GnssHealth.good ? _lastFix?.speedMs : null);
    if (_aligner.displacementCount != _lastDisplacement) {
      _lastDisplacement = _aligner.displacementCount;
      _calibrator.clear(); // phone moved on the mount: old axes are invalid
    }

    final az = phoneAzimuth(s.mag, ghat);

    if (_mode != NavMode.deadReckoning) {
      if (health == GnssHealth.denied) {
        _enterBlackout(s.t, ghat);
      } else {
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
          _mode = NavMode.gnssIns;
        } else {
          _mode = NavMode.degraded;
        }
        final sh = _shadow;
        if (sh != null) {
          sh.step(lin: lin, gyro: s.gyro, ghat: ghat, dt: dt, stationary: stationary);
        }
      }
    }

    if (_mode == NavMode.deadReckoning) {
      final dr = _freeDr!;
      final cal = dr.cal;
      final compass = (az != null && cal.compassUsable) ? az + cal.azOffset : null;
      dr.step(
        lin: lin,
        gyro: s.gyro,
        ghat: ghat,
        dt: dt,
        compassHeading: compass,
        stationary: stationary,
      );
      var m = _mapDr;
      if (m != null) {
        m.step(dr.yawRate(s.gyro, ghat), dr.v, dt);
        if (m.deadEnd) {
          // Ran off the downloaded road network (or a genuine cul-de-sac):
          // try to re-anchor at the free-inertial position, else continue
          // free inertial.
          final g = roadGraph!;
          final fp = dr.position;
          final enu = latLonToEnu(fp.lat, fp.lon, g.lat0, g.lon0);
          final again = MapAidedReckoner(g);
          m = again.anchor(enu.e, enu.n, dr.heading) ? again : null;
          _mapDr = m;
        }
      }
      if (m != null) {
        final p = m.latLon;
        _lat = p.lat;
        _lon = p.lon;
        _headingRad = wrapPi(m.heading);
      } else {
        final p = dr.position;
        _lat = p.lat;
        _lon = p.lon;
        _headingRad = dr.heading;
      }
      _speed = dr.v;
      final lastT = _trail.isEmpty ? null : _trail.last;
      if (lastT == null || distanceM(lastT.lat, lastT.lon, _lat, _lon) >= 3.0) {
        _trail.add((lat: _lat, lon: _lon));
      }
    }

    _emit(s.t);
  }

  // ── blackout transitions ─────────────────────────────────────────────

  DeadReckoner _seedReckoner(GnssFix f, Calibration cal) {
    final h0 = f.speedMs > 1.0 ? degToRad(f.courseDeg) : (_lastMovingCourseRad ?? _headingRad);
    // GNSS speed jitters by a few tenths of a m/s at rest; a reckoner seeded
    // at rest should start from exactly zero.
    final v0 = f.speedMs < 0.5 ? 0.0 : f.speedMs;
    return DeadReckoner(
      lat0: f.lat,
      lon0: f.lon,
      v0: v0,
      h0: h0,
      cal: cal,
      compassGain: cal.compassUsable ? compassGain : 0.0,
    );
  }

  void _enterBlackout(double t, Vec3 ghat) {
    final fix = _lastGoodFix;
    if (fix == null) {
      _mode = NavMode.degraded; // nothing to seed from yet
      return;
    }
    final cal = _cal ?? (_calibrator.compute() ?? Calibration.fallback(ghat));
    _cal = cal;
    // The shadow reckoner has been propagating since the last good fix, so it
    // already covers the seconds between that fix and the detection of the
    // outage. Continue from it.
    final dr = _shadow ?? _seedReckoner(fix, cal);
    _freeDr = dr;
    _shadow = null;

    _mapDr = null;
    final g = roadGraph;
    if (g != null) {
      final p0 = dr.position;
      final enu = latLonToEnu(p0.lat, p0.lon, g.lat0, g.lon0);
      final m = MapAidedReckoner(g);
      if (m.anchor(enu.e, enu.n, dr.heading)) _mapDr = m;
    }

    final p = _mapDr?.latLon ?? dr.position;
    _lat = p.lat;
    _lon = p.lon;
    _headingRad = dr.heading;
    _speed = dr.v;
    _blackoutStart = t;
    _trail = <({double lat, double lon})>[(lat: _lat, lon: _lon)];
    _mode = NavMode.deadReckoning;
    _goodStreak = 0;
  }

  void _exitBlackout(GnssFix f) {
    final dr = _freeDr;
    final err = distanceM(_lat, _lon, f.lat, f.lon);
    _lastRecovery = RecoveryStats(
      errorM: err,
      distanceM: dr?.distance ?? 0,
      seconds: f.t - _blackoutStart,
      mapAided: _mapDr != null,
    );
    _mode = NavMode.gnssIns;
    _freeDr = null;
    _mapDr = null;
    _trail = <({double lat, double lon})>[];
  }

  // ── output ───────────────────────────────────────────────────────────

  void _emit(double t, {bool force = false}) {
    if (!force && _lastEmit >= 0 && t - _lastEmit < emitInterval) return;
    _lastEmit = t;
    final mapAided = _mapDr != null;
    ({double lat, double lon})? ghost;
    if (mapAided) ghost = _freeDr!.position;
    final st = NavState(
      t: t,
      lat: _lat,
      lon: _lon,
      headingDeg: wrapDeg(radToDeg(_headingRad)),
      speedMs: _speed,
      mode: _mode,
      gnssHealth: monitor.assess(_lastFix, t, simulateDenied: simulateBlackout),
      blackoutSeconds: _mode == NavMode.deadReckoning ? t - _blackoutStart : 0,
      blackoutDistanceM: _mode == NavMode.deadReckoning ? (_freeDr?.distance ?? 0) : 0,
      mapAided: mapAided,
      alignment: _aligner.state,
      calibration: _cal,
      trail: List<({double lat, double lon})>.unmodifiable(_trail),
      freeDrPosition: ghost,
      lastRecovery: _lastRecovery,
      roadSegments: roadGraph?.edges.length ?? 0,
      simulated: simulateBlackout,
      hasFix: _lastFix != null,
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
/// sustained period (thresholds from the NHC/ZUPT module).
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
  double accelMeanThresh = 0.6; // sustained |linear accel| above this = pulling away
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
    // Variance alone misses a smooth pull-away (low vibration at low speed);
    // the mean catches the sustained acceleration and releases the ZUPT.
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
