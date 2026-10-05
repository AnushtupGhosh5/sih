import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../engine/navigation_engine.dart';
import '../engine/vec3.dart';
import '../services/location_service.dart';
import '../services/log_service.dart';
import '../services/osm_service.dart';
import '../services/sensor_service.dart';
import '../utils/theme.dart';
import '../widgets/diagnostics_sheet.dart';
import '../widgets/drift_counter.dart';
import '../widgets/engine_status_bar.dart';
import '../widgets/hud_map.dart';
import '../widgets/mode_banner.dart';
import '../widgets/telemetry_panel.dart';

/// Main HUD dashboard screen — feeds phone sensors and GNSS into the
/// on-device IDR engine and renders its solution.
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // ── Services ──────────────────────────────────────────────────────────
  final LocationService _locationService = LocationService();
  final SensorService _sensorService = SensorService();
  final NavigationEngine _engine = NavigationEngine();
  final OsmService _osm = OsmService();
  final LogService _log = LogService();

  // ── UI state ─────────────────────────────────────────────────────────
  final ValueNotifier<NavState?> _nav = ValueNotifier<NavState?>(null);
  bool _mapLoading = false;
  bool _follow = true;
  bool _logging = false;

  static const LatLng _defaultPosition = LatLng(28.6139, 77.2090); // New Delhi

  StreamSubscription<Position>? _posSub;
  StreamSubscription<SensorSnapshot>? _sensorSub;
  StreamSubscription<NavState>? _navSub;

  @override
  void initState() {
    super.initState();
    _navSub = _engine.states.listen((s) {
      if (!mounted) return;
      _nav.value = s;
      setState(() {});
    });
    _locationService.startListening();
    _posSub = _locationService.positionStream.listen(_onPosition);
    _sensorService.startListening();
    _sensorSub = _sensorService.snapshotStream.listen(_onSensor);
  }

  void _onPosition(Position pos) {
    final speed = pos.speed.isNaN ? 0.0 : math.max(0.0, pos.speed);
    final course = pos.heading.isNaN ? 0.0 : pos.heading;
    final fix = GnssFix(
      t: DateTime.now().millisecondsSinceEpoch / 1000.0,
      lat: pos.latitude,
      lon: pos.longitude,
      speedMs: speed,
      courseDeg: course,
      accuracyM: pos.accuracy.isNaN ? 999 : pos.accuracy,
    );
    _engine.onGnss(fix);
    _log.gnss(fix);
    _refreshRoads(pos.latitude, pos.longitude);
  }

  Future<void> _refreshRoads(double lat, double lon) async {
    if (_osm.busy) return;
    final before = _osm.graph;
    if (mounted && before == null) setState(() => _mapLoading = true);
    final g = await _osm.ensureCoverage(lat, lon);
    if (!mounted) return;
    setState(() {
      _mapLoading = _osm.busy;
      if (g != null && !identical(g, before)) _engine.roadGraph = g;
    });
  }

  void _onSensor(SensorSnapshot snap) {
    final sample = ImuSample(
      t: snap.tSeconds,
      accel: Vec3(snap.accelX, snap.accelY, snap.accelZ),
      linear: Vec3(snap.linX, snap.linY, snap.linZ),
      gyro: Vec3(snap.gyroX, snap.gyroY, snap.gyroZ),
      mag: Vec3(snap.magX, snap.magY, snap.magZ),
    );
    _engine.onImu(sample);
    _log.imu(sample, state: _engine.current);
  }

  void _setSimulate(bool v) => setState(() => _engine.simulateBlackout = v);

  Future<void> _setLogging(bool v) async {
    if (v) {
      await _log.start();
    } else {
      await _log.stop();
    }
    if (mounted) setState(() => _logging = v);
  }

  void _openDiagnostics() {
    DiagnosticsSheet.show(
      context,
      DiagnosticsSheet(
        state: _nav,
        simulate: _engine.simulateBlackout,
        logging: _logging,
        logPath: _log.path,
        useCompass: _engine.useCompass,
        follow: _follow,
        mapError: _osm.lastError,
        onSimulate: _setSimulate,
        onLogging: _setLogging,
        onCompass: (v) => setState(() => _engine.useCompass = v),
        onFollow: (v) => setState(() => _follow = v),
      ),
    );
  }

  @override
  void dispose() {
    _posSub?.cancel();
    _sensorSub?.cancel();
    _navSub?.cancel();
    _locationService.dispose();
    _sensorService.dispose();
    _engine.dispose();
    _log.stop();
    _nav.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final nav = _nav.value;
    final hasFix = nav?.hasFix ?? false;
    final position = hasFix ? LatLng(nav!.lat, nav.lon) : _defaultPosition;
    final heading = nav?.headingDeg ?? 0.0;
    final mode = nav?.mode ?? NavMode.degraded;
    final isDr = mode == NavMode.deadReckoning;
    final trail = nav == null ? const <LatLng>[] : [for (final p in nav.trail) LatLng(p.lat, p.lon)];
    final gnssTrack = nav == null ? const <LatLng>[] : [for (final p in nav.gnssTrack) LatLng(p.lat, p.lon)];
    final ghostPt = nav?.freeDrPosition;
    final ghost = ghostPt == null ? null : LatLng(ghostPt.lat, ghostPt.lon);
    final padding = MediaQuery.of(context).padding;

    return Scaffold(
      backgroundColor: HudTheme.background,
      body: Stack(
        children: [
          Positioned.fill(
            child: HudMap(
              position: position,
              heading: heading,
              mode: mode,
              trail: trail,
              gnssTrack: gnssTrack,
              ghost: ghost,
              uncertaintyM: nav?.uncertaintyM,
              follow: _follow,
              onUserGesture: () {
                if (_follow) setState(() => _follow = false);
              },
            ),
          ),

          // ── Mode banner (top) ─────────────────────────────────
          Positioned(
            top: padding.top + 16,
            left: 0,
            right: 0,
            child: Center(child: ModeBanner(mode: mode, acquiring: !hasFix)),
          ),

          // ── Blackout readout (below banner) ───────────────────
          Positioned(
            top: padding.top + 76,
            left: 0,
            right: 0,
            child: Center(
              child: DriftCounter(
                visible: isDr,
                distanceM: nav?.blackoutDistanceM ?? 0,
                elapsedSeconds: nav?.blackoutSeconds ?? 0,
                mapAided: nav?.mapAided ?? false,
              ),
            ),
          ),

          // ── Re-centre button (when the user panned away) ──────
          if (!_follow)
            Positioned(
              right: 16,
              top: padding.top + 130,
              child: GestureDetector(
                onTap: () => setState(() => _follow = true),
                child: HudTheme.glassWrap(
                  padding: const EdgeInsets.all(12),
                  borderRadius: 20,
                  backgroundColor: HudTheme.surface.withValues(alpha: 0.75),
                  child: const Icon(Icons.my_location, size: 18, color: HudTheme.textPrimary),
                ),
              ),
            ),

          // ── Engine status + telemetry (bottom) ────────────────
          Positioned(
            bottom: padding.bottom + 20,
            left: 16,
            right: 16,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                EngineStatusBar(
                  state: nav,
                  simulate: _engine.simulateBlackout,
                  logging: _logging,
                  mapLoading: _mapLoading,
                  onToggleSimulate: () => _setSimulate(!_engine.simulateBlackout),
                  onOpenDiagnostics: _openDiagnostics,
                ),
                const SizedBox(height: 10),
                TelemetryPanel(state: nav),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
