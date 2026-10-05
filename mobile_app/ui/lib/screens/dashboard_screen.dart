import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../engine/navigation_engine.dart';
import '../engine/vec3.dart';
import '../services/location_service.dart';
import '../services/osm_service.dart';
import '../services/sensor_service.dart';
import '../utils/theme.dart';
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

  // ── UI state ─────────────────────────────────────────────────────────
  NavState? _nav;
  double _gpsAccuracy = 999;
  List<double> _accelHistory = [];
  List<double> _gyroHistory = [];
  int _sensorTick = 0;
  bool _mapLoading = false;

  static const LatLng _defaultPosition = LatLng(28.6139, 77.2090); // New Delhi

  // ── Stream subscriptions ─────────────────────────────────────────────
  StreamSubscription<Position>? _posSub;
  StreamSubscription<SensorSnapshot>? _sensorSub;
  StreamSubscription<NavState>? _navSub;

  @override
  void initState() {
    super.initState();
    _navSub = _engine.states.listen((s) {
      if (mounted) setState(() => _nav = s);
    });
    _locationService.startListening();
    _posSub = _locationService.positionStream.listen(_onPosition);
    _sensorService.startListening();
    _sensorSub = _sensorService.snapshotStream.listen(_onSensor);
  }

  void _onPosition(Position pos) {
    _gpsAccuracy = pos.accuracy;
    final speed = pos.speed.isNaN ? 0.0 : math.max(0.0, pos.speed);
    final course = pos.heading.isNaN ? 0.0 : pos.heading;
    _engine.onGnss(GnssFix(
      t: DateTime.now().millisecondsSinceEpoch / 1000.0,
      lat: pos.latitude,
      lon: pos.longitude,
      speedMs: speed,
      courseDeg: course,
      accuracyM: pos.accuracy,
    ));
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
    _engine.onImu(ImuSample(
      t: snap.tSeconds,
      accel: Vec3(snap.accelX, snap.accelY, snap.accelZ),
      linear: Vec3(snap.linX, snap.linY, snap.linZ),
      gyro: Vec3(snap.gyroX, snap.gyroY, snap.gyroZ),
      mag: Vec3(snap.magX, snap.magY, snap.magZ),
    ));
    // Sparklines at 5 Hz is plenty.
    if (++_sensorTick % 4 == 0 && mounted) {
      setState(() {
        _accelHistory = List.from(_sensorService.accelHistory);
        _gyroHistory = List.from(_sensorService.gyroHistory);
      });
    }
  }

  void _toggleSimulate() {
    setState(() => _engine.simulateBlackout = !_engine.simulateBlackout);
  }

  @override
  void dispose() {
    _posSub?.cancel();
    _sensorSub?.cancel();
    _navSub?.cancel();
    _locationService.dispose();
    _sensorService.dispose();
    _engine.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final nav = _nav;
    final hasFix = nav?.hasFix ?? false;
    final position = hasFix ? LatLng(nav!.lat, nav.lon) : _defaultPosition;
    final heading = nav?.headingDeg ?? 0.0;
    final isDr = nav?.isDeadReckoning ?? false;
    final trail = nav == null
        ? const <LatLng>[]
        : [for (final p in nav.trail) LatLng(p.lat, p.lon)];
    final ghostPt = nav?.freeDrPosition;
    final ghost = ghostPt == null ? null : LatLng(ghostPt.lat, ghostPt.lon);
    final padding = MediaQuery.of(context).padding;

    return Scaffold(
      backgroundColor: HudTheme.background,
      body: Stack(
        children: [
          // ── Full-screen map ────────────────────────────────────
          Positioned.fill(
            child: HudMap(
              position: position,
              heading: heading,
              isGnssMode: !isDr,
              trail: trail,
              ghost: ghost,
            ),
          ),

          // ── Mode banner (top) ─────────────────────────────────
          Positioned(
            top: padding.top + 16,
            left: 0,
            right: 0,
            child: Center(child: ModeBanner(isGnssMode: !isDr)),
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

          // ── Engine status + telemetry (bottom) ────────────────
          Positioned(
            bottom: padding.bottom + 24,
            left: 16,
            right: 16,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                EngineStatusBar(
                  state: nav,
                  simulate: _engine.simulateBlackout,
                  mapLoading: _mapLoading,
                  onToggleSimulate: _toggleSimulate,
                ),
                const SizedBox(height: 10),
                TelemetryPanel(
                  speedKmh: (nav?.speedMs ?? 0) * 3.6,
                  heading: heading,
                  gpsAccuracy: _gpsAccuracy,
                  isGnssMode: !isDr,
                  accelHistory: _accelHistory,
                  gyroHistory: _gyroHistory,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
