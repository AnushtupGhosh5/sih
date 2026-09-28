import 'dart:async';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import '../services/location_service.dart';
import '../services/sensor_service.dart';
import '../utils/theme.dart';
import '../widgets/drift_counter.dart';
import '../widgets/hud_map.dart';
import '../widgets/mode_banner.dart';
import '../widgets/telemetry_panel.dart';

/// Main HUD dashboard screen — orchestrates map, sensors, and mode switching.
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // ── Services ──────────────────────────────────────────────────────────
  final LocationService _locationService = LocationService();
  final SensorService _sensorService = SensorService();

  // ── Position & motion state ──────────────────────────────────────────
  LatLng _position = const LatLng(28.6139, 77.2090); // default: New Delhi
  double _heading = 0;
  double _speedKmh = 0;
  double _gpsAccuracy = 999;

  // ── Mode switching with 1.5s debounce ────────────────────────────────
  bool _isGnssMode = true; // displayed mode (debounced)
  bool _rawGnssGood = true; // instantaneous GPS quality
  Timer? _modeSwitchTimer;

  // ── Dead Reckoning drift counter ─────────────────────────────────────
  DateTime? _drStartTime;
  double _drElapsedSeconds = 0;
  Timer? _driftTickTimer;

  // ── Sensor sparkline data ────────────────────────────────────────────
  List<double> _accelHistory = [];
  List<double> _gyroHistory = [];

  // ── Stream subscriptions ─────────────────────────────────────────────
  StreamSubscription<Position>? _posSub;
  StreamSubscription<SensorSnapshot>? _sensorSub;

  @override
  void initState() {
    super.initState();
    _startServices();
  }

  void _startServices() {
    // Location.
    _locationService.startListening();
    _posSub = _locationService.positionStream.listen(_onPosition);

    // Sensors.
    _sensorService.startListening();
    _sensorSub = _sensorService.snapshotStream.listen(_onSensor);
  }

  void _onPosition(Position pos) {
    final newGnssGood = pos.accuracy <= HudTheme.gpsAccuracyThreshold;

    setState(() {
      _position = LatLng(pos.latitude, pos.longitude);
      _heading = pos.heading;
      _speedKmh = pos.speed * 3.6; // m/s → km/h
      _gpsAccuracy = pos.accuracy;
    });

    // Debounced mode switching.
    if (newGnssGood != _rawGnssGood) {
      _rawGnssGood = newGnssGood;
      _modeSwitchTimer?.cancel();
      _modeSwitchTimer = Timer(HudTheme.modeDebounceDuration, () {
        if (!mounted) return;
        setState(() {
          _isGnssMode = _rawGnssGood;
          if (!_isGnssMode) {
            // Entering DR mode.
            _drStartTime = DateTime.now();
            _drElapsedSeconds = 0;
            _startDriftTicker();
          } else {
            // Returning to GNSS mode.
            _stopDriftTicker();
            _drStartTime = null;
            _drElapsedSeconds = 0;
          }
        });
      });
    }
  }

  void _onSensor(SensorSnapshot snap) {
    setState(() {
      _accelHistory = List.from(_sensorService.accelHistory);
      _gyroHistory = List.from(_sensorService.gyroHistory);
    });
  }

  // ── Drift ticker ─────────────────────────────────────────────────────
  void _startDriftTicker() {
    _driftTickTimer?.cancel();
    _driftTickTimer = Timer.periodic(const Duration(milliseconds: 200), (_) {
      if (!mounted || _drStartTime == null) return;
      setState(() {
        _drElapsedSeconds =
            DateTime.now().difference(_drStartTime!).inMilliseconds / 1000.0;
      });
    });
  }

  void _stopDriftTicker() {
    _driftTickTimer?.cancel();
    _driftTickTimer = null;
  }

  @override
  void dispose() {
    _modeSwitchTimer?.cancel();
    _driftTickTimer?.cancel();
    _posSub?.cancel();
    _sensorSub?.cancel();
    _locationService.dispose();
    _sensorService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: HudTheme.background,
      body: Stack(
        children: [
          // ── Full-screen map ────────────────────────────────────
          Positioned.fill(
            child: HudMap(
              position: _position,
              heading: _heading,
              isGnssMode: _isGnssMode,
            ),
          ),

          // ── Mode banner (top) ─────────────────────────────────
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 0,
            right: 0,
            child: Center(
              child: ModeBanner(isGnssMode: _isGnssMode),
            ),
          ),

          // ── Drift counter (below banner) ──────────────────────
          Positioned(
            top: MediaQuery.of(context).padding.top + 76,
            left: 0,
            right: 0,
            child: Center(
              child: DriftCounter(
                visible: !_isGnssMode,
                elapsedSeconds: _drElapsedSeconds,
              ),
            ),
          ),

          // ── Telemetry panel (bottom card) ─────────────────────
          Positioned(
            bottom: MediaQuery.of(context).padding.bottom + 24,
            left: 16,
            right: 16,
            child: TelemetryPanel(
              speedKmh: _speedKmh,
              heading: _heading,
              gpsAccuracy: _gpsAccuracy,
              isGnssMode: _isGnssMode,
              accelHistory: _accelHistory,
              gyroHistory: _gyroHistory,
            ),
          ),
        ],
      ),
    );
  }
}
