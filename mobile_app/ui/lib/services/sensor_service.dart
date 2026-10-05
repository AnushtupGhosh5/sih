import 'dart:async';
import 'dart:math';
import 'package:sensors_plus/sensors_plus.dart';

/// Data class holding a single snapshot of all sensor readings.
class SensorSnapshot {
  final double accelX, accelY, accelZ, accelMag;
  final double linX, linY, linZ; // linear acceleration (gravity removed)
  final double gyroX, gyroY, gyroZ, gyroMag;
  final double magX, magY, magZ, magMag;
  final DateTime? timestamp;

  const SensorSnapshot({
    required this.accelX,
    required this.accelY,
    required this.accelZ,
    required this.accelMag,
    required this.gyroX,
    required this.gyroY,
    required this.gyroZ,
    required this.gyroMag,
    required this.magX,
    required this.magY,
    required this.magZ,
    required this.magMag,
    this.linX = 0,
    this.linY = 0,
    this.linZ = 0,
    this.timestamp,
  });

  static const SensorSnapshot zero = SensorSnapshot(
    accelX: 0, accelY: 0, accelZ: 0, accelMag: 0,
    gyroX: 0, gyroY: 0, gyroZ: 0, gyroMag: 0,
    magX: 0, magY: 0, magZ: 0, magMag: 0,
    timestamp: null,
  );

  /// Seconds since the Unix epoch (engine time base).
  double get tSeconds => (timestamp ?? DateTime.now()).millisecondsSinceEpoch / 1000.0;
}

/// Manages IMU sensor streams and maintains rolling buffers for sparklines.
class SensorService {
  static const int bufferSize = 100; // ~5 seconds at 20Hz
  static const Duration samplingPeriod = Duration(milliseconds: 50); // 20 Hz

  // Latest raw values (updated by individual streams).
  double _ax = 0, _ay = 0, _az = 0;
  double _lx = 0, _ly = 0, _lz = 0;
  double _gx = 0, _gy = 0, _gz = 0;
  double _mx = 0, _my = 0, _mz = 0;

  // Rolling magnitude buffers for sparkline charts.
  final List<double> accelHistory = [];
  final List<double> gyroHistory = [];

  StreamSubscription? _accelSub;
  StreamSubscription? _linSub;
  StreamSubscription? _gyroSub;
  StreamSubscription? _magSub;

  final StreamController<SensorSnapshot> _controller =
      StreamController<SensorSnapshot>.broadcast();

  /// Broadcast stream of merged sensor snapshots (one per accelerometer event).
  Stream<SensorSnapshot> get snapshotStream => _controller.stream;

  /// The most recent snapshot for synchronous reads.
  SensorSnapshot get latest => _lastSnapshot;
  SensorSnapshot _lastSnapshot = SensorSnapshot.zero;

  /// Start all sensor subscriptions.
  void startListening() {
    _accelSub = accelerometerEventStream(samplingPeriod: samplingPeriod)
        .listen((e) {
      _ax = e.x;
      _ay = e.y;
      _az = e.z;
      _pushSnapshot();
    });

    // Linear acceleration (Android removes gravity for us). Optional: if the
    // device has no such sensor the engine falls back to a low-pass estimate.
    _linSub = userAccelerometerEventStream(samplingPeriod: samplingPeriod)
        .listen((e) {
      _lx = e.x;
      _ly = e.y;
      _lz = e.z;
    }, onError: (_) {});

    _gyroSub =
        gyroscopeEventStream(samplingPeriod: samplingPeriod).listen((e) {
      _gx = e.x;
      _gy = e.y;
      _gz = e.z;
    }, onError: (_) {});

    _magSub = magnetometerEventStream(samplingPeriod: samplingPeriod)
        .listen((e) {
      _mx = e.x;
      _my = e.y;
      _mz = e.z;
    }, onError: (_) {});
  }

  void _pushSnapshot() {
    final accelMag = sqrt(_ax * _ax + _ay * _ay + _az * _az);
    final gyroMag = sqrt(_gx * _gx + _gy * _gy + _gz * _gz);
    final magMag = sqrt(_mx * _mx + _my * _my + _mz * _mz);

    // Update rolling buffers.
    accelHistory.add(accelMag);
    if (accelHistory.length > bufferSize) accelHistory.removeAt(0);

    gyroHistory.add(gyroMag);
    if (gyroHistory.length > bufferSize) gyroHistory.removeAt(0);

    _lastSnapshot = SensorSnapshot(
      accelX: _ax, accelY: _ay, accelZ: _az, accelMag: accelMag,
      linX: _lx, linY: _ly, linZ: _lz,
      gyroX: _gx, gyroY: _gy, gyroZ: _gz, gyroMag: gyroMag,
      magX: _mx, magY: _my, magZ: _mz, magMag: magMag,
      timestamp: DateTime.now(),
    );

    if (!_controller.isClosed) {
      _controller.add(_lastSnapshot);
    }
  }

  void stopListening() {
    _accelSub?.cancel();
    _linSub?.cancel();
    _gyroSub?.cancel();
    _magSub?.cancel();
    _accelSub = null;
    _linSub = null;
    _gyroSub = null;
    _magSub = null;
  }

  void dispose() {
    stopListening();
    _controller.close();
  }
}
