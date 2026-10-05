import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../engine/navigation_engine.dart';

/// Records raw sensors, GNSS fixes and the engine's solution to a CSV file so
/// drives can be replayed through the desktop pipeline (own-data evaluation
/// with simulated blackouts) and the thresholds tuned offline.
///
/// Columns: kind,t,ax,ay,az,lx,ly,lz,gx,gy,gz,mx,my,mz,lat,lon,speed,course,acc,mode,est_lat,est_lon,est_v,est_h
/// `kind` is `imu` or `gnss`; unused columns are left empty.
class LogService {
  LogService({Directory? directory}) : _dirOverride = directory;

  static const String header =
      'kind,t,ax,ay,az,lx,ly,lz,gx,gy,gz,mx,my,mz,lat,lon,speed,course,acc,mode,est_lat,est_lon,est_v,est_h';

  final Directory? _dirOverride;
  File? _file;
  IOSink? _sink;
  int rows = 0;

  bool get active => _sink != null;
  String? get path => _file?.path;

  Future<String> start() async {
    await stop();
    Directory base;
    if (_dirOverride != null) {
      base = _dirOverride;
    } else {
      base = await getExternalStorageDirectory() ?? await getApplicationDocumentsDirectory();
    }
    final dir = Directory('${base.path}/idr_logs');
    await dir.create(recursive: true);
    final stamp = DateTime.now().toIso8601String().replaceAll(':', '-').split('.').first;
    _file = File('${dir.path}/drive_$stamp.csv');
    _sink = _file!.openWrite();
    _sink!.writeln(header);
    rows = 0;
    return _file!.path;
  }

  void imu(ImuSample s, {NavState? state}) {
    final sink = _sink;
    if (sink == null) return;
    final est = state == null
        ? ',,,,'
        : '${state.mode.name},${state.lat.toStringAsFixed(7)},${state.lon.toStringAsFixed(7)},'
            '${state.speedMs.toStringAsFixed(3)},${state.headingDeg.toStringAsFixed(2)}';
    sink.writeln('imu,${s.t.toStringAsFixed(3)},'
        '${_f(s.accel.x)},${_f(s.accel.y)},${_f(s.accel.z)},'
        '${_f(s.linear.x)},${_f(s.linear.y)},${_f(s.linear.z)},'
        '${_f(s.gyro.x)},${_f(s.gyro.y)},${_f(s.gyro.z)},'
        '${_f(s.mag.x)},${_f(s.mag.y)},${_f(s.mag.z)},'
        ',,,,,$est');
    rows++;
  }

  void gnss(GnssFix f) {
    final sink = _sink;
    if (sink == null) return;
    sink.writeln('gnss,${f.t.toStringAsFixed(3)},,,,,,,,,,,,,'
        '${f.lat.toStringAsFixed(7)},${f.lon.toStringAsFixed(7)},${f.speedMs.toStringAsFixed(3)},'
        '${f.courseDeg.toStringAsFixed(2)},${f.accuracyM.toStringAsFixed(1)},,,,,');
    rows++;
  }

  Future<void> stop() async {
    final sink = _sink;
    _sink = null;
    if (sink != null) {
      await sink.flush();
      await sink.close();
    }
  }

  static String _f(double v) => v.toStringAsFixed(4);
}
