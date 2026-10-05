import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/navigation_engine.dart';
import 'package:sih_navigation/engine/vec3.dart';
import 'package:sih_navigation/services/log_service.dart';

void main() {
  test('drive log writes a header, IMU rows with the estimate, and GNSS rows', () async {
    final dir = Directory.systemTemp.createTempSync('idrlog');
    final log = LogService(directory: dir);
    final path = await log.start();
    expect(log.active, isTrue);

    final engine = NavigationEngine();
    const imu = ImuSample(t: 1000.0, accel: Vec3(0, 0, 9.81), gyro: Vec3.zero, linear: Vec3.zero, mag: Vec3(20, 5, -40));
    engine.onImu(imu);
    log.imu(imu, state: engine.current);
    log.imu(const ImuSample(t: 1000.05, accel: Vec3(0.1, 0, 9.8), gyro: Vec3(0, 0, 0.01)));
    log.gnss(const GnssFix(t: 1000.1, lat: 28.6139, lon: 77.209, speedMs: 3.2, courseDeg: 91.5, accuracyM: 4));
    await log.stop();
    expect(log.active, isFalse);

    final lines = File(path).readAsLinesSync();
    expect(lines.length, 4);
    expect(lines.first, LogService.header);
    expect(lines[1].split(',').length, LogService.header.split(',').length);
    expect(lines[1], startsWith('imu,1000.000,0.0000,0.0000,9.8100'));
    expect(lines[1], contains('degraded')); // engine state before any fix
    expect(lines[2], startsWith('imu,1000.050'));
    expect(lines[3], startsWith('gnss,1000.100'));
    expect(lines[3], contains('28.6139000,77.2090000,3.200,91.50,4.0'));
    dir.deleteSync(recursive: true);
  });
}
