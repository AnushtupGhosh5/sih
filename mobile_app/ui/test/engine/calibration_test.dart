import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/calibration.dart';
import 'package:sih_navigation/engine/vec3.dart';

void main() {
  test('mount angles are recovered exactly from the up and forward axes', () {
    for (final (p, r, y) in [(30.0, 10.0, 45.0), (-20.0, 5.0, -120.0), (0.0, 0.0, 0.0), (60.0, -40.0, 170.0)]) {
      final mount = Mat3.fromEuler(pitch: p * math.pi / 180, roll: r * math.pi / 180, yaw: y * math.pi / 180);
      final m = MountAngles.from(up: mount.apply(Vec3.unitZ), fwd: mount.apply(Vec3.unitX));
      expect(m.pitchDeg, closeTo(p, 1e-6));
      expect(m.rollDeg, closeTo(r, 1e-6));
      expect(m.yawDeg, closeTo(y, 1e-6));
    }
  });

  test('fallback calibration has the Android yaw sign and no forward axis', () {
    final c = Calibration.fallback(Vec3.unitZ);
    expect(c.yawSign, -1.0);
    expect(c.hasForwardAxis, isFalse);
    expect(c.mount, isNull);
    expect(c.fromData, isFalse);
  });

  test('calibrator refuses to calibrate on too little data', () {
    final cal = BlackoutCalibrator(windowSeconds: 30, minSeconds: 8, fs: 10);
    for (var i = 0; i < 40; i++) {
      cal.add(CalSample(
        t: i * 0.1,
        lin: Vec3.zero,
        gyro: Vec3.zero,
        ghat: Vec3.unitZ,
        speed: 10,
        courseRad: 0,
        fixT: (i * 0.1).floorToDouble(),
      ));
    }
    expect(cal.compute(), isNull);
    expect(cal.seconds, closeTo(3.9, 1e-9));
  });
}
