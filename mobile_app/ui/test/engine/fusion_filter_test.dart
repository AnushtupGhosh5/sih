import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/fusion_filter.dart';

void main() {
  test('prediction follows the kinematic model', () {
    final f = FusionFilter()..init(e: 0, n: 0, heading: math.pi / 2, speed: 10);
    final p0 = f.posSigma;
    for (var i = 0; i < 200; i++) {
      f.predict(omega: 0, aFwd: 0, dt: 0.05);
    }
    expect(f.east, closeTo(100, 1e-6));
    expect(f.north, closeTo(0, 1e-6));
    expect(f.speed, closeTo(10, 1e-9));
    expect(f.posSigma, greaterThan(p0)); // uncertainty grows without updates
  });

  test('learns a gyro bias from GNSS course and position updates', () {
    final f = FusionFilter()..init(e: 0, n: 0, heading: 0, speed: 10);
    const bias = 0.02; // rad/s on the measured yaw rate
    var t = 0.0;
    var nextFix = 1.0;
    while (t < 60) {
      f.predict(omega: bias, aFwd: 0, dt: 0.05);
      t += 0.05;
      if (t >= nextFix - 1e-9) {
        nextFix += 1;
        f.updatePosition(0, 10 * t, 2.0);
        f.updateSpeed(10, 0.3);
        f.updateCourse(0, 0.05);
      }
    }
    expect(f.gyroBias, closeTo(bias, 0.004));
    expect(f.heading.abs(), lessThan(0.03));
    expect(f.east.abs(), lessThan(5));
  });

  test('learns an accelerometer bias from GNSS speed updates', () {
    final f = FusionFilter()..init(e: 0, n: 0, heading: 0, speed: 10);
    const bias = 0.2; // m/s² on the measured forward acceleration
    var t = 0.0;
    var nextFix = 1.0;
    while (t < 60) {
      f.predict(omega: 0, aFwd: bias, dt: 0.05);
      t += 0.05;
      if (t >= nextFix - 1e-9) {
        nextFix += 1;
        f.updateSpeed(10, 0.3);
      }
    }
    expect(f.accelBias, closeTo(bias, 0.05));
    expect(f.speed, closeTo(10, 0.3));
  });

  test('innovation gate rejects a 100 m jump once converged', () {
    final f = FusionFilter()..init(e: 0, n: 0, heading: 0, speed: 0, posSigma: 3);
    for (var i = 0; i < 5; i++) {
      f.predict(omega: 0, aFwd: 0, dt: 0.5);
      f.updatePosition(0, 0, 3);
    }
    expect(f.updatePosition(100, 0, 5), isFalse);
    expect(f.rejectedInRow, 1);
    expect(f.lastInnovationM, closeTo(100, 1));
    expect(f.updatePosition(1, 0, 5), isTrue);
    expect(f.rejectedInRow, 0);
    expect(f.east.abs(), lessThan(2));
  });

  test('zero-velocity update stops the vehicle', () {
    final f = FusionFilter()..init(e: 0, n: 0, heading: 0, speed: 5);
    for (var i = 0; i < 3; i++) {
      f.predict(omega: 0, aFwd: 0, dt: 0.05);
      f.updateZupt();
    }
    expect(f.speed, lessThan(0.1));
  });

  test('soft dead-band', () {
    expect(FusionFilter.softDeadband(0.1, 0.15), 0);
    expect(FusionFilter.softDeadband(0.5, 0.15), closeTo(0.35, 1e-12));
    expect(FusionFilter.softDeadband(-0.5, 0.15), closeTo(-0.35, 1e-12));
  });

  test('without a forward axis the speed is held and GNSS still corrects it', () {
    final f = FusionFilter()..init(e: 0, n: 0, heading: 0, speed: 10);
    for (var i = 0; i < 100; i++) {
      f.predict(omega: 0, aFwd: null, dt: 0.05);
    }
    expect(f.speed, closeTo(10, 1e-9));
    f.updateSpeed(12, 0.3);
    expect(f.speed, greaterThan(11));
  });
}
