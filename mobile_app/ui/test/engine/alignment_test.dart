import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/alignment.dart';
import 'package:sih_navigation/engine/vec3.dart';

Vec3 _noise(math.Random r, double s) {
  double g() {
    final u1 = math.max(1e-12, r.nextDouble()), u2 = r.nextDouble();
    return s * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
  }

  return Vec3(g(), g(), g());
}

void main() {
  final mount = Mat3.fromEuler(pitch: 30 * math.pi / 180, roll: 10 * math.pi / 180, yaw: 45 * math.pi / 180);
  final gVeh = const Vec3(0, 0, 9.81);

  test('GNSS-denied path: gravity then PCA recovers an arbitrary mount', () {
    final rng = math.Random(3);
    final al = InVehicleAligner(samplingRate: 10);

    // Stationary for 3 s.
    for (var i = 0; i < 30; i++) {
      al.processFrame(mount.apply(gVeh) + _noise(rng, 0.01), _noise(rng, 0.005), gpsSpeed: 0.0);
    }
    expect(al.state, AlignmentState.zAligned);

    // Straight driving with accelerate/brake cycles (braking sharper).
    var aligned = false;
    for (var i = 0; i < 200 && !aligned; i++) {
      final t = i / 10.0;
      final phase = math.sin(2 * math.pi * t / 8);
      final a = phase > 0 ? 1.5 * phase : 3.0 * phase; // +1.5 accel, -3.0 brake
      final aVeh = Vec3(a, 0, 9.81);
      aligned = al.processFrame(mount.apply(aVeh) + _noise(rng, 0.02), _noise(rng, 0.005));
    }
    expect(aligned, isTrue);
    expect(al.state, AlignmentState.fullyAligned);

    // Forward axis points along the vehicle x-axis as seen by the phone.
    final trueFwd = mount.apply(Vec3.unitX);
    expect(al.xVeh!.dot(trueFwd), greaterThan(0.95));

    // A pothole (pure vertical shock) maps back to the vehicle vertical axis.
    final pothole = mount.apply(const Vec3(0, 0, 15));
    final rec = al.transform(pothole);
    expect((rec - const Vec3(0, 0, 15)).norm, lessThan(0.2));

    // Forward acceleration maps to vehicle x.
    final fwd = al.transform(mount.apply(const Vec3(2.5, 0, 9.81)));
    expect(fwd.x, closeTo(2.5, 0.3));
    expect(fwd.z, closeTo(9.81, 0.3));
  });

  test('GNSS-available path aligns while the vehicle speeds up', () {
    final rng = math.Random(5);
    final al = InVehicleAligner(samplingRate: 10);
    for (var i = 0; i < 30; i++) {
      al.processFrame(mount.apply(gVeh) + _noise(rng, 0.01), _noise(rng, 0.005), gpsSpeed: 0.0);
    }
    var aligned = false;
    for (var i = 0; i < 60 && !aligned; i++) {
      aligned = al.processFrame(mount.apply(const Vec3(1.8, 0, 9.81)) + _noise(rng, 0.02), _noise(rng, 0.005),
          gpsSpeed: 5.0 + 0.18 * i, gpsAccel: 1.8);
    }
    expect(aligned, isTrue);
    expect(al.xVeh!.dot(mount.apply(Vec3.unitX)), greaterThan(0.95));
  });

  test('a violent knock invalidates the alignment', () {
    final rng = math.Random(9);
    final al = InVehicleAligner(samplingRate: 10);
    for (var i = 0; i < 30; i++) {
      al.processFrame(mount.apply(gVeh) + _noise(rng, 0.01), _noise(rng, 0.005), gpsSpeed: 0.0);
    }
    for (var i = 0; i < 60; i++) {
      al.processFrame(mount.apply(const Vec3(1.8, 0, 9.81)) + _noise(rng, 0.02), _noise(rng, 0.005),
          gpsSpeed: 5.0 + 0.18 * i, gpsAccel: 1.8);
    }
    expect(al.isAligned, isTrue);
    for (var i = 0; i < 20; i++) {
      al.processFrame(mount.apply(gVeh), Vec3(i.isEven ? 3.0 : -3.0, 2.0, -1.0));
    }
    expect(al.state, AlignmentState.uncalibrated);
    expect(al.displacementCount, 1);
  });
}
