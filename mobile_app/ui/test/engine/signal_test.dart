import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/geo.dart';
import 'package:sih_navigation/engine/signal.dart';

void main() {
  test('filtfilt keeps a constant and attenuates a fast sine', () {
    final c = butterLowpassFiltfilt(List.filled(200, 3.0), 0.5, 10);
    expect(c.every((v) => (v - 3.0).abs() < 1e-9), isTrue);

    final fast = List.generate(400, (i) => math.sin(2 * math.pi * 4.0 * i / 10));
    final f = butterLowpassFiltfilt(fast, 0.5, 10);
    final amp = f.sublist(50, 350).map((v) => v.abs()).reduce(math.max);
    expect(amp, lessThan(0.05));
  });

  test('unwrap removes 2pi jumps', () {
    final a = [3.0, 3.1, -3.1, -3.0];
    final u = unwrap(a);
    expect(u[2], closeTo(3.1 + (2 * math.pi - 6.2), 1e-9));
    expect(u[3] - u[2], closeTo(0.1, 1e-9));
  });

  test('interpolateHeld turns a staircase into a ramp', () {
    final held = [0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 8.0];
    final r = interpolateHeld(held);
    expect(r[2], closeTo(2.0, 1e-9));
    expect(r[6], closeTo(6.0, 1e-9));
    expect(r[8], closeTo(8.0, 1e-9));
  });

  test('pearson and circular statistics', () {
    final a = List.generate(50, (i) => i.toDouble());
    expect(pearson(a, a), closeTo(1.0, 1e-9));
    expect(pearson(a, a.map((v) => -v).toList()), closeTo(-1.0, 1e-9));
    expect(pearson(a, List.filled(50, 1.0)), 0);
    expect(circularMean([3.1, -3.1]), closeTo(math.pi, 0.01));
    expect(circularR([0.5, 0.5, 0.5]), closeTo(1.0, 1e-9));
  });

  test('geo round trip and bearings', () {
    final p = latLonToEnu(28.6149, 77.2100, 28.6139, 77.2090);
    final back = enuToLatLon(p.e, p.n, 28.6139, 77.2090);
    expect(back.lat, closeTo(28.6149, 1e-9));
    expect(back.lon, closeTo(77.2100, 1e-9));
    expect(bearingOf(1, 0), closeTo(math.pi / 2, 1e-9)); // east
    expect(bearingOf(0, -1).abs(), closeTo(math.pi, 1e-9)); // south
    expect(wrapDeg(-10), closeTo(350, 1e-9));
  });
}
