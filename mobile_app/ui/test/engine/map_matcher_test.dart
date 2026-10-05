import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/map_matcher.dart';
import 'package:sih_navigation/engine/road_graph.dart';

/// A north-south road with a side road heading east at the midpoint.
///
///   3 (0,1000)
///   |
///   2 (0,500) ---- 4 (300,500)
///   |
///   1 (0,0)
RoadGraph _tee() => RoadGraph.build(
      lat0: 28.6,
      lon0: 77.2,
      nodes: {
        1: (e: 0.0, n: 0.0),
        2: (e: 0.0, n: 500.0),
        3: (e: 0.0, n: 1000.0),
        4: (e: 300.0, n: 500.0),
      },
      edges: [(1, 2), (2, 3), (2, 4)],
    );

void main() {
  test('candidates finds the nearest segments', () {
    final g = _tee();
    final c = g.candidates(5, 250, radius: 40);
    expect(c, isNotEmpty);
    expect(c.first.distance, closeTo(5, 1e-9));
    expect(c.first.alongFromA, closeTo(250, 1e-9));
    expect(g.candidates(200, 200, radius: 40), isEmpty);
  });

  test('anchor picks the edge direction matching the travel heading', () {
    final g = _tee();
    final m = MapAidedReckoner(g);
    expect(m.anchor(2, 100, 0.0), isTrue); // heading north
    expect(m.heading, closeTo(0.0, 1e-9));
    expect(m.position.n, closeTo(100, 1e-9));
    final south = MapAidedReckoner(g);
    expect(south.anchor(2, 100, math.pi), isTrue); // heading south
    expect(south.heading.abs(), closeTo(math.pi, 1e-9));
    expect(MapAidedReckoner(g).anchor(500, 500, 0.0), isFalse); // no road nearby
  });

  test('a gyro turn selects the side road and heading resets to its bearing', () {
    final g = _tee();
    final m = MapAidedReckoner(g);
    m.anchor(0, 100, 0.0);
    const v = 10.0, dt = 0.1;
    for (var t = 0.0; t < 50.0; t += dt) {
      // 90 degree right turn between 30 s and 32 s (before the junction at 40 s)
      final yaw = (t >= 30 && t < 32) ? (math.pi / 2) / 2.0 : 0.0;
      m.step(yaw, v, dt);
    }
    expect(m.junctionsPassed, 1);
    expect(m.position.n, closeTo(500, 1.0));
    expect(m.position.e, closeTo(100, 1.0)); // 10 s past the junction at 10 m/s
    expect(m.heading, closeTo(math.pi / 2, 1e-9)); // reset to the road bearing
    expect(m.distance, closeTo(500, 1e-6));
  });

  test('without a turn the vehicle continues straight through the junction', () {
    final g = _tee();
    final m = MapAidedReckoner(g);
    m.anchor(0, 100, 0.0);
    for (var t = 0.0; t < 50.0; t += 0.1) {
      m.step(0.0, 10.0, 0.1);
    }
    expect(m.position.e, closeTo(0, 1e-6));
    expect(m.position.n, closeTo(600, 1.0));
    expect(m.heading, closeTo(0.0, 1e-9));
  });

  test('builds from Overpass JSON', () {
    final json = {
      'elements': [
        {
          'type': 'way',
          'id': 1,
          'nodes': [10, 11, 12],
          'geometry': [
            {'lat': 28.6000, 'lon': 77.2000},
            {'lat': 28.6010, 'lon': 77.2000},
            {'lat': 28.6020, 'lon': 77.2000},
          ],
        },
        {
          'type': 'way',
          'id': 2,
          'nodes': [11, 13],
          'geometry': [
            {'lat': 28.6010, 'lon': 77.2000},
            {'lat': 28.6010, 'lon': 77.2010},
          ],
        },
        {'type': 'node', 'id': 99, 'lat': 1.0, 'lon': 1.0},
      ],
    };
    final g = RoadGraph.fromOverpassJson(json, 28.6, 77.2);
    expect(g.nodes.length, 4);
    expect(g.edges.length, 3);
    expect(g.nodes[11]!.neighbors.length, 3);
    expect(g.candidates(1, 55, radius: 30), isNotEmpty);
  });
}
