import 'dart:math' as math;

import 'geo.dart';

class RoadNode {
  RoadNode(this.id, this.e, this.n);
  final int id;
  final double e, n;
  final List<RoadNode> neighbors = <RoadNode>[];
}

class RoadEdge {
  RoadEdge(this.index, this.a, this.b)
      : length = math.sqrt((a.e - b.e) * (a.e - b.e) + (a.n - b.n) * (a.n - b.n));
  final int index;
  final RoadNode a, b;
  final double length;
}

class RoadCandidate {
  const RoadCandidate(this.edge, this.e, this.n, this.distance, this.alongFromA);
  final RoadEdge edge;
  final double e, n; // projection of the query point onto the segment
  final double distance; // metres from the query point
  final double alongFromA; // metres from edge.a to the projection
}

/// Routable road graph in a local ENU frame with a grid index over segments
/// (port of `RoadNetwork` in `src/idr/osm.py`, without shapely/networkx).
class RoadGraph {
  RoadGraph._(this.lat0, this.lon0, this.cellSize);

  final double lat0, lon0;
  final double cellSize;
  final Map<int, RoadNode> nodes = <int, RoadNode>{};
  final List<RoadEdge> edges = <RoadEdge>[];
  final Map<int, List<int>> _grid = <int, List<int>>{};

  static const String drivableRegex =
      'motorway|trunk|primary|secondary|tertiary|unclassified|residential|'
      'living_street|service|road|motorway_link|trunk_link|primary_link|'
      'secondary_link|tertiary_link';

  /// Build from an Overpass `out geom` JSON response.
  factory RoadGraph.fromOverpassJson(Map<String, dynamic> json, double lat0, double lon0,
      {double cellSize = 150.0}) {
    final g = RoadGraph._(lat0, lon0, cellSize);
    final elements = (json['elements'] as List<dynamic>?) ?? const [];
    var synthetic = -1;
    for (final el in elements) {
      if (el is! Map<String, dynamic>) continue;
      if (el['type'] != 'way' || el['geometry'] == null) continue;
      final geom = el['geometry'] as List<dynamic>;
      final rawIds = el['nodes'] as List<dynamic>?;
      final ids = <int>[];
      if (rawIds != null && rawIds.length == geom.length) {
        for (final id in rawIds) {
          ids.add((id as num).toInt());
        }
      } else {
        for (var i = 0; i < geom.length; i++) {
          ids.add(synthetic--);
        }
      }
      for (var i = 0; i < geom.length; i++) {
        final pt = geom[i] as Map<String, dynamic>;
        final id = ids[i];
        if (!g.nodes.containsKey(id)) {
          final p = latLonToEnu((pt['lat'] as num).toDouble(), (pt['lon'] as num).toDouble(), lat0, lon0);
          g.nodes[id] = RoadNode(id, p.e, p.n);
        }
      }
      for (var i = 0; i + 1 < ids.length; i++) {
        g._addEdge(ids[i], ids[i + 1]);
      }
    }
    g._buildIndex();
    return g;
  }

  /// Build directly from ENU coordinates (tests, offline extracts).
  factory RoadGraph.build({
    required double lat0,
    required double lon0,
    required Map<int, ({double e, double n})> nodes,
    required List<(int, int)> edges,
    double cellSize = 150.0,
  }) {
    final g = RoadGraph._(lat0, lon0, cellSize);
    nodes.forEach((id, p) => g.nodes[id] = RoadNode(id, p.e, p.n));
    for (final (a, b) in edges) {
      g._addEdge(a, b);
    }
    g._buildIndex();
    return g;
  }

  void _addEdge(int ia, int ib) {
    final a = nodes[ia], b = nodes[ib];
    if (a == null || b == null || identical(a, b)) return;
    if (a.neighbors.contains(b)) return;
    final e = RoadEdge(edges.length, a, b);
    if (e.length <= 0) return;
    edges.add(e);
    a.neighbors.add(b);
    b.neighbors.add(a);
  }

  int _cellKey(int ix, int iy) => (ix + (1 << 20)) * (1 << 22) + (iy + (1 << 20));

  void _buildIndex() {
    _grid.clear();
    for (final e in edges) {
      final x0 = (math.min(e.a.e, e.b.e) / cellSize).floor();
      final x1 = (math.max(e.a.e, e.b.e) / cellSize).floor();
      final y0 = (math.min(e.a.n, e.b.n) / cellSize).floor();
      final y1 = (math.max(e.a.n, e.b.n) / cellSize).floor();
      for (var ix = x0; ix <= x1; ix++) {
        for (var iy = y0; iy <= y1; iy++) {
          _grid.putIfAbsent(_cellKey(ix, iy), () => <int>[]).add(e.index);
        }
      }
    }
  }

  double bearing(RoadNode from, RoadNode to) => bearingOf(to.e - from.e, to.n - from.n);

  /// Road segments within [radius] metres of (e, n), nearest first.
  List<RoadCandidate> candidates(double e, double n, {double radius = 40.0, int k = 8}) {
    final x0 = ((e - radius) / cellSize).floor(), x1 = ((e + radius) / cellSize).floor();
    final y0 = ((n - radius) / cellSize).floor(), y1 = ((n + radius) / cellSize).floor();
    final seen = <int>{};
    final out = <RoadCandidate>[];
    for (var ix = x0; ix <= x1; ix++) {
      for (var iy = y0; iy <= y1; iy++) {
        final list = _grid[_cellKey(ix, iy)];
        if (list == null) continue;
        for (final idx in list) {
          if (!seen.add(idx)) continue;
          final c = _project(edges[idx], e, n);
          if (c.distance <= radius) out.add(c);
        }
      }
    }
    out.sort((a, b) => a.distance.compareTo(b.distance));
    return out.length > k ? out.sublist(0, k) : out;
  }

  static RoadCandidate _project(RoadEdge edge, double e, double n) {
    final ax = edge.a.e, ay = edge.a.n;
    final dx = edge.b.e - ax, dy = edge.b.n - ay;
    final len2 = dx * dx + dy * dy;
    var t = len2 > 0 ? ((e - ax) * dx + (n - ay) * dy) / len2 : 0.0;
    t = t.clamp(0.0, 1.0);
    final px = ax + t * dx, py = ay + t * dy;
    final d = math.sqrt((e - px) * (e - px) + (n - py) * (n - py));
    return RoadCandidate(edge, px, py, d, t * edge.length);
  }
}
