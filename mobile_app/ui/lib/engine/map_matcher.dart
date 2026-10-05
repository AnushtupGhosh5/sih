import 'dart:math' as math;

import 'geo.dart';
import 'road_graph.dart';

/// Route-constrained (map-aided) dead reckoning (port of `map_aided_dr()` in
/// `src/idr/map_matching.py`).
///
/// Odometry distance advances the vehicle along the road graph; the gyro only
/// supplies the short-term relative heading used to pick the branch at each
/// junction. Heading is reset to the road bearing on every edge, so inertial
/// heading drift never accumulates.
class MapAidedReckoner {
  MapAidedReckoner(this.graph);

  final RoadGraph graph;

  RoadNode? _from, _to;
  double _along = 0, _segLen = 0;

  /// Current heading estimate (radians clockwise from north).
  double heading = 0;
  double distance = 0;
  bool anchored = false;
  int junctionsPassed = 0;

  /// Snap onto the road network at the blackout start. Returns false when no
  /// road is within [radius] metres.
  bool anchor(double e0, double n0, double headingRad, {double radius = 60.0}) {
    final cands = graph.candidates(e0, n0, radius: radius, k: 8);
    if (cands.isEmpty) return false;
    double? bestScore;
    RoadNode? bf, bt;
    for (final c in cands) {
      for (final (frm, to) in [(c.edge.a, c.edge.b), (c.edge.b, c.edge.a)]) {
        final br = graph.bearing(frm, to);
        final score = angDiff(br, headingRad) + 0.02 * c.distance;
        if (bestScore == null || score < bestScore) {
          bestScore = score;
          bf = frm;
          bt = to;
        }
      }
    }
    _from = bf;
    _to = bt;
    final f = _from!, t = _to!;
    final dx = t.e - f.e, dy = t.n - f.n;
    _segLen = math.sqrt(dx * dx + dy * dy);
    final ux = _segLen > 1e-9 ? dx / _segLen : 0.0;
    final uy = _segLen > 1e-9 ? dy / _segLen : 0.0;
    _along = ((e0 - f.e) * ux + (n0 - f.n) * uy).clamp(0.0, _segLen);
    heading = graph.bearing(f, t);
    anchored = true;
    distance = 0;
    junctionsPassed = 0;
    return true;
  }

  /// Advance by one IMU step. [yawRate] is the calibrated heading rate
  /// (rad/s, clockwise positive), [v] the speed (m/s).
  void step(double yawRate, double v, double dt) {
    if (!anchored) return;
    heading += yawRate * dt;
    var remaining = math.max(0.0, v) * dt;
    var guard = 0;
    while (remaining > 1e-6 && guard < 50) {
      guard++;
      if (_along + remaining <= _segLen) {
        _along += remaining;
        remaining = 0;
      } else {
        remaining -= _segLen - _along;
        final to = _to!;
        var nbrs = to.neighbors.where((n) => !identical(n, _from)).toList();
        if (nbrs.isEmpty) nbrs = to.neighbors.toList(); // dead end: allow U-turn
        if (nbrs.isEmpty) {
          _along = _segLen;
          remaining = 0;
          break;
        }
        RoadNode? best;
        double? bestD;
        for (final nb in nbrs) {
          final d = angDiff(graph.bearing(to, nb), heading);
          if (bestD == null || d < bestD) {
            bestD = d;
            best = nb;
          }
        }
        _from = to;
        _to = best;
        final f = _from!, t = _to!;
        _segLen = math.sqrt((t.e - f.e) * (t.e - f.e) + (t.n - f.n) * (t.n - f.n));
        _along = 0;
        heading = graph.bearing(f, t); // RESET heading to the road bearing
        junctionsPassed++;
      }
    }
    distance += math.max(0.0, v) * dt;
  }

  ({double e, double n}) get position {
    final f = _from!, t = _to!;
    final r = _segLen > 1e-9 ? _along / _segLen : 0.0;
    return (e: f.e + (t.e - f.e) * r, n: f.n + (t.n - f.n) * r);
  }

  ({double lat, double lon}) get latLon {
    final p = position;
    return enuToLatLon(p.e, p.n, graph.lat0, graph.lon0);
  }
}
