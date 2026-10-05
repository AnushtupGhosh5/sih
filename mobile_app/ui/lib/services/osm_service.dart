import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart' show compute;
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../engine/geo.dart';
import '../engine/road_graph.dart';

class _Bbox {
  const _Bbox(this.south, this.west, this.north, this.east);
  final double south, west, north, east;

  static _Bbox around(double lat, double lon, double radiusM) {
    final dlat = radToDeg(radiusM / earthRadiusM);
    final dlon = dlat / math.max(math.cos(degToRad(lat)), 1e-6);
    return _Bbox(lat - dlat, lon - dlon, lat + dlat, lon + dlon);
  }

  bool containsWithMargin(double lat, double lon, double marginM) {
    final dlat = radToDeg(marginM / earthRadiusM);
    final dlon = dlat / math.max(math.cos(degToRad(lat)), 1e-6);
    return lat - dlat >= south && lat + dlat <= north && lon - dlon >= west && lon + dlon <= east;
  }
}

/// Parses an Overpass response and builds the graph off the UI isolate.
RoadGraph _buildGraph((String body, double lat, double lon) args) {
  final json = jsonDecode(args.$1) as Map<String, dynamic>;
  return RoadGraph.fromOverpassJson(json, args.$2, args.$3);
}

/// Fetches the drivable OpenStreetMap road network around the vehicle from
/// Overpass while GNSS is available, caches it on disk, and builds the
/// [RoadGraph] used for map-aided dead reckoning.
class OsmService {
  OsmService({
    this.radiusM = 2000,
    this.refetchMarginM = 600,
    this.minRefetchSeconds = 60,
    this.endpoint = 'https://overpass-api.de/api/interpreter',
  });

  final double radiusM;
  final double refetchMarginM;
  final int minRefetchSeconds;
  final String endpoint;

  RoadGraph? graph;
  String? lastError;
  _Bbox? _bbox;
  DateTime? _lastAttempt;
  bool _inFlight = false;

  bool get busy => _inFlight;

  /// Make sure [graph] covers (lat, lon). Returns the current graph (possibly
  /// unchanged) and never throws.
  Future<RoadGraph?> ensureCoverage(double lat, double lon) async {
    final b = _bbox;
    if (b != null && graph != null && b.containsWithMargin(lat, lon, refetchMarginM)) return graph;
    if (_inFlight) return graph;
    final now = DateTime.now();
    final last = _lastAttempt;
    if (last != null && now.difference(last).inSeconds < minRefetchSeconds) return graph;
    _lastAttempt = now;
    _inFlight = true;
    try {
      final bbox = _Bbox.around(lat, lon, radiusM);
      final body = await _load(bbox);
      if (body != null) {
        final g = await compute(_buildGraph, (body, lat, lon));
        if (g.edges.isEmpty) {
          // Overpass answered (often a timeout "remark") without roads: do
          // not treat this box as covered, so the next attempt retries.
          lastError = 'no roads in response';
        } else {
          graph = g;
          _bbox = bbox;
          lastError = null;
        }
      }
    } catch (e) {
      lastError = e.toString();
    } finally {
      _inFlight = false;
    }
    return graph;
  }

  Future<String?> _load(_Bbox b) async {
    final key = '${(b.south * 1000).round()}_${(b.west * 1000).round()}_'
        '${(b.north * 1000).round()}_${(b.east * 1000).round()}';
    File? cache;
    try {
      final dir = await getApplicationDocumentsDirectory();
      cache = File('${dir.path}/osm_cache/$key.json');
      if (await cache.exists() && await cache.length() > 100) {
        return await cache.readAsString();
      }
    } catch (_) {
      cache = null;
    }

    final query = '[out:json][timeout:60];'
        'way(${b.south},${b.west},${b.north},${b.east})'
        '["highway"~"${RoadGraph.drivableRegex}"];out geom;';
    final resp = await http
        .post(Uri.parse(endpoint), body: {'data': query}, headers: {'User-Agent': 'sih-idr-app'})
        .timeout(const Duration(seconds: 90));
    if (resp.statusCode != 200) {
      throw HttpException('Overpass HTTP ${resp.statusCode}');
    }
    final body = resp.body;
    if (cache != null && body.contains('"elements"')) {
      try {
        await cache.parent.create(recursive: true);
        await cache.writeAsString(body);
      } catch (_) {}
    }
    return body;
  }
}
