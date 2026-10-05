import 'dart:math' as math;

const double earthRadiusM = 6371000.0;

double degToRad(double d) => d * math.pi / 180.0;
double radToDeg(double r) => r * 180.0 / math.pi;

/// Normalise an angle in radians to (-pi, pi].
double wrapPi(double a) => math.atan2(math.sin(a), math.cos(a));

/// Normalise a heading in degrees to [0, 360).
double wrapDeg(double d) {
  var r = d % 360.0;
  if (r < 0) r += 360.0;
  return r;
}

/// Local east/north offset in metres of (lat, lon) from (lat0, lon0).
/// Equirectangular approximation, identical to the Python pipeline.
({double e, double n}) latLonToEnu(double lat, double lon, double lat0, double lon0) {
  final lat0r = degToRad(lat0);
  final e = degToRad(lon - lon0) * earthRadiusM * math.cos(lat0r);
  final n = degToRad(lat - lat0) * earthRadiusM;
  return (e: e, n: n);
}

({double lat, double lon}) enuToLatLon(double e, double n, double lat0, double lon0) {
  final lat0r = degToRad(lat0);
  final lat = lat0 + radToDeg(n / earthRadiusM);
  final lon = lon0 + radToDeg(e / (earthRadiusM * math.cos(lat0r)));
  return (lat: lat, lon: lon);
}

/// Planar distance in metres between two lat/lon points (short baselines).
double distanceM(double lat1, double lon1, double lat2, double lon2) {
  final p = latLonToEnu(lat2, lon2, lat1, lon1);
  return math.sqrt(p.e * p.e + p.n * p.n);
}

/// Compass bearing (radians, clockwise from north) of the vector (de, dn).
double bearingOf(double de, double dn) => math.atan2(de, dn);

/// Absolute angular difference in radians, in [0, pi].
double angDiff(double a, double b) => wrapPi(a - b).abs();
