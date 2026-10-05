enum GnssHealth { good, degraded, denied }

/// One GNSS fix as delivered by the phone.
class GnssFix {
  const GnssFix({
    required this.t,
    required this.lat,
    required this.lon,
    required this.speedMs,
    required this.courseDeg,
    required this.accuracyM,
  });

  final double t; // seconds, same clock as the IMU samples
  final double lat, lon;
  final double speedMs;
  final double courseDeg; // clockwise from north
  final double accuracyM; // horizontal accuracy (1-sigma-ish) in metres
}

/// Seamless GNSS deficit handler: grades the latest fix as GOOD, DEGRADED or
/// DENIED from its reported accuracy and age (phones do not expose HDOP or
/// C/N0 reliably, so accuracy and staleness stand in for them).
class GnssMonitor {
  GnssMonitor({
    this.goodAccuracyM = 20.0,
    this.degradedAccuracyM = 50.0,
    this.staleSeconds = 3.0,
  });

  final double goodAccuracyM;
  final double degradedAccuracyM;
  final double staleSeconds;

  GnssHealth assess(GnssFix? fix, double now, {bool simulateDenied = false}) {
    if (simulateDenied || fix == null) return GnssHealth.denied;
    if (now - fix.t > staleSeconds) return GnssHealth.denied;
    if (fix.accuracyM <= goodAccuracyM) return GnssHealth.good;
    if (fix.accuracyM <= degradedAccuracyM) return GnssHealth.degraded;
    return GnssHealth.denied;
  }
}
