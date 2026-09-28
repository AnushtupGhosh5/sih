/// Converts a heading in degrees [0, 360) to a compass direction string.
String headingToCompass(double degrees) {
  const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  final index = ((degrees % 360) / 45).round() % 8;
  return directions[index];
}

/// Formats heading as "247° WSW" style string.
String formatHeading(double degrees) {
  return '${degrees.toStringAsFixed(0)}° ${headingToCompass(degrees)}';
}
