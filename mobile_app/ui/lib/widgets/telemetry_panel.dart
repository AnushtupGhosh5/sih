import 'package:flutter/material.dart';
import '../utils/compass_utils.dart';
import '../utils/theme.dart';
import 'sparkline_chart.dart';

/// Bottom telemetry panel showing speed, heading, accuracy, and sensor sparklines.
class TelemetryPanel extends StatelessWidget {
  final double speedKmh;
  final double heading;
  final double gpsAccuracy;
  final bool isGnssMode;
  final List<double> accelHistory;
  final List<double> gyroHistory;

  const TelemetryPanel({
    super.key,
    required this.speedKmh,
    required this.heading,
    required this.gpsAccuracy,
    required this.isGnssMode,
    required this.accelHistory,
    required this.gyroHistory,
  });

  @override
  Widget build(BuildContext context) {
    final accent = HudTheme.textPrimary;

    return HudTheme.glassWrap(
      backgroundColor: HudTheme.surface.withValues(alpha: 0.85),
      borderRadius: 32,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Top row: Speed + Heading ──────────────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              // Speed — large prominent number.
              Expanded(
                flex: 3,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.speed, size: 14, color: HudTheme.labelMuted),
                        const SizedBox(width: 4),
                        Text('SPEED', style: HudTheme.hudLabel()),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.baseline,
                      textBaseline: TextBaseline.alphabetic,
                      children: [
                        TweenAnimationBuilder<double>(
                          tween: Tween(end: speedKmh),
                          duration: HudTheme.animNormal,
                          curve: HudTheme.animCurve,
                          builder: (context, value, _) {
                            return Text(
                              value.toStringAsFixed(1),
                              style: HudTheme.hudMassive(accent),
                            );
                          },
                        ),
                        const SizedBox(width: 4),
                        Text(
                          'km/h',
                          style: HudTheme.hudUnit(),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              // Heading.
              Expanded(
                flex: 2,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.explore, size: 14, color: HudTheme.labelMuted),
                        const SizedBox(width: 4),
                        Text('HEADING', style: HudTheme.hudLabel()),
                      ],
                    ),
                    const SizedBox(height: 2),
                    TweenAnimationBuilder<double>(
                      tween: Tween(end: heading),
                      duration: HudTheme.animNormal,
                      curve: HudTheme.animCurve,
                      builder: (context, value, _) {
                        return Text(
                          formatHeading(value % 360),
                          style: HudTheme.hudMedium(HudTheme.textPrimary).copyWith(fontWeight: FontWeight.w700),
                        );
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 32),

          // ── Bottom row: Accuracy + Sparklines ────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // GPS Accuracy.
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.satellite_alt, size: 14, color: HudTheme.labelMuted),
                      const SizedBox(width: 4),
                      Text('GPS ACCURACY', style: HudTheme.hudLabel()),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Icon(
                        gpsAccuracy < 20
                            ? Icons.gps_fixed
                            : Icons.gps_not_fixed,
                        color: HudTheme.textPrimary,
                        size: 16,
                      ),
                      const SizedBox(width: 6),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.baseline,
                        textBaseline: TextBaseline.alphabetic,
                        children: [
                          Text(
                            gpsAccuracy.toStringAsFixed(1),
                            style: HudTheme.hudSmall(HudTheme.textPrimary).copyWith(fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(width: 4),
                          Text(
                            'm',
                            style: HudTheme.hudUnit(),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),

              const Spacer(), // Pushes sensors to the right

              // Accelerometer sparkline.
              Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.vibration, size: 14, color: HudTheme.labelMuted),
                      const SizedBox(width: 4),
                      Text('ACCEL', style: HudTheme.hudLabel()),
                    ],
                  ),
                  const SizedBox(height: 4),
                  SparklineChart(
                    data: accelHistory,
                    lineColor: HudTheme.textPrimary,
                    width: 80, // Reduced from 100 to prevent overflow
                    height: 40,
                  ),
                ],
              ),

              const SizedBox(width: 12), // Reduced spacing

              // Gyroscope sparkline.
              Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.screen_rotation, size: 14, color: HudTheme.labelMuted),
                      const SizedBox(width: 4),
                      Text('GYRO', style: HudTheme.hudLabel()),
                    ],
                  ),
                  const SizedBox(height: 4),
                  SparklineChart(
                    data: gyroHistory,
                    lineColor: HudTheme.textPrimary,
                    width: 80, // Reduced from 100 to prevent overflow
                    height: 40,
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }
}
