import 'package:flutter/material.dart';

import '../engine/navigation_engine.dart';
import '../utils/compass_utils.dart';
import '../utils/theme.dart';
import 'sparkline_chart.dart';

/// Bottom telemetry panel: fused speed and heading, GNSS / mount / uncertainty
/// read-outs, and the aligned IMU traces (forward acceleration, yaw rate).
class TelemetryPanel extends StatelessWidget {
  final NavState? state;

  const TelemetryPanel({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    final s = state;
    final speedKmh = (s?.speedMs ?? 0) * 3.6;
    final heading = s?.headingDeg ?? 0;
    final isDr = s?.isDeadReckoning ?? false;

    final gnssText = s == null || !s.hasFix
        ? '— m'
        : '${s.fixAccuracyM.isFinite ? s.fixAccuracyM.toStringAsFixed(0) : '—'} m · '
            '${s.fixAgeS.isFinite ? s.fixAgeS.toStringAsFixed(0) : '—'} s';
    final gnssIcon = switch (s?.gnssHealth) {
      GnssHealth.good => Icons.gps_fixed,
      GnssHealth.degraded => Icons.gps_not_fixed,
      _ => Icons.gps_off,
    };
    final mountText = s?.mount == null
        ? (s?.alignment == AlignmentState.fullyAligned ? 'aligned' : 'calibrating')
        : s!.mount.toString();
    final unc = s == null || !s.uncertaintyM.isFinite
        ? '—'
        : (s.uncertaintyM >= 100 ? '${(s.uncertaintyM / 1000).toStringAsFixed(1)} km' : '${s.uncertaintyM.toStringAsFixed(0)} m');

    final vib = (s?.vibrationRms ?? 0).clamp(0.0, 3.0) / 3.0;

    return HudTheme.glassWrap(
      backgroundColor: HudTheme.surface.withValues(alpha: 0.85),
      borderRadius: 28,
      padding: const EdgeInsets.fromLTRB(22, 18, 22, 18),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Speed + heading ────────────────────────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                flex: 3,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _label(Icons.speed, isDr ? 'SPEED · INS' : 'SPEED · FUSED'),
                    const SizedBox(height: 2),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.baseline,
                      textBaseline: TextBaseline.alphabetic,
                      children: [
                        TweenAnimationBuilder<double>(
                          tween: Tween(end: speedKmh),
                          duration: const Duration(milliseconds: 180),
                          builder: (context, value, _) => Text(
                            value.toStringAsFixed(0),
                            style: HudTheme.hudMassive(isDr ? HudTheme.drAmber : HudTheme.textPrimary),
                          ),
                        ),
                        const SizedBox(width: 4),
                        Text('km/h', style: HudTheme.hudUnit()),
                      ],
                    ),
                  ],
                ),
              ),
              Expanded(
                flex: 2,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    _label(Icons.explore, 'HEADING'),
                    const SizedBox(height: 2),
                    TweenAnimationBuilder<double>(
                      tween: Tween(end: heading),
                      duration: const Duration(milliseconds: 180),
                      builder: (context, value, _) => Text(
                        formatHeading(value % 360),
                        style: HudTheme.hudMedium(HudTheme.textPrimary).copyWith(fontWeight: FontWeight.w700),
                      ),
                    ),
                    if (s != null && s.headingSigmaDeg < 90)
                      Text('±${s.headingSigmaDeg.toStringAsFixed(0)}°', style: HudTheme.hudUnit()),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          // ── GNSS · mount · uncertainty ─────────────────────────
          Row(
            children: [
              Expanded(child: _stat(gnssIcon, 'GNSS', gnssText)),
              Expanded(child: _stat(Icons.screen_rotation_alt, 'MOUNT', mountText)),
              Expanded(child: _stat(Icons.radio_button_unchecked, '2σ ERROR', unc, alignEnd: true)),
            ],
          ),

          const SizedBox(height: 16),

          // ── Aligned IMU traces ─────────────────────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _label(Icons.trending_flat, 'FWD ACCEL  ±4 m/s²'),
                    const SizedBox(height: 4),
                    LayoutBuilder(
                      builder: (context, c) => SparklineChart(
                        data: s?.fwdAccelHistory ?? const [],
                        lineColor: HudTheme.textPrimary,
                        width: c.maxWidth,
                        height: 40,
                        minValue: -4,
                        maxValue: 4,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _label(Icons.rotate_right, 'YAW RATE  ±45°/s'),
                    const SizedBox(height: 4),
                    LayoutBuilder(
                      builder: (context, c) => SparklineChart(
                        data: s?.yawRateHistory ?? const [],
                        lineColor: HudTheme.textPrimary,
                        width: c.maxWidth,
                        height: 40,
                        minValue: -45,
                        maxValue: 45,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 10),

          // ── Road vibration ─────────────────────────────────────
          Row(
            children: [
              Text('ROAD', style: HudTheme.hudLabel()),
              const SizedBox(width: 10),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(3),
                  child: LinearProgressIndicator(
                    value: vib,
                    minHeight: 4,
                    backgroundColor: Colors.white.withValues(alpha: 0.08),
                    valueColor: AlwaysStoppedAnimation<Color>(
                      vib > 0.66 ? HudTheme.drAmber : HudTheme.textSecondary,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Text(
                s == null ? '' : '${s.vibrationRms.toStringAsFixed(2)} m/s² · ${s.shockCount} shocks',
                style: HudTheme.hudLabel(),
              ),
            ],
          ),
        ],
      ),
    );
  }

  static Widget _label(IconData icon, String text) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: HudTheme.labelMuted),
          const SizedBox(width: 4),
          Text(text, style: HudTheme.hudLabel()),
        ],
      );

  static Widget _stat(IconData icon, String label, String value, {bool alignEnd = false}) => Column(
        crossAxisAlignment: alignEnd ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: [
          _label(icon, label),
          const SizedBox(height: 3),
          Text(
            value,
            style: HudTheme.hudSmall(HudTheme.textPrimary).copyWith(fontWeight: FontWeight.w700),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      );
}
