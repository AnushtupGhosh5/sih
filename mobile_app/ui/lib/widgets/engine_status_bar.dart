import 'package:flutter/material.dart';

import '../engine/navigation_engine.dart';
import '../utils/theme.dart';

/// Compact strip above the telemetry panel: engine health on the left,
/// the "simulate tunnel" demo switch on the right.
class EngineStatusBar extends StatelessWidget {
  const EngineStatusBar({
    super.key,
    required this.state,
    required this.simulate,
    required this.onToggleSimulate,
    this.mapLoading = false,
  });

  final NavState? state;
  final bool simulate;
  final bool mapLoading;
  final VoidCallback onToggleSimulate;

  String _alignLabel(AlignmentState s) => switch (s) {
        AlignmentState.fullyAligned => 'ALIGNED',
        AlignmentState.zAligned => 'ALIGN Z',
        AlignmentState.uncalibrated => 'ALIGNING',
      };

  @override
  Widget build(BuildContext context) {
    final s = state;
    final parts = <String>[];
    if (s == null) {
      parts.add('WAITING FOR SENSORS');
    } else {
      parts.add(_alignLabel(s.alignment));
      final cal = s.calibration;
      if (cal != null && cal.fromData) {
        parts.add('CAL ${cal.forwardCorr.clamp(0, 1).toStringAsFixed(2)}');
      } else {
        parts.add('CAL —');
      }
      if (s.roadSegments > 0) {
        parts.add('MAP ${(s.roadSegments / 1000).toStringAsFixed(1)}k');
      } else if (mapLoading) {
        parts.add('MAP …');
      } else {
        parts.add('MAP —');
      }
      if (s.isDeadReckoning) parts.add(s.mapAided ? 'ON ROAD' : 'FREE INS');
    }
    final recovery = s?.lastRecovery;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: HudTheme.glassWrap(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            borderRadius: 20,
            backgroundColor: HudTheme.surface.withValues(alpha: 0.7),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  parts.join('  ·  '),
                  style: HudTheme.hudLabel().copyWith(color: HudTheme.textPrimary),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                if (recovery != null) ...[
                  const SizedBox(height: 4),
                  Text(
                    'LAST DR ${recovery.distanceM.toStringAsFixed(0)} m · '
                    '${recovery.errorM.toStringAsFixed(1)} m OFF '
                    '(${recovery.driftPct.toStringAsFixed(1)} %)'
                    '${recovery.mapAided ? ' · MAP' : ''}',
                    style: HudTheme.hudLabel().copyWith(
                      color: recovery.driftPct < 10 ? HudTheme.textPrimary : HudTheme.drAccent,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(width: 10),
        GestureDetector(
          onTap: onToggleSimulate,
          child: HudTheme.glassWrap(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            borderRadius: 20,
            backgroundColor: simulate ? HudTheme.drAccent : HudTheme.surface.withValues(alpha: 0.7),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(simulate ? Icons.gps_fixed : Icons.gps_off, size: 14, color: HudTheme.textPrimary),
                const SizedBox(width: 6),
                Text(
                  simulate ? 'END TUNNEL' : 'SIMULATE TUNNEL',
                  style: HudTheme.hudLabel().copyWith(color: HudTheme.textPrimary),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
