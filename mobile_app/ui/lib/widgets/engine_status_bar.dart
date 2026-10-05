import 'package:flutter/material.dart';

import '../engine/navigation_engine.dart';
import '../utils/theme.dart';

/// Compact strip above the telemetry panel: engine health on the left, the
/// diagnostics and "simulate GNSS loss" buttons on the right.
class EngineStatusBar extends StatelessWidget {
  const EngineStatusBar({
    super.key,
    required this.state,
    required this.simulate,
    required this.logging,
    required this.onToggleSimulate,
    required this.onOpenDiagnostics,
    this.mapLoading = false,
  });

  final NavState? state;
  final bool simulate;
  final bool logging;
  final bool mapLoading;
  final VoidCallback onToggleSimulate;
  final VoidCallback onOpenDiagnostics;

  @override
  Widget build(BuildContext context) {
    final s = state;
    final parts = <String>[];
    if (s == null) {
      parts.add('WAITING FOR SENSORS');
    } else {
      final cal = s.calibration;
      final calibrated = s.mount != null;
      parts.add(calibrated ? 'ALIGNED' : (s.alignment == AlignmentState.fullyAligned ? 'ALIGNED' : 'ALIGNING'));
      parts.add(cal != null && cal.fromData ? 'CAL ${cal.forwardCorr.clamp(0, 1).toStringAsFixed(2)}' : 'CAL —');
      if (s.roadSegments > 0) {
        parts.add('MAP ${(s.roadSegments / 1000).toStringAsFixed(1)}k');
      } else {
        parts.add(mapLoading ? 'MAP …' : 'MAP —');
      }
      if (s.isDeadReckoning) parts.add(s.mapAided ? 'ON ROAD' : 'FREE INS');
      if (logging) parts.add('REC');
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
                      color: recovery.driftPct < 10 ? HudTheme.gnssDot : HudTheme.drAccent,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(width: 8),
        _roundButton(Icons.tune, false, onOpenDiagnostics),
        const SizedBox(width: 8),
        _roundButton(simulate ? Icons.gps_fixed : Icons.gps_off, simulate, onToggleSimulate),
      ],
    );
  }

  static Widget _roundButton(IconData icon, bool active, VoidCallback onTap) => GestureDetector(
        onTap: onTap,
        child: HudTheme.glassWrap(
          padding: const EdgeInsets.all(12),
          borderRadius: 20,
          backgroundColor: active ? HudTheme.drAccent : HudTheme.surface.withValues(alpha: 0.7),
          child: Icon(icon, size: 18, color: HudTheme.textPrimary),
        ),
      );
}
