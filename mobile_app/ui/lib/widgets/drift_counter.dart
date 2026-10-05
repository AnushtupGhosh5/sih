import 'package:flutter/material.dart';
import '../utils/theme.dart';

/// Blackout readout shown while dead reckoning: distance travelled on the
/// inertial solution, time without GNSS, and whether the road graph is
/// constraining the path.
class DriftCounter extends StatelessWidget {
  const DriftCounter({
    super.key,
    required this.visible,
    required this.distanceM,
    required this.elapsedSeconds,
    required this.mapAided,
  });

  final bool visible;
  final double distanceM;
  final double elapsedSeconds;
  final bool mapAided;

  @override
  Widget build(BuildContext context) {
    final valueStyle = HudTheme.hudSmall(HudTheme.drAmber).copyWith(fontWeight: FontWeight.w700);

    return AnimatedOpacity(
      opacity: visible ? 1.0 : 0.0,
      duration: HudTheme.animNormal,
      curve: HudTheme.animCurve,
      child: AnimatedSlide(
        offset: visible ? Offset.zero : const Offset(0, 0.5),
        duration: HudTheme.animNormal,
        curve: HudTheme.animCurve,
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 16),
          child: HudTheme.glassWrap(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            borderRadius: 32,
            backgroundColor: HudTheme.surface.withValues(alpha: 0.7),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.straighten, color: HudTheme.drAmber, size: 14),
                const SizedBox(width: 8),
                Text('DR', style: HudTheme.hudLabel()),
                const SizedBox(width: 6),
                Text(distanceM.toStringAsFixed(0), style: valueStyle),
                const SizedBox(width: 3),
                Text('m', style: HudTheme.hudUnit()),
                const SizedBox(width: 10),
                Text(elapsedSeconds.toStringAsFixed(1), style: valueStyle),
                const SizedBox(width: 3),
                Text('s', style: HudTheme.hudUnit()),
                const SizedBox(width: 10),
                Text(mapAided ? 'ON ROAD' : 'FREE INS', style: HudTheme.hudLabel()),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
