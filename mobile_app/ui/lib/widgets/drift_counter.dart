import 'package:flutter/material.dart';
import '../utils/theme.dart';

/// Shows estimated accumulated drift when in Dead Reckoning mode.
///
/// Displays: elapsed_seconds × 0.5 meters (placeholder formula).
/// Animated counter with pulsing amber border.
class DriftCounter extends StatefulWidget {
  /// Elapsed seconds since entering Dead Reckoning mode.
  final double elapsedSeconds;

  /// Whether the counter should be visible.
  final bool visible;

  const DriftCounter({
    super.key,
    required this.elapsedSeconds,
    required this.visible,
  });

  @override
  State<DriftCounter> createState() => _DriftCounterState();
}

class _DriftCounterState extends State<DriftCounter>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1600),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedOpacity(
      opacity: widget.visible ? 1.0 : 0.0,
      duration: HudTheme.animNormal,
      curve: HudTheme.animCurve,
      child: AnimatedSlide(
        offset: widget.visible ? Offset.zero : const Offset(0, 0.5),
        duration: HudTheme.animNormal,
        curve: HudTheme.animCurve,
        child: AnimatedBuilder(
          animation: _pulseController,
          builder: (context, _) {
            final pulseVal = _pulseController.value;
            final driftMeters = widget.elapsedSeconds * 0.5;

            return Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              child: HudTheme.glassWrap(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                borderRadius: 32, // Pill shape
                backgroundColor: HudTheme.surface.withValues(alpha: 0.7),
                shadowColor: Colors.black.withValues(alpha: 0.2 + pulseVal * 0.1),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.trending_up,
                      color: HudTheme.drAmber,
                      size: 14,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'EST. DRIFT',
                      style: HudTheme.hudLabel(),
                    ),
                    const SizedBox(width: 8),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.baseline,
                      textBaseline: TextBaseline.alphabetic,
                      children: [
                        TweenAnimationBuilder<double>(
                          tween: Tween(end: driftMeters),
                          duration: const Duration(milliseconds: 300),
                          builder: (context, value, _) {
                            return Text(
                              value.toStringAsFixed(1),
                              style: HudTheme.hudSmall(HudTheme.drAmber).copyWith(
                                fontWeight: FontWeight.w700,
                              ),
                            );
                          },
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
              ),
            );
          },
        ),
      ),
    );
  }
}
