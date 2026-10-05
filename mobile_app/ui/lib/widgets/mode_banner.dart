import 'package:flutter/material.dart';

import '../engine/navigation_engine.dart';
import '../utils/theme.dart';

/// Animated banner showing the current navigation mode:
/// GNSS + INS (quiet), GNSS degraded (amber outline), Dead reckoning (solid
/// orange, pulsing).
class ModeBanner extends StatefulWidget {
  final NavMode mode;
  final bool acquiring;

  const ModeBanner({super.key, required this.mode, this.acquiring = false});

  @override
  State<ModeBanner> createState() => _ModeBannerState();
}

class _ModeBannerState extends State<ModeBanner> with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200));
    _sync();
  }

  void _sync() {
    if (widget.mode == NavMode.deadReckoning) {
      if (!_pulseController.isAnimating) _pulseController.repeat(reverse: true);
    } else {
      _pulseController.stop();
      _pulseController.value = 0;
    }
  }

  @override
  void didUpdateWidget(ModeBanner oldWidget) {
    super.didUpdateWidget(oldWidget);
    _sync();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final (Color bg, IconData icon, String label) = switch (widget.mode) {
      NavMode.deadReckoning => (HudTheme.drAccent, Icons.warning_amber_rounded, 'DEAD RECKONING'),
      NavMode.degraded => (
          HudTheme.surface.withValues(alpha: 0.75),
          Icons.gps_not_fixed,
          widget.acquiring ? 'ACQUIRING GNSS' : 'GNSS DEGRADED · INS'
        ),
      NavMode.gnssIns => (HudTheme.surface.withValues(alpha: 0.7), Icons.satellite_alt, 'GNSS + INS'),
    };
    final dot = switch (widget.mode) {
      NavMode.deadReckoning => HudTheme.textPrimary,
      NavMode.degraded => HudTheme.drAmber,
      NavMode.gnssIns => HudTheme.gnssDot,
    };

    return AnimatedBuilder(
      animation: _pulseController,
      builder: (context, child) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: HudTheme.glassWrap(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            borderRadius: 32,
            backgroundColor: bg,
            shadowColor: Colors.black.withValues(alpha: 0.3 + _pulseController.value * 0.1),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(shape: BoxShape.circle, color: dot),
                ),
                const SizedBox(width: 8),
                Icon(icon, color: HudTheme.textPrimary, size: 14),
                const SizedBox(width: 8),
                AnimatedSwitcher(
                  duration: HudTheme.animNormal,
                  child: Text(
                    label,
                    key: ValueKey(label),
                    style: HudTheme.hudSmall(HudTheme.textPrimary).copyWith(fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
