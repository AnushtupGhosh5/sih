import 'package:flutter/material.dart';
import '../utils/theme.dart';

/// Animated banner showing the current navigation mode.
///
/// Transitions smoothly between GNSS+INS FUSION (cyan/green)
/// and DEAD RECKONING (amber/orange) with debounced switching
/// handled by the parent.
class ModeBanner extends StatefulWidget {
  final bool isGnssMode;

  const ModeBanner({super.key, required this.isGnssMode});

  @override
  State<ModeBanner> createState() => _ModeBannerState();
}

class _ModeBannerState extends State<ModeBanner>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    if (!widget.isGnssMode) {
      _pulseController.repeat(reverse: true);
    }
  }

  @override
  void didUpdateWidget(ModeBanner oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isGnssMode) {
      _pulseController.stop();
      _pulseController.value = 0;
    } else {
      if (!_pulseController.isAnimating) {
        _pulseController.repeat(reverse: true);
      }
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isGnss = widget.isGnssMode;

    final bgColor = isGnss
        ? HudTheme.surface.withValues(alpha: 0.7) // Deep dark blur for GNSS
        : HudTheme.drAccent; // Solid orange for DR mode
    final textColor = HudTheme.textPrimary;
    final iconColor = HudTheme.textPrimary; // Keep icon white against the orange/dark background
    final icon = isGnss ? Icons.satellite_alt : Icons.warning_amber_rounded;
    final label = isGnss ? 'GNSS + INS' : 'Dead Reckoning';

    return AnimatedBuilder(
      animation: _pulseController,
      builder: (context, child) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: HudTheme.glassWrap(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            borderRadius: 32, // Pill shape
            backgroundColor: bgColor,
            shadowColor: Colors.black.withValues(alpha: 0.3 + _pulseController.value * 0.1),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                AnimatedSwitcher(
                  duration: HudTheme.animFast,
                  child: Icon(
                    icon,
                    key: ValueKey(icon),
                    color: iconColor,
                    size: 14, // Smaller icon
                  ),
                ),
                const SizedBox(width: 8),
                AnimatedSwitcher(
                  duration: HudTheme.animNormal,
                  transitionBuilder: (child, animation) => FadeTransition(
                    opacity: animation,
                    child: ScaleTransition(
                      scale: Tween<double>(begin: 0.85, end: 1.0).animate(
                        CurvedAnimation(parent: animation, curve: Curves.easeOutBack),
                      ),
                      child: child,
                    ),
                  ),
                  child: Text(
                    label,
                    key: ValueKey(label),
                    style: HudTheme.hudSmall(textColor).copyWith(fontWeight: FontWeight.w600),
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
