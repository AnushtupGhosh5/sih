import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import '../utils/theme.dart';
import 'dashboard_screen.dart';

/// Clean, dark-themed permission request screen shown on first launch.
class PermissionScreen extends StatefulWidget {
  const PermissionScreen({super.key});

  @override
  State<PermissionScreen> createState() => _PermissionScreenState();
}

class _PermissionScreenState extends State<PermissionScreen>
    with SingleTickerProviderStateMixin {
  bool _requesting = false;
  String _statusMessage = '';
  late AnimationController _glowController;

  @override
  void initState() {
    super.initState();
    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);

    // Check if already granted.
    _checkAndNavigate();
  }

  @override
  void dispose() {
    _glowController.dispose();
    super.dispose();
  }

  Future<void> _checkAndNavigate() async {
    final locationStatus = await Permission.locationWhenInUse.status;
    final sensorsStatus = await Permission.sensors.status;

    if (locationStatus.isGranted && sensorsStatus.isGranted) {
      if (!mounted) return;
      _navigateToDashboard();
    }
  }

  Future<void> _requestPermissions() async {
    setState(() {
      _requesting = true;
      _statusMessage = 'Requesting location access...';
    });

    // Request location.
    var locationStatus = await Permission.locationWhenInUse.request();

    if (locationStatus.isPermanentlyDenied) {
      setState(() {
        _statusMessage = 'Location permanently denied. Opening settings...';
      });
      await openAppSettings();
      setState(() => _requesting = false);
      return;
    }

    if (!locationStatus.isGranted) {
      setState(() {
        _statusMessage = 'Location permission denied.';
        _requesting = false;
      });
      return;
    }

    setState(() {
      _statusMessage = 'Requesting sensor access...';
    });

    // Request sensors.
    var sensorsStatus = await Permission.sensors.request();

    if (sensorsStatus.isPermanentlyDenied) {
      setState(() {
        _statusMessage = 'Sensors permanently denied. Opening settings...';
      });
      await openAppSettings();
      setState(() => _requesting = false);
      return;
    }

    if (!sensorsStatus.isGranted) {
      // Some devices don't require explicit sensor permission.
      // Proceed anyway — sensors_plus will just return zero values.
    }

    setState(() {
      _statusMessage = 'All permissions granted!';
    });

    await Future.delayed(const Duration(milliseconds: 500));
    if (!mounted) return;
    _navigateToDashboard();
  }

  void _navigateToDashboard() {
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) =>
            const DashboardScreen(),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(opacity: animation, child: child);
        },
        transitionDuration: const Duration(milliseconds: 600),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: HudTheme.background,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(40, 60, 40, 40),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Smaller, left-aligned elegant icon.
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: HudTheme.surface,
                ),
                child: Icon(
                  Icons.navigation_rounded,
                  size: 40,
                  color: HudTheme.textPrimary,
                ),
              ),

              const SizedBox(height: 48),

              Text(
                'SIH NAVIGATION',
                style: HudTheme.hudMedium(HudTheme.textPrimary).copyWith(
                  letterSpacing: 4,
                  fontSize: 32,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Navigation System',
                style: HudTheme.hudLabel().copyWith(
                  color: HudTheme.textSecondary,
                  letterSpacing: 2,
                ),
              ),

              const SizedBox(height: 48),

              // Explanation text, left-aligned.
              Text(
                'This app requires access to your location and motion sensors for real-time navigation and dead reckoning.',
                textAlign: TextAlign.left,
                style: HudTheme.hudSmall(HudTheme.textSecondary).copyWith(
                  height: 1.6,
                ),
              ),

              const Spacer(),

              // Asymmetric bottom button.
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  SizedBox(
                    width: 220,
                    height: 52,
                    child: ElevatedButton(
                      onPressed: _requesting ? null : _requestPermissions,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: HudTheme.textPrimary,
                        foregroundColor: HudTheme.background,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(32),
                        ),
                        elevation: 0,
                      ),
                      child: _requesting
                          ? SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: HudTheme.background,
                              ),
                            )
                          : Text(
                              'Grant Access',
                              style: HudTheme.bannerText().copyWith(
                                color: HudTheme.background,
                              ),
                            ),
                    ),
                  ),
                ],
              ),

              if (_statusMessage.isNotEmpty) ...[
                const SizedBox(height: 24),
                AnimatedOpacity(
                  opacity: _statusMessage.isNotEmpty ? 1.0 : 0.0,
                  duration: HudTheme.animFast,
                  child: Align(
                    alignment: Alignment.centerRight,
                    child: Text(
                      _statusMessage,
                      textAlign: TextAlign.right,
                      style: HudTheme.hudSmall(HudTheme.textSecondary),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
