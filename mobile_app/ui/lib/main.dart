import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/permission_screen.dart';
import 'utils/theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // Force dark status bar & navigation bar for immersive HUD feel.
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: HudTheme.background,
    systemNavigationBarIconBrightness: Brightness.light,
  ));

  // Prefer portrait for phone-sized dashboard.
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  runApp(const SihNavigationApp());
}

class SihNavigationApp extends StatelessWidget {
  const SihNavigationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SIH Navigation',
      debugShowCheckedModeBanner: false,
      theme: HudTheme.darkTheme,
      home: const PermissionScreen(),
    );
  }
}
