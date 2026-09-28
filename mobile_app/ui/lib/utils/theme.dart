import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Central design system for the HUD dashboard.
class HudTheme {
  HudTheme._();

  // ── Core palette ──────────────────────────────────────────────────────
  static const Color background = Color(0xFF000000); // Pure black
  static const Color surface = Color(0xFF000000); // Solid pure black panels
  static const Color surfaceGlass = Color(0x66000000);
  static const Color borderSubtle = Color(0x11FFFFFF);

  // GNSS mode — Minimalist white/grey
  static const Color gnssAccent = Color(0xFFFFFFFF);
  static const Color gnssGlow = Color(0x33FFFFFF);
  static const Color gnssGreen = Color(0xFFFFFFFF); // No longer green, pure white
  static const Color mapMarker = Color(0xFF4285F4); // Google Maps blue

  // Dead Reckoning mode — Specific Orange Accent
  static const Color drAccent = Color(0xFFFF8A3D);
  static const Color drGlow = Color(0x33FF8A3D);
  static const Color drAmber = Color(0xFFFF8A3D);

  // General
  static const Color textPrimary = Color(0xFFFFFFFF); // Pure white
  static const Color textSecondary = Color(0x8CFFFFFF); // ~55% opacity white
  static const Color labelMuted = Color(0x8CFFFFFF); // ~55% opacity white
  static const Color danger = Color(0xFFFF1744);

  // ── Animation constants ───────────────────────────────────────────────
  static const Duration animFast = Duration(milliseconds: 200);
  static const Duration animNormal = Duration(milliseconds: 400);
  static const Duration animSlow = Duration(milliseconds: 800);
  static const Curve animCurve = Curves.easeInOutCubic;

  // ── GPS threshold ─────────────────────────────────────────────────────
  static const double gpsAccuracyThreshold = 20.0; // meters
  static const Duration modeDebounceDuration = Duration(milliseconds: 1500);

  // ── Typography (Geometric / Clean) ────────────────────────────────────
  static TextStyle hudMassive(Color color) => GoogleFonts.inter(
        fontSize: 72,
        fontWeight: FontWeight.w700,
        color: color,
        height: 1.0,
        letterSpacing: -2,
      );

  static TextStyle hudLarge(Color color) => GoogleFonts.inter(
        fontSize: 32,
        fontWeight: FontWeight.w700,
        color: color,
        letterSpacing: -0.5,
      );

  static TextStyle hudMedium(Color color) => GoogleFonts.inter(
        fontSize: 22,
        fontWeight: FontWeight.w600,
        color: color,
        letterSpacing: -0.5,
      );

  static TextStyle hudSmall(Color color) => GoogleFonts.inter(
        fontSize: 15,
        fontWeight: FontWeight.w500,
        color: color,
        letterSpacing: 0,
      );

  static TextStyle hudLabel() => GoogleFonts.inter(
        fontSize: 11,
        fontWeight: FontWeight.w600,
        color: labelMuted,
        letterSpacing: 1.5,
      );

  static TextStyle hudUnit() => GoogleFonts.inter(
        fontSize: 14,
        fontWeight: FontWeight.w500,
        color: textSecondary,
        letterSpacing: 0,
      );

  static TextStyle bannerText() => GoogleFonts.inter(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 0,
      );

  // ── Glassmorphism decoration ──────────────────────────────────────────
  static BoxDecoration glassCard({Color? backgroundColor}) => BoxDecoration(
        color: backgroundColor ?? surfaceGlass,
        borderRadius: BorderRadius.circular(24),
      );

  /// Wraps a child in a [ClipRRect] + [BackdropFilter] for glassmorphism.
  static Widget glassWrap({
    required Widget child,
    Color? shadowColor,
    Color? backgroundColor,
    EdgeInsets? padding,
    double borderRadius = 24,
    BorderRadiusGeometry? customBorderRadius,
  }) {
    final effectiveRadius = customBorderRadius ?? BorderRadius.circular(borderRadius);
    
    return Container(
      decoration: BoxDecoration(
        borderRadius: effectiveRadius,
        boxShadow: [
          BoxShadow(
            color: (shadowColor ?? Colors.black).withValues(alpha: 0.2),
            blurRadius: 24,
            spreadRadius: 0,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: effectiveRadius,
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 30, sigmaY: 30),
          child: Container(
            padding: padding ?? const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: backgroundColor ?? surface.withValues(alpha: 0.65), // deeper translucency
            ),
            child: child,
          ),
        ),
      ),
    );
  }

  // ── App-wide ThemeData ────────────────────────────────────────────────
  static ThemeData get darkTheme => ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: background,
        colorScheme: const ColorScheme.dark(
          surface: surface,
          primary: gnssAccent,
          secondary: drAccent,
        ),
        textTheme: GoogleFonts.interTextTheme(
          ThemeData.dark().textTheme,
        ),
      );
}
