import 'dart:math' show pi;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart' hide Path, pi;

import '../engine/navigation_engine.dart';
import '../utils/theme.dart';

// Invert + scale + offset matrix to turn OSM light maps into dark charcoal (#1C1C1E) with grey roads (#707070)
const ColorFilter _darkMapFilter = ColorFilter.matrix(<double>[
  -0.070, -0.235, -0.024, 0, 112, // Red channel
  -0.070, -0.235, -0.024, 0, 112, // Green channel
  -0.070, -0.235, -0.024, 0, 114, // Blue channel (slightly more blue for charcoal)
  0, 0, 0, 1, 0, // Alpha channel
]);

/// Live map: smoothly animated vehicle marker, recent GNSS fixes, the
/// dead-reckoned trail, the free-inertial ghost and the filter's uncertainty
/// circle. Follows the vehicle until the user pans; [onUserGesture] lets the
/// parent drop follow mode and [follow] re-enables it.
class HudMap extends StatefulWidget {
  final LatLng position;
  final double heading; // degrees
  final NavMode mode;
  final List<LatLng> trail;
  final List<LatLng> gnssTrack;
  final LatLng? ghost;
  final double? uncertaintyM;
  final bool follow;
  final VoidCallback? onUserGesture;

  const HudMap({
    super.key,
    required this.position,
    required this.heading,
    required this.mode,
    this.trail = const [],
    this.gnssTrack = const [],
    this.ghost,
    this.uncertaintyM,
    this.follow = true,
    this.onUserGesture,
  });

  @override
  State<HudMap> createState() => _HudMapState();
}

class _HudMapState extends State<HudMap> with TickerProviderStateMixin {
  late AnimationController _posController;
  late Animation<double> _latAnim;
  late Animation<double> _lngAnim;
  late AnimationController _headingController;
  late Animation<double> _headingAnim;

  LatLng _currentPos = const LatLng(0, 0);
  double _currentHeading = 0;
  late final MapController _mapController;
  DateTime? _lastPosUpdate;
  DateTime? _lastHeadingUpdate;
  bool _programmaticMove = false;

  @override
  void initState() {
    super.initState();
    _mapController = MapController();
    _currentPos = widget.position;
    _currentHeading = widget.heading;
    _posController = AnimationController(vsync: this, duration: const Duration(milliseconds: 800));
    _headingController = AnimationController(vsync: this, duration: const Duration(milliseconds: 500));
    _latAnim = Tween(begin: _currentPos.latitude, end: _currentPos.latitude).animate(_posController);
    _lngAnim = Tween(begin: _currentPos.longitude, end: _currentPos.longitude).animate(_posController);
    _headingAnim = Tween(begin: _currentHeading, end: _currentHeading).animate(_headingController);
  }

  /// Animation length matched to the update cadence: GNSS fixes arrive at
  /// ~1 Hz and deserve a long eased glide, the engine emits at 10 Hz and
  /// needs a short linear hop or the marker would never catch up.
  static (Duration, Curve) _cadence(DateTime? last, DateTime now, int maxMs) {
    final gap = last == null ? maxMs : now.difference(last).inMilliseconds;
    final ms = gap.clamp(60, maxMs);
    return (Duration(milliseconds: ms), ms < 300 ? Curves.linear : Curves.easeInOutCubic);
  }

  @override
  void didUpdateWidget(HudMap oldWidget) {
    super.didUpdateWidget(oldWidget);
    final now = DateTime.now();

    if (oldWidget.position != widget.position) {
      final (dur, curve) = _cadence(_lastPosUpdate, now, 800);
      _lastPosUpdate = now;
      _posController.duration = dur;
      _latAnim = Tween(begin: _currentPos.latitude, end: widget.position.latitude)
          .animate(CurvedAnimation(parent: _posController, curve: curve));
      _lngAnim = Tween(begin: _currentPos.longitude, end: widget.position.longitude)
          .animate(CurvedAnimation(parent: _posController, curve: curve));
      _posController.forward(from: 0);
    }

    if (oldWidget.heading != widget.heading) {
      final (dur, curve) = _cadence(_lastHeadingUpdate, now, 500);
      _lastHeadingUpdate = now;
      _headingController.duration = dur;
      double diff = widget.heading - _currentHeading;
      if (diff > 180) diff -= 360;
      if (diff < -180) diff += 360;
      _headingAnim = Tween(begin: _currentHeading, end: _currentHeading + diff)
          .animate(CurvedAnimation(parent: _headingController, curve: curve));
      _headingController.forward(from: 0);
    }

    // Re-centre at once when follow mode is switched back on.
    if (!oldWidget.follow && widget.follow) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _move(widget.position));
    }
  }

  void _move(LatLng p) {
    _programmaticMove = true;
    try {
      _mapController.move(p, _mapController.camera.zoom);
    } catch (_) {}
    _programmaticMove = false;
  }

  @override
  void dispose() {
    _posController.dispose();
    _headingController.dispose();
    _mapController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDr = widget.mode == NavMode.deadReckoning;
    final accent = isDr ? HudTheme.drAccent : HudTheme.mapMarker;

    return AnimatedBuilder(
      animation: Listenable.merge([_posController, _headingController]),
      builder: (context, _) {
        final animPos = LatLng(_latAnim.value, _lngAnim.value);
        _currentPos = animPos;
        _currentHeading = _headingAnim.value;

        if (widget.follow) {
          WidgetsBinding.instance.addPostFrameCallback((_) => _move(animPos));
        }

        final unc = widget.uncertaintyM;

        return FlutterMap(
          mapController: _mapController,
          options: MapOptions(
            initialCenter: animPos,
            initialZoom: 17,
            interactionOptions: const InteractionOptions(flags: InteractiveFlag.all),
            onPositionChanged: (camera, hasGesture) {
              if (hasGesture && !_programmaticMove) widget.onUserGesture?.call();
            },
          ),
          children: [
            ColorFiltered(
              colorFilter: _darkMapFilter,
              child: TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'com.example.sih_navigation',
                maxZoom: 19,
                tileProvider: NetworkTileProvider(),
              ),
            ),

            // Filter uncertainty (2 sigma).
            if (unc != null && unc.isFinite && unc > 1)
              CircleLayer(
                circles: [
                  CircleMarker(
                    point: animPos,
                    radius: unc,
                    useRadiusInMeter: true,
                    color: accent.withValues(alpha: 0.10),
                    borderColor: accent.withValues(alpha: 0.45),
                    borderStrokeWidth: 1.5,
                  ),
                ],
              ),

            // Recent GNSS fixes and the dead-reckoning trail.
            PolylineLayer(
              polylines: [
                if (widget.gnssTrack.length >= 2)
                  Polyline(points: widget.gnssTrack, color: HudTheme.gnssTrack, strokeWidth: 3),
                if (widget.trail.length >= 2)
                  Polyline(points: widget.trail, color: HudTheme.drAccent, strokeWidth: 4),
              ],
            ),

            MarkerLayer(
              markers: [
                if (widget.ghost != null)
                  Marker(
                    point: widget.ghost!,
                    width: 16,
                    height: 16,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: HudTheme.drAccent.withValues(alpha: 0.35),
                        border: Border.all(color: HudTheme.drAccent, width: 1.5),
                      ),
                    ),
                  ),
                Marker(
                  point: animPos,
                  width: 48,
                  height: 48,
                  child: Transform.rotate(
                    angle: _currentHeading * (pi / 180),
                    child: _VehicleArrow(color: accent),
                  ),
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}

/// Custom painted arrow / chevron marker for the vehicle.
class _VehicleArrow extends StatefulWidget {
  final Color color;
  const _VehicleArrow({required this.color});

  @override
  State<_VehicleArrow> createState() => _VehicleArrowState();
}

class _VehicleArrowState extends State<_VehicleArrow> with SingleTickerProviderStateMixin {
  late AnimationController _glowController;

  @override
  void initState() {
    super.initState();
    _glowController = AnimationController(vsync: this, duration: const Duration(milliseconds: 1500))
      ..repeat(reverse: true);
  }

  @override
  void dispose() {
    _glowController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _glowController,
      builder: (context, _) => CustomPaint(
        size: const Size(48, 48),
        painter: _ArrowPainter(color: widget.color, glowIntensity: _glowController.value),
      ),
    );
  }
}

class _ArrowPainter extends CustomPainter {
  final Color color;
  final double glowIntensity;
  _ArrowPainter({required this.color, required this.glowIntensity});

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;

    final glowRadius = 14 + (glowIntensity * 6);
    final glowPaint = Paint()
      ..color = color.withValues(alpha: 0.15 + (glowIntensity * 0.15))
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(cx, cy), glowRadius, glowPaint);

    final borderPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    final dotPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(cx, cy), 8, borderPaint);
    canvas.drawCircle(Offset(cx, cy), 6, dotPaint);

    final arrowPath = Path()
      ..moveTo(cx, cy - 24)
      ..lineTo(cx + 6, cy - 10)
      ..lineTo(cx - 6, cy - 10)
      ..close();
    final arrowGlow = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3);
    final arrowPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawPath(arrowPath, arrowGlow);
    canvas.drawPath(arrowPath, arrowPaint);
  }

  @override
  bool shouldRepaint(covariant _ArrowPainter oldDelegate) =>
      oldDelegate.color != color || oldDelegate.glowIntensity != glowIntensity;
}
