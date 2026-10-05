import 'dart:math';
import 'package:flutter/material.dart';

/// A small real-time strip chart drawn with CustomPaint.
///
/// With [minValue]/[maxValue] the vertical scale is fixed (so the trace is
/// readable and comparable between frames) and a zero baseline is drawn;
/// otherwise the scale follows the data.
class SparklineChart extends StatelessWidget {
  final List<double> data;
  final Color lineColor;
  final double height;
  final double width;
  final double? minValue;
  final double? maxValue;

  const SparklineChart({
    super.key,
    required this.data,
    required this.lineColor,
    this.height = 40,
    this.width = 120,
    this.minValue,
    this.maxValue,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      width: width,
      child: CustomPaint(
        painter: _SparklinePainter(
          data: data,
          lineColor: lineColor,
          minValue: minValue,
          maxValue: maxValue,
        ),
      ),
    );
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> data;
  final Color lineColor;
  final double? minValue;
  final double? maxValue;

  _SparklinePainter({
    required this.data,
    required this.lineColor,
    this.minValue,
    this.maxValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.08)
      ..strokeWidth = 1;
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(0, 0, size.width, size.height), const Radius.circular(6)),
      gridPaint..style = PaintingStyle.stroke,
    );

    double lo, hi;
    if (minValue != null && maxValue != null) {
      lo = minValue!;
      hi = maxValue!;
    } else if (data.length >= 2) {
      lo = data.reduce(min);
      hi = data.reduce(max);
      if (hi - lo < 0.01) {
        lo -= 0.5;
        hi += 0.5;
      }
    } else {
      lo = -1;
      hi = 1;
    }
    final range = hi - lo;
    double yOf(double v) => size.height - ((v - lo) / range).clamp(0.0, 1.0) * size.height;

    // Zero baseline.
    if (lo < 0 && hi > 0) {
      final zeroY = yOf(0);
      final zeroPaint = Paint()
        ..color = Colors.white.withValues(alpha: 0.35)
        ..strokeWidth = 1;
      for (double x = 2; x < size.width; x += 6) {
        canvas.drawLine(Offset(x, zeroY), Offset(x + 3, zeroY), zeroPaint);
      }
    }

    if (data.length < 2) return;

    final linePaint = Paint()
      ..color = lineColor
      ..strokeWidth = 1.6
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [lineColor.withValues(alpha: 0.22), lineColor.withValues(alpha: 0.0)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    final baseY = (lo < 0 && hi > 0) ? yOf(0) : size.height;
    final linePath = Path();
    final fillPath = Path();
    for (var i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final y = yOf(data[i]);
      if (i == 0) {
        linePath.moveTo(x, y);
        fillPath.moveTo(x, baseY);
        fillPath.lineTo(x, y);
      } else {
        linePath.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }
    fillPath.lineTo(size.width, baseY);
    fillPath.close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(linePath, linePaint);
  }

  @override
  bool shouldRepaint(covariant _SparklinePainter oldDelegate) => true;
}
