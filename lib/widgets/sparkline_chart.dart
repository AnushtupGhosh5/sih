import 'dart:math';
import 'package:flutter/material.dart';

/// A tiny, real-time sparkline chart drawn with CustomPaint.
///
/// Renders a neon-colored line with a gradient fill beneath it,
/// auto-scaling the Y axis to fit the data.
class SparklineChart extends StatelessWidget {
  final List<double> data;
  final Color lineColor;
  final double height;
  final double width;

  const SparklineChart({
    super.key,
    required this.data,
    required this.lineColor,
    this.height = 40,
    this.width = 120,
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
        ),
      ),
    );
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> data;
  final Color lineColor;

  _SparklinePainter({required this.data, required this.lineColor});

  @override
  void paint(Canvas canvas, Size size) {
    if (data.length < 2) return;

    // Draw chart background / border
    final borderPaint = Paint()
      ..color = const Color(0xFF2A2A2A)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), borderPaint);

    // Draw grid lines
    final gridPaint = Paint()
      ..color = const Color(0xFF2A2A2A)
      ..strokeWidth = 1;

    // Horizontal grid
    for (var i = 1; i < 4; i++) {
      final y = size.height * (i / 4);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }
    // Vertical grid
    for (var i = 1; i < 10; i++) {
      final x = size.width * (i / 10);
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }

    final minVal = data.reduce(min);
    final maxVal = data.reduce(max);
    final range = (maxVal - minVal).clamp(0.01, double.infinity);

    // Draw true zero baseline if it falls within the current range.
    if (minVal < 0 && maxVal > 0) {
      final zeroY = size.height - ((0 - minVal) / range) * size.height;
      final zeroPaint = Paint()
        ..color = Colors.white.withValues(alpha: 0.3)
        ..strokeWidth = 1;
      
      for (double x = 0; x < size.width; x += 6) {
        canvas.drawLine(Offset(x, zeroY), Offset(x + 3, zeroY), zeroPaint);
      }
    }

    final linePaint = Paint()
      ..color = lineColor.withValues(alpha: 0.8) // 80% opacity soft white
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          lineColor.withValues(alpha: 0.3),
          lineColor.withValues(alpha: 0.0),
        ],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    final linePath = Path();
    final fillPath = Path();

    for (var i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final y = size.height - ((data[i] - minVal) / range) * size.height;

      if (i == 0) {
        linePath.moveTo(x, y);
        fillPath.moveTo(x, size.height);
        fillPath.lineTo(x, y);
      } else {
        linePath.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }

    // Close fill path.
    fillPath.lineTo(size.width, size.height);
    fillPath.close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(linePath, linePaint);

    // Glow effect removed for a cleaner scientific look, or made very subtle.
    final glowPaint = Paint()
      ..color = lineColor.withValues(alpha: 0.1)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2);
    canvas.drawPath(linePath, glowPaint);
  }

  @override
  bool shouldRepaint(covariant _SparklinePainter oldDelegate) => true;
}
