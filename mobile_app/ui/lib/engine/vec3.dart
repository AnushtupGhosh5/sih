import 'dart:math' as math;

/// Minimal immutable 3-vector used by the navigation engine.
class Vec3 {
  final double x, y, z;
  const Vec3(this.x, this.y, this.z);

  static const Vec3 zero = Vec3(0, 0, 0);
  static const Vec3 unitX = Vec3(1, 0, 0);
  static const Vec3 unitY = Vec3(0, 1, 0);
  static const Vec3 unitZ = Vec3(0, 0, 1);

  Vec3 operator +(Vec3 o) => Vec3(x + o.x, y + o.y, z + o.z);
  Vec3 operator -(Vec3 o) => Vec3(x - o.x, y - o.y, z - o.z);
  Vec3 operator -() => Vec3(-x, -y, -z);
  Vec3 operator *(double s) => Vec3(x * s, y * s, z * s);

  double dot(Vec3 o) => x * o.x + y * o.y + z * o.z;

  Vec3 cross(Vec3 o) => Vec3(
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
      );

  double get norm2 => dot(this);
  double get norm => math.sqrt(norm2);

  Vec3 get normalized {
    final n = norm;
    return n > 1e-12 ? this * (1.0 / n) : zero;
  }

  /// Component of this vector orthogonal to [unitNormal].
  Vec3 projectOntoPlane(Vec3 unitNormal) => this - unitNormal * dot(unitNormal);

  @override
  String toString() =>
      '(${x.toStringAsFixed(3)}, ${y.toStringAsFixed(3)}, ${z.toStringAsFixed(3)})';
}

/// 3x3 matrix stored as three row vectors.
class Mat3 {
  final Vec3 r0, r1, r2;
  const Mat3(this.r0, this.r1, this.r2);

  static const Mat3 identity = Mat3(Vec3.unitX, Vec3.unitY, Vec3.unitZ);

  Vec3 apply(Vec3 v) => Vec3(r0.dot(v), r1.dot(v), r2.dot(v));

  Mat3 get transpose => Mat3(
        Vec3(r0.x, r1.x, r2.x),
        Vec3(r0.y, r1.y, r2.y),
        Vec3(r0.z, r1.z, r2.z),
      );

  Mat3 multiply(Mat3 o) {
    final t = o.transpose;
    return Mat3(
      Vec3(r0.dot(t.r0), r0.dot(t.r1), r0.dot(t.r2)),
      Vec3(r1.dot(t.r0), r1.dot(t.r1), r1.dot(t.r2)),
      Vec3(r2.dot(t.r0), r2.dot(t.r1), r2.dot(t.r2)),
    );
  }

  /// R = Rz(yaw) * Ry(pitch) * Rx(roll), same convention as the Python test.
  static Mat3 fromEuler({double pitch = 0, double roll = 0, double yaw = 0}) {
    final cp = math.cos(pitch), sp = math.sin(pitch);
    final cr = math.cos(roll), sr = math.sin(roll);
    final cy = math.cos(yaw), sy = math.sin(yaw);
    const one = 1.0;
    final rx = Mat3(const Vec3(one, 0, 0), Vec3(0, cr, -sr), Vec3(0, sr, cr));
    final ry = Mat3(Vec3(cp, 0, sp), const Vec3(0, one, 0), Vec3(-sp, 0, cp));
    final rz = Mat3(Vec3(cy, -sy, 0), Vec3(sy, cy, 0), const Vec3(0, 0, one));
    return rz.multiply(ry).multiply(rx);
  }
}
