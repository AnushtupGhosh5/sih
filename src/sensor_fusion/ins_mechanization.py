import numpy as np
from typing import Dict, Tuple, Optional

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Returns the 3x3 skew-symmetric matrix of a vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=float)

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion multiplication [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion to 3x3 DCM."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=float)

def rotation_vector_to_quat(rv: np.ndarray) -> np.ndarray:
    """Small rotation vector to quaternion."""
    angle = np.linalg.norm(rv)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    sin_half = np.sin(angle / 2)
    cos_half = np.cos(angle / 2)
    return np.array([cos_half, axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half])

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion."""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

class INSMechanization:
    """
    Strapdown INS Mechanization (high-rate prediction loop).
    Propagates position, velocity, and attitude using IMU measurements.
    """
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gravity = np.array([0.0, 0.0, 9.81])

    def initialize(self, position: np.ndarray, velocity: np.ndarray, quaternion: np.ndarray, 
                   accel_bias: Optional[np.ndarray] = None, gyro_bias: Optional[np.ndarray] = None):
        """Set initial state."""
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.quaternion = quat_normalize(np.array(quaternion, dtype=float))
        if accel_bias is not None:
            self.accel_bias = np.array(accel_bias, dtype=float)
        if gyro_bias is not None:
            self.gyro_bias = np.array(gyro_bias, dtype=float)

    def propagate(self, accel_veh: np.ndarray, gyro_veh: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One time step of strapdown integration.
        """
        # 1. Bias compensation
        a_corrected = accel_veh - self.accel_bias
        w_corrected = gyro_veh - self.gyro_bias

        # 2. Attitude update
        q_rot = rotation_vector_to_quat(w_corrected * dt)
        q_new = quat_multiply(self.quaternion, q_rot)
        q_new = quat_normalize(q_new)

        # 3. Rotate specific force to NED
        R = quat_to_rotation_matrix(self.quaternion)
        a_ned = R @ a_corrected

        # 4. Velocity update
        v_new = self.velocity + (a_ned - self.gravity) * dt

        # 5. Position update
        p_new = self.position + self.velocity * dt + 0.5 * (a_ned - self.gravity) * (dt ** 2)

        # 6. Update internal state
        self.position = p_new
        self.velocity = v_new
        self.quaternion = q_new

        return self.position, self.velocity, self.quaternion

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns dict with current state."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.quaternion.copy(),
            'accel_bias': self.accel_bias.copy(),
            'gyro_bias': self.gyro_bias.copy()
        }

    def update_biases(self, delta_accel_bias: np.ndarray, delta_gyro_bias: np.ndarray):
        """Add corrections to biases."""
        self.accel_bias += delta_accel_bias
        self.gyro_bias += delta_gyro_bias

    def inject_error(self, delta_pos: np.ndarray, delta_vel: np.ndarray, delta_theta: np.ndarray, 
                     delta_ba: np.ndarray, delta_bg: np.ndarray):
        """Applies ES-EKF corrections."""
        self.position += delta_pos
        self.velocity += delta_vel
        
        # Quaternion correction via q (x) rotation_vector_to_quat(delta_theta)
        q_err = rotation_vector_to_quat(delta_theta)
        self.quaternion = quat_normalize(quat_multiply(self.quaternion, q_err))
        
        self.accel_bias += delta_ba
        self.gyro_bias += delta_bg

    def compute_F_matrix(self, accel_corrected: np.ndarray, gyro_corrected: np.ndarray, dt: float) -> np.ndarray:
        """
        Returns the 15x15 state transition matrix F_k.
        State vector: [p, v, theta, ba, bg]
        """
        F = np.eye(15)
        R = quat_to_rotation_matrix(self.quaternion)
        
        # Position from velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity from attitude error
        a_skew = skew_symmetric(accel_corrected)
        F[3:6, 6:9] = -R @ a_skew * dt
        
        # Velocity from accel bias
        F[3:6, 9:12] = -R * dt
        
        # Attitude error from gyro bias
        w_skew = skew_symmetric(gyro_corrected)
        F[6:9, 6:9] = np.eye(3) - w_skew * dt
        
        # Attitude error from gyro bias
        F[6:9, 12:15] = -np.eye(3) * dt
        
        return F

    def compute_Q_matrix(self, dt: float, sigma_a_noise: float = 0.02, sigma_g_noise: float = 0.001, 
                         sigma_a_walk: float = 0.001, sigma_g_walk: float = 0.0001) -> np.ndarray:
        """
        Returns 15x15 process noise matrix Q_k.
        """
        Q = np.zeros((15, 15))
        
        # Velocity noise (accel noise)
        Q[3:6, 3:6] = np.eye(3) * (sigma_a_noise ** 2) * (dt ** 2)
        
        # Attitude noise (gyro noise)
        Q[6:9, 6:9] = np.eye(3) * (sigma_g_noise ** 2) * (dt ** 2)
        
        # Accel bias random walk
        Q[9:12, 9:12] = np.eye(3) * (sigma_a_walk ** 2) * dt
        
        # Gyro bias random walk
        Q[12:15, 12:15] = np.eye(3) * (sigma_g_walk ** 2) * dt
        
        return Q
