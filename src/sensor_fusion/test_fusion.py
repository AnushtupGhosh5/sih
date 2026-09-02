"""
Unit tests for the GNSS+INS Sensor Fusion Engine.

Tests cover:
  - Quaternion utilities
  - INS Mechanization (stationary, constant velocity, gravity removal)
  - GNSS Quality Assessment
  - NHC and ZUPT updates
  - ES-EKF predict and update
  - Fusion Engine end-to-end integration
"""

import numpy as np
import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sensor_fusion.ins_mechanization import (
    skew_symmetric,
    quat_multiply,
    quat_to_rotation_matrix,
    rotation_vector_to_quat,
    quat_normalize,
    INSMechanization,
)
from src.sensor_fusion.eskf import (
    ErrorStateKalmanFilter,
    ESKFConfig,
    FusionMode,
)
from src.sensor_fusion.gnss_quality import (
    GNSSQualityAssessor,
    GNSSMeasurement,
    GNSSHealth,
    geodetic_to_ned,
    ned_to_geodetic,
)
from src.sensor_fusion.nhc import NonHolonomicConstraints, apply_eskf_update
from src.sensor_fusion.lstm_predictor import LSTMDriftPredictor
from src.sensor_fusion.fusion_engine import (
    SensorFusionEngine,
    FusionEngineConfig,
    FusionResult,
)


class TestQuaternionUtils(unittest.TestCase):
    """Tests for quaternion utility functions."""

    def test_skew_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        # Skew-symmetric: S + S^T = 0
        np.testing.assert_array_almost_equal(S + S.T, np.zeros((3, 3)))
        # S @ v = 0 (cross product of v with itself)
        np.testing.assert_array_almost_equal(S @ v, np.zeros(3))

    def test_identity_quaternion(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_quat_multiply_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.7071, 0.7071, 0.0, 0.0])
        result = quat_multiply(q, q2)
        np.testing.assert_array_almost_equal(result, q2, decimal=4)

    def test_rotation_vector_zero(self):
        rv = np.zeros(3)
        q = rotation_vector_to_quat(rv)
        np.testing.assert_array_almost_equal(q, [1, 0, 0, 0])

    def test_rotation_vector_90_deg_z(self):
        # 90 degree rotation about Z axis
        rv = np.array([0, 0, np.pi / 2])
        q = rotation_vector_to_quat(rv)
        R = quat_to_rotation_matrix(q)
        # Rotating [1, 0, 0] by 90 about Z should give [0, 1, 0]
        v_rotated = R @ np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(v_rotated, [0, 1, 0], decimal=5)

    def test_quat_normalize(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = quat_normalize(q)
        self.assertAlmostEqual(np.linalg.norm(q_norm), 1.0)


class TestINSMechanization(unittest.TestCase):
    """Tests for the INS Mechanization module."""

    def test_stationary_no_drift(self):
        """A stationary vehicle with only gravity should not move."""
        ins = INSMechanization()
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        ins.initialize(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=q_identity,
        )
        # Apply gravity-only accelerometer reading (body frame = NED when identity quat)
        # In NED with identity rotation, body Z = Down, so accel = [0, 0, 9.81] (gravity)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)

        for _ in range(100):  # 10 seconds at 10Hz
            pos, vel, q = ins.propagate(accel, gyro, dt=0.1)

        # Should remain near origin
        np.testing.assert_array_almost_equal(pos, np.zeros(3), decimal=1)
        np.testing.assert_array_almost_equal(vel, np.zeros(3), decimal=1)

    def test_constant_forward_acceleration(self):
        """Constant forward acceleration should produce expected displacement."""
        ins = INSMechanization()
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        ins.initialize(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=q_identity,
        )
        # 1 m/s^2 forward (North) + gravity
        accel = np.array([1.0, 0.0, 9.81])
        gyro = np.zeros(3)
        dt = 0.1

        for _ in range(100):  # 10 seconds
            pos, vel, q = ins.propagate(accel, gyro, dt)

        # v = a*t = 1.0 * 10 = 10 m/s (North)
        self.assertAlmostEqual(vel[0], 10.0, delta=0.5)
        # p = 0.5 * a * t^2 = 0.5 * 1 * 100 = 50m
        self.assertAlmostEqual(pos[0], 50.0, delta=2.0)

    def test_F_matrix_shape(self):
        ins = INSMechanization()
        ins.initialize(np.zeros(3), np.zeros(3), np.array([1., 0., 0., 0.]))
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        F = ins.compute_F_matrix(accel, gyro, 0.1)
        self.assertEqual(F.shape, (15, 15))

    def test_Q_matrix_shape(self):
        ins = INSMechanization()
        ins.initialize(np.zeros(3), np.zeros(3), np.array([1., 0., 0., 0.]))
        Q = ins.compute_Q_matrix(0.1)
        self.assertEqual(Q.shape, (15, 15))
        # Q should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(Q)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_inject_error(self):
        ins = INSMechanization()
        ins.initialize(np.zeros(3), np.zeros(3), np.array([1., 0., 0., 0.]))
        ins.inject_error(
            delta_pos=np.array([1.0, 2.0, 3.0]),
            delta_vel=np.array([0.1, 0.2, 0.3]),
            delta_theta=np.zeros(3),
            delta_ba=np.zeros(3),
            delta_bg=np.zeros(3),
        )
        state = ins.get_state()
        np.testing.assert_array_almost_equal(state['position'], [1, 2, 3])
        np.testing.assert_array_almost_equal(state['velocity'], [0.1, 0.2, 0.3])


class TestGNSSQuality(unittest.TestCase):
    """Tests for GNSS quality assessment."""

    def test_good_signal(self):
        assessor = GNSSQualityAssessor()
        meas = GNSSMeasurement(
            latitude=28.6,
            longitude=77.2,
            altitude=200.0,
            vel_n=10.0,
            vel_e=0.0,
            vel_d=0.0,
            hdop=1.0,
            num_satellites=12,
            fix_type='3D',
            cn0_mean=40.0,
            timestamp=0.0,
        )
        result = assessor.assess(meas)
        self.assertEqual(result.health, GNSSHealth.GOOD)
        self.assertAlmostEqual(result.scale_factor, 1.0, delta=0.1)

    def test_degraded_high_hdop(self):
        assessor = GNSSQualityAssessor()
        meas = GNSSMeasurement(
            latitude=28.6, longitude=77.2, altitude=200.0,
            vel_n=10.0, vel_e=0.0, vel_d=0.0,
            hdop=6.0, num_satellites=6, fix_type='3D',
            cn0_mean=30.0, timestamp=0.0,
        )
        result = assessor.assess(meas)
        self.assertEqual(result.health, GNSSHealth.DEGRADED)
        self.assertGreater(result.scale_factor, 1.0)

    def test_denied_no_fix(self):
        assessor = GNSSQualityAssessor()
        meas = GNSSMeasurement(
            latitude=0, longitude=0, altitude=0,
            vel_n=0, vel_e=0, vel_d=0,
            hdop=99.0, num_satellites=2, fix_type='none',
            cn0_mean=10.0, timestamp=0.0,
        )
        result = assessor.assess(meas)
        self.assertEqual(result.health, GNSSHealth.DENIED)

    def test_geodetic_ned_roundtrip(self):
        lat_ref, lon_ref, alt_ref = 28.6139, 77.2090, 200.0
        # A point 100m North, 50m East, 10m up from reference
        lat, lon, alt = ned_to_geodetic(100, 50, -10, lat_ref, lon_ref, alt_ref)
        n, e, d = geodetic_to_ned(lat, lon, alt, lat_ref, lon_ref, alt_ref)
        np.testing.assert_array_almost_equal([n, e, d], [100, 50, -10], decimal=0)


class TestNHC(unittest.TestCase):
    """Tests for Non-Holonomic Constraints."""

    def test_nhc_reduces_lateral_velocity(self):
        nhc = NonHolonomicConstraints()
        P = np.eye(15) * 10.0  # Large initial uncertainty
        # Vehicle with lateral velocity of 1 m/s (should be ~0)
        vel_body = np.array([10.0, 1.0, 0.5])
        delta_x, P_new = nhc.compute_nhc_update(vel_body, P)
        # The correction should push lateral velocity toward 0
        # delta_x[4] should be negative (correcting positive lateral vel)
        self.assertLess(delta_x[4], 0)

    def test_zupt_zeros_velocity(self):
        nhc = NonHolonomicConstraints()
        P = np.eye(15) * 10.0
        vel_ned = np.array([0.5, 0.3, 0.1])  # Small residual velocity
        delta_x, P_new = nhc.compute_zupt_update(vel_ned, P)
        # Corrections should push velocity toward zero
        self.assertLess(delta_x[3], 0)
        self.assertLess(delta_x[4], 0)
        self.assertLess(delta_x[5], 0)

    def test_stationary_detection(self):
        nhc = NonHolonomicConstraints()
        self.assertTrue(nhc.detect_stationary(0.05, 0.01))
        self.assertFalse(nhc.detect_stationary(5.0, 0.01))
        self.assertFalse(nhc.detect_stationary(0.05, 0.1))


class TestESKF(unittest.TestCase):
    """Tests for the Error-State EKF."""

    def test_initialization(self):
        eskf = ErrorStateKalmanFilter()
        P = eskf.get_covariance()
        self.assertEqual(P.shape, (15, 15))
        # P should be positive definite
        eigenvalues = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_predict_increases_uncertainty(self):
        eskf = ErrorStateKalmanFilter()
        P_before = eskf.get_covariance().copy()
        F = np.eye(15)
        Q = np.eye(15) * 0.01
        eskf.predict(F, Q)
        P_after = eskf.get_covariance()
        # Uncertainty should increase after prediction
        self.assertGreater(np.trace(P_after), np.trace(P_before))

    def test_gnss_update_reduces_uncertainty(self):
        eskf = ErrorStateKalmanFilter()
        # Inflate covariance
        F = np.eye(15)
        Q = np.eye(15) * 1.0
        for _ in range(10):
            eskf.predict(F, Q)
        P_before = np.trace(eskf.get_covariance())

        # Apply GNSS update
        innovation = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])
        R = np.diag([2.5**2, 2.5**2, 5.0**2, 0.1**2, 0.1**2, 0.1**2])
        delta_x, accepted = eskf.update_gnss(innovation, R)
        P_after = np.trace(eskf.get_covariance())

        self.assertTrue(accepted)
        self.assertGreater(len(delta_x), 0)
        self.assertLess(P_after, P_before)

    def test_outlier_rejection(self):
        eskf = ErrorStateKalmanFilter()
        # Keep covariance small — a huge innovation should be rejected
        innovation = np.array([1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0])
        R = np.diag([2.5**2, 2.5**2, 5.0**2, 0.1**2, 0.1**2, 0.1**2])
        delta_x, accepted = eskf.update_gnss(innovation, R)
        self.assertFalse(accepted)
        np.testing.assert_array_almost_equal(delta_x, np.zeros(15))

    def test_outage_timer(self):
        eskf = ErrorStateKalmanFilter()
        eskf.update_outage_timer(1.0, gnss_available=False)
        eskf.update_outage_timer(1.0, gnss_available=False)
        self.assertAlmostEqual(eskf.get_outage_duration(), 2.0)
        self.assertEqual(eskf.get_mode(), FusionMode.DEAD_RECKONING)

        eskf.update_outage_timer(0.0, gnss_available=True)
        self.assertAlmostEqual(eskf.get_outage_duration(), 0.0)
        self.assertEqual(eskf.get_mode(), FusionMode.GNSS_INS)


class TestLSTMPredictor(unittest.TestCase):
    """Tests for the LSTM drift predictor."""

    def test_no_model_returns_zeros(self):
        """Without a loaded model, predict() should return zeros."""
        predictor = LSTMDriftPredictor(model_path=None)
        for _ in range(60):
            predictor.add_sample(np.zeros(3), np.zeros(3), 0.1)
        pos, vel = predictor.predict()
        np.testing.assert_array_equal(pos, np.zeros(3))
        np.testing.assert_array_equal(vel, np.zeros(3))

    def test_buffer_not_ready(self):
        predictor = LSTMDriftPredictor(model_path=None, seq_len=50)
        predictor.add_sample(np.zeros(3), np.zeros(3), 0.1)
        self.assertFalse(predictor.is_ready())

    def test_buffer_ready(self):
        predictor = LSTMDriftPredictor(model_path=None, seq_len=10)
        for _ in range(10):
            predictor.add_sample(np.zeros(3), np.zeros(3), 0.1)
        # Buffer is full but model not loaded
        self.assertFalse(predictor.is_ready())

    def test_reset(self):
        predictor = LSTMDriftPredictor(model_path=None, seq_len=5)
        for _ in range(5):
            predictor.add_sample(np.ones(3), np.ones(3), 0.1)
        predictor.reset()
        self.assertFalse(predictor.is_ready())


class TestApplyESKFUpdate(unittest.TestCase):
    """Tests for the generic ES-EKF update function."""

    def test_basic_update(self):
        P = np.eye(15) * 10.0
        H = np.zeros((3, 15))
        H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        R = np.eye(3) * 1.0
        innovation = np.array([5.0, 3.0, 1.0])

        delta_x, P_new, K = apply_eskf_update(P, H, R, innovation)

        # Correction should be in the direction of innovation
        self.assertGreater(delta_x[0], 0)
        self.assertGreater(delta_x[1], 0)
        self.assertGreater(delta_x[2], 0)

        # P should decrease
        self.assertLess(np.trace(P_new), np.trace(P))


class TestFusionEngineIntegration(unittest.TestCase):
    """Integration tests for the full Sensor Fusion Engine."""

    def _create_engine(self) -> SensorFusionEngine:
        config = FusionEngineConfig()
        config.apply_nhc = False  # Simplify for testing
        engine = SensorFusionEngine(config)
        engine.initialize(
            latitude=28.6139,
            longitude=77.2090,
            altitude=200.0,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        return engine

    def test_initialization(self):
        engine = self._create_engine()
        self.assertTrue(engine.is_initialized)
        self.assertEqual(engine.get_mode(), FusionMode.GNSS_INS)

    def test_imu_processing(self):
        engine = self._create_engine()
        # Gravity-only reading (stationary)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)

        result = engine.process_imu(accel, gyro, dt=0.1)

        self.assertIsInstance(result, FusionResult)
        self.assertAlmostEqual(result.latitude, 28.6139, places=3)
        self.assertAlmostEqual(result.longitude, 77.2090, places=3)

    def test_gnss_update(self):
        engine = self._create_engine()

        # Process some IMU samples first
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        for _ in range(10):
            engine.process_imu(accel, gyro, dt=0.1)

        # GNSS fix
        gnss = GNSSMeasurement(
            latitude=28.6140,  # slightly different
            longitude=77.2091,
            altitude=200.0,
            vel_n=0.0, vel_e=0.0, vel_d=0.0,
            hdop=1.0, num_satellites=12,
            fix_type='3D', cn0_mean=40.0, timestamp=1.0,
        )
        result = engine.process_gnss(gnss)
        self.assertEqual(result.gnss_health, GNSSHealth.GOOD)

    def test_gnss_denied_triggers_dr(self):
        engine = self._create_engine()

        accel = np.array([1.0, 0.0, 9.81])  # Forward acceleration
        gyro = np.zeros(3)
        engine.set_filtered_speed(5.0)

        # Run without GNSS for several seconds
        for _ in range(20):
            result = engine.process_imu(accel, gyro, dt=0.1)

        self.assertEqual(result.mode, FusionMode.DEAD_RECKONING)
        self.assertGreater(result.outage_duration, 1.0)

    def test_reset(self):
        engine = self._create_engine()
        engine.reset()
        self.assertFalse(engine.is_initialized)


if __name__ == '__main__':
    unittest.main(verbosity=2)

