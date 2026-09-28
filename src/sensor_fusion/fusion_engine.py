"""
GNSS+INS Sensor Fusion Engine — Main Orchestrator.

This is the top-level module that ties together all sensor fusion components:
  - INS Mechanization (strapdown integration)
  - Error-State Extended Kalman Filter
  - GNSS Quality Assessment
  - Non-Holonomic Constraints & ZUPT
  - LSTM Drift Predictor (AI augmentation)

Usage:
    engine = SensorFusionEngine()
    engine.initialize(lat, lon, alt, quaternion)

    # Per IMU sample (10Hz / 200Hz):
    result = engine.process_imu(accel_veh, gyro_veh, dt)

    # Per GNSS fix (1Hz):
    result = engine.process_gnss(gnss_measurement)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging

from .ins_mechanization import (
    INSMechanization,
    quat_to_rotation_matrix,
    rotation_vector_to_quat,
    quat_normalize,
)
from .eskf import ErrorStateKalmanFilter, ESKFConfig, FusionMode
from .gnss_quality import (
    GNSSQualityAssessor,
    GNSSMeasurement,
    GNSSHealth,
    geodetic_to_ned,
    ned_to_geodetic,
)
from .nhc import NonHolonomicConstraints
from .lstm_predictor import LSTMDriftPredictor

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of a single fusion processing step."""
    # Fused navigation solution
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))

    # Uncertainty
    position_std: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    velocity_std: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s

    # Bias estimates
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Mode & diagnostics
    mode: FusionMode = FusionMode.GNSS_INS
    gnss_health: GNSSHealth = GNSSHealth.DENIED
    outage_duration: float = 0.0
    timestamp: float = 0.0


@dataclass
class FusionEngineConfig:
    """Configuration for the Sensor Fusion Engine."""
    # ES-EKF config
    eskf_config: ESKFConfig = field(default_factory=ESKFConfig)

    # NHC parameters
    nhc_sigma_lateral: float = 0.1      # m/s
    nhc_sigma_vertical: float = 0.1     # m/s
    zupt_sigma: float = 0.01            # m/s

    # GNSS quality thresholds
    gnss_sigma_pos: float = 2.5         # m
    gnss_sigma_alt: float = 5.0         # m
    gnss_sigma_vel: float = 0.1         # m/s
    gnss_hdop_threshold: float = 4.0
    gnss_cn0_threshold: float = 25.0

    # AI augmentation
    lstm_model_path: Optional[str] = None
    lstm_backend: str = 'onnx'
    lstm_seq_len: int = 50

    # NHC application
    apply_nhc: bool = True
    nhc_min_speed: float = 0.5  # Don't apply NHC below this speed (parking maneuvers)

    # ZUPT detection
    zupt_speed_threshold: float = 0.1   # m/s
    zupt_gyro_threshold: float = 0.02   # rad/s


class SensorFusionEngine:
    """
    Main Sensor Fusion Engine orchestrating GNSS+INS fusion.
    
    Processes IMU data at high rate (10-200Hz) and GNSS fixes at low rate (1Hz),
    producing continuous navigation solutions with bounded drift.
    """

    def __init__(self, config: Optional[FusionEngineConfig] = None):
        self.config = config or FusionEngineConfig()

        # Core components
        self._ins = INSMechanization()
        self._eskf = ErrorStateKalmanFilter(self.config.eskf_config)
        self._gnss_assessor = GNSSQualityAssessor(
            sigma_pos=self.config.gnss_sigma_pos,
            sigma_alt=self.config.gnss_sigma_alt,
            sigma_vel=self.config.gnss_sigma_vel,
            hdop_threshold=self.config.gnss_hdop_threshold,
            cn0_threshold=self.config.gnss_cn0_threshold,
        )
        self._nhc = NonHolonomicConstraints(
            sigma_lateral=self.config.nhc_sigma_lateral,
            sigma_vertical=self.config.nhc_sigma_vertical,
            sigma_zupt=self.config.zupt_sigma,
        )
        self._lstm = LSTMDriftPredictor(
            model_path=self.config.lstm_model_path,
            backend=self.config.lstm_backend,
            seq_len=self.config.lstm_seq_len,
        )

        # Reference geodetic origin (set at initialization)
        self._ref_lat: float = 0.0
        self._ref_lon: float = 0.0
        self._ref_alt: float = 0.0
        self._ref_set: bool = False

        # State tracking
        self._initialized: bool = False
        self._timestamp: float = 0.0
        self._filtered_speed: float = 0.0  # from AI Speed Filter

        # Last GNSS measurement for diagnostics
        self._last_gnss: Optional[GNSSMeasurement] = None
        self._last_gnss_health: GNSSHealth = GNSSHealth.DENIED

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    def initialize(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        quaternion: np.ndarray,
        velocity_ned: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> None:
        """
        Initialize the fusion engine with a known position and orientation.
        
        Args:
            latitude: Initial latitude in degrees
            longitude: Initial longitude in degrees
            altitude: Initial altitude in meters (above WGS84 ellipsoid)
            quaternion: Initial attitude quaternion [w, x, y, z] (body-to-NED)
            velocity_ned: Initial NED velocity [v_N, v_E, v_D] in m/s
            timestamp: Initial timestamp in seconds
        """
        # Set geodetic reference origin
        self._ref_lat = latitude
        self._ref_lon = longitude
        self._ref_alt = altitude
        self._ref_set = True

        # Initial position in NED (origin, so [0, 0, 0])
        position_ned = np.zeros(3)

        vel = velocity_ned if velocity_ned is not None else np.zeros(3)

        # Initialize INS mechanization
        self._ins.initialize(
            position=position_ned,
            velocity=vel,
            quaternion=quat_normalize(np.array(quaternion, dtype=float)),
        )

        # Initialize ES-EKF
        self._eskf.reset()

        # Reset AI predictor buffer
        self._lstm.reset()

        self._timestamp = timestamp
        self._initialized = True
        self._filtered_speed = 0.0

        logger.info(
            "Fusion engine initialized at (%.6f, %.6f, %.1f)",
            latitude, longitude, altitude,
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------ #
    #  External Speed Input (from AI Speed & Vibration Filter)             #
    # ------------------------------------------------------------------ #

    def set_filtered_speed(self, speed: float) -> None:
        """
        Update the filtered forward speed from the AI Speed & Vibration Filter.
        Used for ZUPT detection and NHC gating.
        
        Args:
            speed: Filtered forward speed in m/s
        """
        self._filtered_speed = speed

    # ------------------------------------------------------------------ #
    #  IMU Processing (High-Rate Loop)                                     #
    # ------------------------------------------------------------------ #

    def process_imu(
        self,
        accel_veh: np.ndarray,
        gyro_veh: np.ndarray,
        dt: float,
        timestamp: Optional[float] = None,
    ) -> FusionResult:
        """
        Process a single IMU sample. Called at IMU rate (10Hz or 200Hz).
        
        This executes:
          1. INS Mechanization (attitude, velocity, position propagation)
          2. ES-EKF covariance prediction
          3. NHC constraint update (if enabled and vehicle is moving)
          4. ZUPT update (if vehicle is stationary)
          5. LSTM buffer accumulation
          6. LSTM pseudo-measurement (if in DR mode)

        Args:
            accel_veh: (3,) accelerometer in vehicle frame [fwd, right, down] m/s²
            gyro_veh: (3,) gyroscope in vehicle frame [fwd, right, down] rad/s
            dt: time step in seconds
            timestamp: optional absolute timestamp

        Returns:
            FusionResult with current navigation solution
        """
        if not self._initialized:
            raise RuntimeError("Fusion engine not initialized. Call initialize() first.")

        accel = np.asarray(accel_veh, dtype=float)
        gyro = np.asarray(gyro_veh, dtype=float)

        if timestamp is not None:
            self._timestamp = timestamp
        else:
            self._timestamp += dt

        # ---- Step 1: INS Mechanization ----
        pos, vel, quat = self._ins.propagate(accel, gyro, dt)

        # ---- Step 2: ES-EKF Covariance Prediction ----
        ins_state = self._ins.get_state()
        accel_corrected = accel - ins_state['accel_bias']
        gyro_corrected = gyro - ins_state['gyro_bias']

        F = self._ins.compute_F_matrix(accel_corrected, gyro_corrected, dt)
        Q = self._ins.compute_Q_matrix(dt)
        self._eskf.predict(F, Q)

        # ---- Step 3: NHC Update ----
        if self.config.apply_nhc and self._filtered_speed > self.config.nhc_min_speed:
            # Get velocity in vehicle/body frame
            R_nav2body = quat_to_rotation_matrix(quat).T
            vel_body = R_nav2body @ vel
            lateral_vel = vel_body[1]   # right
            vertical_vel = vel_body[2]  # down

            R_nhc = np.diag([
                self.config.nhc_sigma_lateral ** 2,
                self.config.nhc_sigma_vertical ** 2,
            ])
            delta_x = self._eskf.update_nhc(lateral_vel, vertical_vel, R_nhc)
            self._ins.inject_error(
                delta_pos=delta_x[0:3],
                delta_vel=delta_x[3:6],
                delta_theta=delta_x[6:9],
                delta_ba=delta_x[9:12],
                delta_bg=delta_x[12:15],
            )

        # ---- Step 4: ZUPT ----
        gyro_mag = float(np.linalg.norm(gyro))
        is_stationary = self._nhc.detect_stationary(
            self._filtered_speed, gyro_mag,
            speed_threshold=self.config.zupt_speed_threshold,
            gyro_threshold=self.config.zupt_gyro_threshold,
        )
        if is_stationary:
            R_zupt = np.diag([self.config.zupt_sigma ** 2] * 3)
            delta_x = self._eskf.update_zupt(vel, R_zupt)
            self._ins.inject_error(
                delta_pos=delta_x[0:3],
                delta_vel=delta_x[3:6],
                delta_theta=delta_x[6:9],
                delta_ba=delta_x[9:12],
                delta_bg=delta_x[12:15],
            )

        # ---- Step 5: LSTM Buffer ----
        self._lstm.add_sample(accel_corrected, gyro_corrected, dt)

        # ---- Step 6: LSTM Pseudo-Measurement (DR mode) ----
        self._eskf.update_outage_timer(dt, gnss_available=False)

        if (self._eskf.get_mode() == FusionMode.DEAD_RECKONING
                and self._lstm.is_ready()):
            pos_corr, vel_corr = self._lstm.predict()
            if np.any(pos_corr != 0) or np.any(vel_corr != 0):
                delta_x = self._eskf.update_pseudo_measurement(
                    pos_corr, vel_corr,
                    self._eskf.get_outage_duration(),
                )
                self._ins.inject_error(
                    delta_pos=delta_x[0:3],
                    delta_vel=delta_x[3:6],
                    delta_theta=delta_x[6:9],
                    delta_ba=delta_x[9:12],
                    delta_bg=delta_x[12:15],
                )

        # ---- Build Result ----
        return self._build_result()

    # ------------------------------------------------------------------ #
    #  GNSS Processing (Low-Rate Loop)                                     #
    # ------------------------------------------------------------------ #

    def process_gnss(self, measurement: GNSSMeasurement) -> FusionResult:
        """
        Process a GNSS measurement. Called at ~1Hz.
        
        This executes:
          1. GNSS quality assessment
          2. Geodetic to NED conversion
          3. ES-EKF measurement update (if GOOD or DEGRADED)
          4. Error-state injection into INS
          5. Outage timer update

        Args:
            measurement: GNSSMeasurement dataclass with position, velocity, quality

        Returns:
            FusionResult with updated navigation solution
        """
        if not self._initialized:
            raise RuntimeError("Fusion engine not initialized. Call initialize() first.")

        self._last_gnss = measurement

        # ---- Step 1: Quality Assessment ----
        quality = self._gnss_assessor.assess(measurement)
        self._last_gnss_health = quality.health

        if quality.health == GNSSHealth.DENIED:
            # No valid fix — skip update, stay in DR mode
            self._eskf.update_outage_timer(0.0, gnss_available=False)
            logger.debug("GNSS DENIED — staying in DR mode")
            return self._build_result()

        # ---- Step 2: Geodetic to NED ----
        gnss_ned = geodetic_to_ned(
            measurement.latitude, measurement.longitude, measurement.altitude,
            self._ref_lat, self._ref_lon, self._ref_alt,
        )
        gnss_vel = np.array([measurement.vel_n, measurement.vel_e, measurement.vel_d])

        # ---- Step 3: Compute Innovation ----
        ins_state = self._ins.get_state()
        pos_innovation = gnss_ned - ins_state['position']
        vel_innovation = gnss_vel - ins_state['velocity']
        innovation = np.concatenate([pos_innovation, vel_innovation])

        # ---- Step 4: ES-EKF Update ----
        R = quality.R_adaptive

        # Optional: LSTM augmentation during GOOD/DEGRADED mode
        if self._lstm.is_ready():
            pos_corr, vel_corr = self._lstm.predict()
            lstm_correction = np.concatenate([pos_corr, vel_corr])
            beta = self.config.eskf_config.lstm_blend_weight
            innovation = innovation + beta * lstm_correction

        delta_x, accepted = self._eskf.update_gnss(innovation, R)

        if accepted:
            self._ins.inject_error(
                delta_pos=delta_x[0:3],
                delta_vel=delta_x[3:6],
                delta_theta=delta_x[6:9],
                delta_ba=delta_x[9:12],
                delta_bg=delta_x[12:15],
            )
            logger.debug(
                "GNSS %s update accepted (NIS=%.2f)",
                quality.health.name, self._eskf.state.last_nis,
            )
        else:
            logger.debug(
                "GNSS update rejected (NIS=%.2f > %.2f)",
                self._eskf.state.last_nis, self.config.eskf_config.chi2_threshold,
            )

        # ---- Step 5: Outage Timer ----
        gnss_available = quality.health in (GNSSHealth.GOOD, GNSSHealth.DEGRADED)
        self._eskf.update_outage_timer(0.0, gnss_available=gnss_available)

        return self._build_result()

    # ------------------------------------------------------------------ #
    #  Result Builder                                                      #
    # ------------------------------------------------------------------ #

    def _build_result(self) -> FusionResult:
        """Construct a FusionResult from the current internal state."""
        ins_state = self._ins.get_state()
        pos_ned = ins_state['position']
        vel_ned = ins_state['velocity']
        quat = ins_state['quaternion']

        # Convert NED position back to geodetic
        lat, lon, alt = ned_to_geodetic(
            pos_ned[0], pos_ned[1], pos_ned[2],
            self._ref_lat, self._ref_lon, self._ref_alt,
        )

        return FusionResult(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            velocity_ned=vel_ned.copy(),
            quaternion=quat.copy(),
            position_std=self._eskf.get_position_uncertainty(),
            velocity_std=self._eskf.get_velocity_uncertainty(),
            accel_bias=ins_state['accel_bias'].copy(),
            gyro_bias=ins_state['gyro_bias'].copy(),
            mode=self._eskf.get_mode(),
            gnss_health=self._last_gnss_health,
            outage_duration=self._eskf.get_outage_duration(),
            timestamp=self._timestamp,
        )

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    def get_covariance(self) -> np.ndarray:
        """Return the 15x15 error-state covariance matrix."""
        return self._eskf.get_covariance()

    def get_ins_state(self) -> Dict[str, Any]:
        """Return the full INS nominal state as a dictionary."""
        return self._ins.get_state()

    def get_mode(self) -> FusionMode:
        """Return current operating mode (GNSS_INS, DEGRADED, or DEAD_RECKONING)."""
        return self._eskf.get_mode()

    def get_outage_duration(self) -> float:
        """Return seconds since last valid GNSS fix."""
        return self._eskf.get_outage_duration()

    def reset(self) -> None:
        """Full reset of the fusion engine."""
        self._eskf.reset()
        self._lstm.reset()
        self._initialized = False
        self._ref_set = False
        self._timestamp = 0.0
        self._filtered_speed = 0.0
        self._last_gnss = None
        self._last_gnss_health = GNSSHealth.DENIED
        logger.info("Fusion engine reset.")

