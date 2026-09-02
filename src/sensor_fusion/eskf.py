"""
Error-State Extended Kalman Filter (ES-EKF) for GNSS+INS Fusion.

This module implements the core Kalman filter logic:
  - State prediction (covariance propagation)
  - Measurement update with chi-squared gating (Joseph form)
  - Error-state injection into the nominal INS state
  - AI-augmented pseudo-measurement integration

The 15-dimensional error state vector is:
    δx = [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class FusionMode(Enum):
    """Current operating mode of the fusion engine."""
    GNSS_INS = 0       # Normal GNSS-aided INS
    DEGRADED = 1        # GNSS available but low quality
    DEAD_RECKONING = 2  # Pure INS + AI correction


@dataclass
class ESKFConfig:
    """Configuration parameters for the Error-State EKF."""
    # Initial covariance diagonals
    sigma_pos: float = 10.0         # m
    sigma_vel: float = 1.0          # m/s
    sigma_att: float = 0.0873       # rad (~5 degrees)
    sigma_ba: float = 0.5           # m/s^2
    sigma_bg: float = 0.01          # rad/s

    # IMU noise parameters (smartphone MEMS defaults)
    sigma_a_noise: float = 0.02     # m/s^2/sqrt(Hz)
    sigma_g_noise: float = 0.001    # rad/s/sqrt(Hz)
    sigma_a_walk: float = 0.001     # m/s^3/sqrt(Hz)
    sigma_g_walk: float = 0.0001    # rad/s^2/sqrt(Hz)

    # Chi-squared gating threshold (6 DoF, 99% confidence)
    chi2_threshold: float = 16.81

    # AI augmentation blending weight (0 = pure EKF, 1 = pure LSTM)
    lstm_blend_weight: float = 0.2

    # Pseudo-measurement confidence growth rate during GNSS outage
    pseudo_confidence_growth: float = 1.0


@dataclass
class ESKFState:
    """Internal state of the Error-State EKF."""
    # Error-state vector (15,) — always reset to zero after injection
    delta_x: np.ndarray = field(default_factory=lambda: np.zeros(15))

    # Error-state covariance (15x15)
    P: np.ndarray = field(default_factory=lambda: np.eye(15))

    # Current operating mode
    mode: FusionMode = FusionMode.GNSS_INS

    # GNSS outage timer (seconds since last valid GNSS fix)
    outage_duration: float = 0.0

    # Innovation statistics for monitoring
    last_innovation: Optional[np.ndarray] = None
    last_nis: float = 0.0  # Normalized Innovation Squared


class ErrorStateKalmanFilter:
    """
    Error-State Extended Kalman Filter for GNSS+INS sensor fusion.
    
    Implements the predict-update cycle with:
      - High-rate covariance propagation (IMU rate: 10Hz or 200Hz)
      - Low-rate GNSS measurement updates (1Hz)
      - Non-Holonomic Constraint updates (IMU rate)
      - Zero Velocity Updates (when stationary)
      - AI-augmented pseudo-measurements (during GNSS outages)
    """

    STATE_DIM = 15  # [δp(3), δv(3), δθ(3), δba(3), δbg(3)]

    def __init__(self, config: Optional[ESKFConfig] = None):
        self.config = config or ESKFConfig()
        self.state = ESKFState()
        self._initialize_covariance()

    def _initialize_covariance(self) -> None:
        """Set the initial error-state covariance matrix P_0."""
        cfg = self.config
        diag = np.array([
            cfg.sigma_pos**2, cfg.sigma_pos**2, cfg.sigma_pos**2,     # position
            cfg.sigma_vel**2, cfg.sigma_vel**2, cfg.sigma_vel**2,     # velocity
            cfg.sigma_att**2, cfg.sigma_att**2, cfg.sigma_att**2,     # attitude
            cfg.sigma_ba**2,  cfg.sigma_ba**2,  cfg.sigma_ba**2,      # accel bias
            cfg.sigma_bg**2,  cfg.sigma_bg**2,  cfg.sigma_bg**2,      # gyro bias
        ])
        self.state.P = np.diag(diag)
        self.state.delta_x = np.zeros(self.STATE_DIM)

    def reset(self) -> None:
        """Reset the filter to its initial state."""
        self._initialize_covariance()
        self.state.mode = FusionMode.GNSS_INS
        self.state.outage_duration = 0.0
        self.state.last_innovation = None
        self.state.last_nis = 0.0

    # ------------------------------------------------------------------ #
    #  PREDICT (Time Update)                                               #
    # ------------------------------------------------------------------ #

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        """
        Propagate the error-state covariance using the state transition
        matrix F and process noise Q computed by INS Mechanization.
        
        P_{k+1|k} = F_k @ P_k @ F_k.T + Q_k
        
        The error-state itself remains at zero (it's only estimated at
        measurement update time).

        Args:
            F: 15x15 state transition matrix from INSMechanization.compute_F_matrix()
            Q: 15x15 process noise matrix from INSMechanization.compute_Q_matrix()
        """
        self.state.P = F @ self.state.P @ F.T + Q

        # Ensure symmetry (numerical drift prevention)
        self.state.P = 0.5 * (self.state.P + self.state.P.T)

    # ------------------------------------------------------------------ #
    #  UPDATE (Measurement Correction)                                     #
    # ------------------------------------------------------------------ #

    def update_gnss(
        self,
        innovation: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Perform a GNSS measurement update.
        
        The innovation is z = [P_gnss - P_ins; V_gnss - V_ins] (6x1).
        The observation matrix H is [I_3 0 0 0 0; 0 I_3 0 0 0] (6x15).

        Args:
            innovation: (6,) measurement residual [delta_pos(3), delta_vel(3)]
            R: (6,6) measurement noise covariance (adaptive)

        Returns:
            Tuple of (delta_x correction (15,), accepted: bool)
        """
        H = np.zeros((6, self.STATE_DIM))
        H[0:3, 0:3] = np.eye(3)   # position observation
        H[3:6, 3:6] = np.eye(3)   # velocity observation

        return self._measurement_update(innovation, H, R)

    def update_nhc(
        self,
        lateral_vel: float,
        vertical_vel: float,
        R_nhc: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Non-Holonomic Constraint update.
        Constrains lateral and vertical velocity to approximately zero.
        
        Args:
            lateral_vel: Current lateral (right) velocity in vehicle frame (m/s)
            vertical_vel: Current vertical (down) velocity in vehicle frame (m/s)
            R_nhc: (2,2) NHC measurement noise covariance

        Returns:
            delta_x correction (15,)
        """
        H = np.zeros((2, self.STATE_DIM))
        H[0, 4] = 1.0   # lateral velocity (v_E in NED ~ v_right in vehicle)
        H[1, 5] = 1.0   # vertical velocity (v_D)

        innovation = np.array([0.0 - lateral_vel, 0.0 - vertical_vel])

        delta_x, _ = self._measurement_update(innovation, H, R_nhc, gate=False)
        return delta_x

    def update_zupt(
        self,
        velocity_ned: np.ndarray,
        R_zupt: np.ndarray,
    ) -> np.ndarray:
        """
        Zero Velocity Update — all velocity components should be zero
        when the vehicle is stationary.
        
        Args:
            velocity_ned: (3,) current NED velocity estimate
            R_zupt: (3,3) ZUPT measurement noise covariance

        Returns:
            delta_x correction (15,)
        """
        H = np.zeros((3, self.STATE_DIM))
        H[0, 3] = 1.0   # v_N
        H[1, 4] = 1.0   # v_E
        H[2, 5] = 1.0   # v_D

        innovation = -velocity_ned  # z = 0 - v

        delta_x, _ = self._measurement_update(innovation, H, R_zupt, gate=False)
        return delta_x

    def update_pseudo_measurement(
        self,
        lstm_pos_correction: np.ndarray,
        lstm_vel_correction: np.ndarray,
        outage_duration: float,
    ) -> np.ndarray:
        """
        Apply LSTM-predicted pseudo-measurement during GNSS outage.
        
        The LSTM predicts the accumulated drift [δp, δv]. We treat the
        corrected position/velocity as a pseudo-GNSS measurement.
        Confidence degrades with outage duration.

        Args:
            lstm_pos_correction: (3,) predicted position drift (NED)
            lstm_vel_correction: (3,) predicted velocity drift (NED)
            outage_duration: seconds since GNSS outage began

        Returns:
            delta_x correction (15,)
        """
        cfg = self.config

        H = np.zeros((6, self.STATE_DIM))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        innovation = np.concatenate([lstm_pos_correction, lstm_vel_correction])

        # Confidence degrades with outage duration
        gamma = 1.0 + cfg.pseudo_confidence_growth * outage_duration ** 1.5
        R_pseudo = np.diag([
            (2.5 * gamma)**2, (2.5 * gamma)**2, (5.0 * gamma)**2,
            (0.1 * gamma)**2, (0.1 * gamma)**2, (0.1 * gamma)**2,
        ])

        delta_x, _ = self._measurement_update(innovation, H, R_pseudo, gate=False)
        return delta_x

    # ------------------------------------------------------------------ #
    #  Core Measurement Update (Joseph Form)                               #
    # ------------------------------------------------------------------ #

    def _measurement_update(
        self,
        innovation: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        gate: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        Generic ES-EKF measurement update with Joseph-form covariance update.
        
        Args:
            innovation: (m,) measurement residual
            H: (m, 15) observation matrix
            R: (m, m) measurement noise covariance
            gate: whether to apply chi-squared gating

        Returns:
            Tuple of (delta_x (15,), accepted: bool)
        """
        P = self.state.P
        m = innovation.shape[0]

        # Innovation covariance
        S = H @ P @ H.T + R

        # Chi-squared gating
        if gate:
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return np.zeros(self.STATE_DIM), False

            nis = float(innovation.T @ S_inv @ innovation)
            self.state.last_nis = nis
            self.state.last_innovation = innovation.copy()

            if nis > self.config.chi2_threshold:
                # Outlier — reject measurement
                return np.zeros(self.STATE_DIM), False

        # Kalman Gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.zeros(self.STATE_DIM), False

        K = P @ H.T @ S_inv

        # Error-state correction
        delta_x = K @ innovation

        # Joseph-form covariance update (numerically stable)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.state.P = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Ensure symmetry
        self.state.P = 0.5 * (self.state.P + self.state.P.T)

        return delta_x, True

    # ------------------------------------------------------------------ #
    #  State Accessors                                                     #
    # ------------------------------------------------------------------ #

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the current error-state covariance matrix."""
        return self.state.P.copy()

    def get_position_uncertainty(self) -> np.ndarray:
        """Return position standard deviations (3,) in meters."""
        return np.sqrt(np.diag(self.state.P)[0:3])

    def get_velocity_uncertainty(self) -> np.ndarray:
        """Return velocity standard deviations (3,) in m/s."""
        return np.sqrt(np.diag(self.state.P)[3:6])

    def get_mode(self) -> FusionMode:
        """Return the current fusion operating mode."""
        return self.state.mode

    def set_mode(self, mode: FusionMode) -> None:
        """Set the fusion operating mode."""
        self.state.mode = mode

    def update_outage_timer(self, dt: float, gnss_available: bool) -> None:
        """
        Update the GNSS outage timer.
        
        Args:
            dt: time step in seconds
            gnss_available: True if a valid GNSS fix was received this step
        """
        if gnss_available:
            self.state.outage_duration = 0.0
            if self.state.mode == FusionMode.DEAD_RECKONING:
                self.state.mode = FusionMode.GNSS_INS
        else:
            self.state.outage_duration += dt
            if self.state.outage_duration > 1.0:
                self.state.mode = FusionMode.DEAD_RECKONING

    def get_outage_duration(self) -> float:
        """Return seconds since last valid GNSS fix."""
        return self.state.outage_duration

