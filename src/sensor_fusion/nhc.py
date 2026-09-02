import numpy as np

def apply_eskf_update(P: np.ndarray, H: np.ndarray, R: np.ndarray, innovation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic Error-State Extended Kalman Filter (ES-EKF) measurement update.
    
    Args:
        P: State covariance matrix (15x15)
        H: Observation matrix (mx15)
        R: Measurement noise covariance matrix (mxm)
        innovation: Measurement innovation vector (m,)
        
    Returns:
        tuple containing:
            - delta_x: Error state update (15,)
            - P_new: Updated state covariance matrix (15x15)
            - K: Kalman gain (15xm)
    """
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    delta_x = K @ innovation
    
    I = np.eye(P.shape[0])
    # Joseph form update for numerical stability
    I_KH = I - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    
    return delta_x, P_new, K

class NonHolonomicConstraints:
    """
    Handles Non-Holonomic Constraints (NHC) and Zero Velocity Updates (ZUPT)
    for the Error-State EKF.
    """
    def __init__(self, sigma_lateral: float = 0.1, sigma_vertical: float = 0.1, sigma_zupt: float = 0.01):
        """
        Initialize noise parameters.
        
        Args:
            sigma_lateral: Standard deviation of lateral velocity constraint (m/s)
            sigma_vertical: Standard deviation of vertical velocity constraint (m/s)
            sigma_zupt: Standard deviation of zero velocity constraint (m/s)
        """
        self.sigma_lateral = sigma_lateral
        self.sigma_vertical = sigma_vertical
        self.sigma_zupt = sigma_zupt
        
        # State dimension for the ES-EKF is 15
        self.state_dim = 15

    def compute_nhc_update(self, velocity_body: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies the NHC constraint that lateral and vertical velocity in the body frame are ~ 0.
        
        Args:
            velocity_body: Vehicle velocity in body frame [v_x, v_y, v_z] (Forward, Right, Down)
            P: Current state covariance matrix (15x15)
            
        Returns:
            tuple containing:
                - delta_x: Error state update (15,)
                - P_new: Updated covariance matrix (15x15)
        """
        H_nhc = np.zeros((2, self.state_dim))
        # Select delta_v_y and delta_v_z (indices 4 and 5 in the 15-state vector)
        H_nhc[0, 4] = 1.0
        H_nhc[1, 5] = 1.0
        
        R_nhc = np.diag([self.sigma_lateral**2, self.sigma_vertical**2])
        
        # Innovation: z - h(x)
        innovation = np.array([0.0, 0.0]) - np.array([velocity_body[1], velocity_body[2]])
        
        delta_x, P_new, _ = apply_eskf_update(P, H_nhc, R_nhc, innovation)
        
        return delta_x, P_new

    def compute_zupt_update(self, velocity_ned: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Zero Velocity Update — assumes all velocity components are zero.
        
        Args:
            velocity_ned: Vehicle velocity in NED frame
            P: Current state covariance matrix (15x15)
            
        Returns:
            tuple containing:
                - delta_x: Error state update (15,)
                - P_new: Updated covariance matrix (15x15)
        """
        H_zupt = np.zeros((3, self.state_dim))
        # Select full velocity error (indices 3, 4, 5)
        H_zupt[0, 3] = 1.0
        H_zupt[1, 4] = 1.0
        H_zupt[2, 5] = 1.0
        
        R_zupt = np.diag([self.sigma_zupt**2, self.sigma_zupt**2, self.sigma_zupt**2])
        
        # Innovation: z - h(x)
        innovation = np.array([0.0, 0.0, 0.0]) - velocity_ned
        
        delta_x, P_new, _ = apply_eskf_update(P, H_zupt, R_zupt, innovation)
        
        return delta_x, P_new

    def detect_stationary(self, filtered_speed: float, gyro_magnitude: float, speed_threshold: float = 0.1, gyro_threshold: float = 0.02) -> bool:
        """
        Detects if the vehicle is stationary based on filtered speed and gyro magnitude.
        
        Args:
            filtered_speed: AI-filtered speed estimate (m/s)
            gyro_magnitude: Magnitude of angular velocity (rad/s)
            speed_threshold: Threshold below which speed is considered zero
            gyro_threshold: Threshold below which rotation is considered zero
            
        Returns:
            True if stationary, False otherwise.
        """
        return (abs(filtered_speed) < speed_threshold) and (gyro_magnitude < gyro_threshold)
