import numpy as np
from collections import deque
from enum import Enum

class AlignmentState(Enum):
    UNCALIBRATED = 0
    Z_ALIGNED = 1       # Gravity vector extracted
    FULLY_ALIGNED = 2   # X and Y axes established, Rotation Matrix ready

class InVehicleAligner:
    """
    In-Vehicle Alignment & Calibration Engine.
    Dynamically computes the transformation matrix from the smartphone's
    local frame to the vehicle's frame (Forward, Right, Down) using IMU data.
    """
    
    def __init__(self, sampling_rate=10.0):
        self.sampling_rate = sampling_rate
        self.state = AlignmentState.UNCALIBRATED
        
        # Calibration Thresholds
        self.static_accel_var_thresh = 0.05  # m/s^2 variance threshold
        self.static_gyro_var_thresh = 0.01   # rad/s variance threshold
        self.turn_yaw_rate_thresh = 0.05     # rad/s threshold for identifying curved trajectory
        
        # Sliding Window Buffers
        self.window_size = int(sampling_rate * 2) # 2-second short window for gravity/static detection
        self.accel_buffer = deque(maxlen=self.window_size)
        self.gyro_buffer = deque(maxlen=self.window_size)
        
        # X-axis estimation buffers (Long window for PCA - e.g., 10 seconds of linear driving)
        self.pca_window_size = int(sampling_rate * 10) 
        self.horizontal_accel_buffer = deque(maxlen=self.pca_window_size)
        
        # Axes and Transformation Matrix
        self.z_veh = None
        self.x_veh = None
        self.y_veh = None
        self.R_p2v = np.eye(3) # Rotation matrix: Phone frame to Vehicle frame
        
    def process_frame(self, accel, gyro, gps_speed=None):
        """
        Process a single timestamp frame of sensor data.
        
        Args:
            accel (list/np.array): [x, y, z] in m/s^2
            gyro (list/np.array): [x, y, z] in rad/s
            gps_speed (float, optional): Vehicle speed in m/s.
            
        Returns:
            tuple: (is_aligned (bool), R_p2v (np.ndarray or None))
        """
        self.accel_buffer.append(np.array(accel))
        self.gyro_buffer.append(np.array(gyro))
        
        # Wait until we have enough data to fill the short window
        if len(self.accel_buffer) < self.window_size:
            return False, None
            
        if self.state == AlignmentState.UNCALIBRATED:
            self._attempt_z_alignment(gps_speed)
            
        elif self.state == AlignmentState.Z_ALIGNED:
            self._attempt_x_alignment(gps_speed)
            
        elif self.state == AlignmentState.FULLY_ALIGNED:
            self._check_displacement()
            
        if self.state == AlignmentState.FULLY_ALIGNED:
            return True, self.R_p2v
        return False, None
        
    def _attempt_z_alignment(self, gps_speed):
        """Phase 1: Vertical Alignment (Pitch & Roll)"""
        accel_var = np.var(self.accel_buffer, axis=0).sum()
        gyro_var = np.var(self.gyro_buffer, axis=0).sum()
        
        is_stationary = False
        if gps_speed is not None:
            # Situation 1: GPS Available
            if gps_speed < 0.2 and accel_var < self.static_accel_var_thresh:
                is_stationary = True
        else:
            # Situation 2: GPS Unavailable (Strict IMU variance check)
            if accel_var < self.static_accel_var_thresh and gyro_var < self.static_gyro_var_thresh:
                is_stationary = True
                
        if is_stationary:
            # Gravity vector is the mean of the stationary accelerometer data
            gravity = np.mean(self.accel_buffer, axis=0)
            norm = np.linalg.norm(gravity)
            if norm > 0:
                self.z_veh = gravity / norm
                self.state = AlignmentState.Z_ALIGNED
                print("[Alignment] Z-Axis (Gravity) Extracted & Aligned.")
                
    def _attempt_x_alignment(self, gps_speed):
        """Phase 2: Longitudinal Alignment (Yaw) with Trajectory Classification"""
        current_accel = self.accel_buffer[-1]
        current_gyro = self.gyro_buffer[-1]
        
        # Step 2.1: Trajectory Classification (Linear vs Curved)
        # Yaw rate is the gyro rotation projected around the true vertical (z_veh) axis
        yaw_rate = np.dot(current_gyro, self.z_veh)
        is_linear = abs(yaw_rate) < self.turn_yaw_rate_thresh
        
        # Remove gravity to get pure linear acceleration
        gravity_vector = self.z_veh * 9.81 
        lin_accel = current_accel - gravity_vector
        
        # Project linear acceleration onto the horizontal plane (orthogonal to Z)
        a_horiz = lin_accel - (np.dot(lin_accel, self.z_veh) * self.z_veh)
        
        if gps_speed is not None:
            # Situation 1: GPS Available
            if is_linear:
                # Detect forward acceleration using GPS delta (simplified here as raw speed > threshold)
                if gps_speed > 1.0 and np.linalg.norm(a_horiz) > 0.5: 
                    self.x_veh = a_horiz / np.linalg.norm(a_horiz)
                    self._finalize_alignment()
            else:
                # Curved Trajectory: Suspend for safety, or implement a_lat = v * yaw_rate separation
                # To be robust, we simply wait for a linear segment to align X.
                pass 
        else:
            # Situation 2: GPS Unavailable
            if is_linear:
                # Resume axis sampling
                self.horizontal_accel_buffer.append(a_horiz)
                
                # Check if we have enough linear data for PCA
                if len(self.horizontal_accel_buffer) >= self.pca_window_size:
                    self._apply_pca_for_x_axis()
            else:
                # Curved Trajectory: Suspend sampling to prevent centripetal forces from skewing X_veh
                pass

    def _apply_pca_for_x_axis(self):
        """Calculates the First Principal Component to isolate the forward axis."""
        data = np.array(self.horizontal_accel_buffer)
        
        # Center the data
        data_mean = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_mean, rowvar=False)
        
        # Extract Eigenvalues & Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by largest eigenvalue (First Principal Component)
        sort_indices = np.argsort(eigenvalues)[::-1]
        principal_axis = eigenvectors[:, sort_indices[0]]
        
        # Disambiguate Forward vs Backward
        # Heuristic: Braking spikes are typically sharper/larger than forward acceleration.
        # We project data onto the axis; if the minimum (negative) is larger than the maximum (positive),
        # it means the sharp spikes are pointing backward, so the axis is correctly pointing forward.
        # Otherwise, flip it.
        proj = np.dot(data, principal_axis)
        if np.abs(np.min(proj)) > np.max(proj):
            principal_axis = -principal_axis
            
        self.x_veh = principal_axis / np.linalg.norm(principal_axis)
        self._finalize_alignment()

    def _finalize_alignment(self):
        """Phase 3: Lateral Alignment & Rotation Matrix Generation"""
        # Y = Z x X (Right hand rule)
        self.y_veh = np.cross(self.z_veh, self.x_veh)
        self.y_veh /= np.linalg.norm(self.y_veh)
        
        # Ensure perfect orthogonality for X = Y x Z
        self.x_veh = np.cross(self.y_veh, self.z_veh)
        self.x_veh /= np.linalg.norm(self.x_veh)
        
        # Construct Transformation Matrix R_p2v
        self.R_p2v = np.vstack([self.x_veh, self.y_veh, self.z_veh])
        self.state = AlignmentState.FULLY_ALIGNED
        print("[Alignment] Fully Aligned. Rotation Matrix R_p2v generated.")

    def _check_displacement(self):
        """Phase 4: Dynamic Re-Calibration trigger"""
        # A sudden massive spike in gyro variance indicates the phone fell or was moved
        gyro_var = np.var(self.gyro_buffer, axis=0).sum()
        
        # Using 50x the static threshold as a simplistic drop/move detection
        if gyro_var > self.static_gyro_var_thresh * 50: 
            print("[Alignment] Displacement detected! Invalidating matrix and recalibrating.")
            self.state = AlignmentState.UNCALIBRATED
            self.horizontal_accel_buffer.clear()
            self.R_p2v = np.eye(3)

    def transform_reading(self, raw_vector):
        """
        Transforms a 3D raw sensor vector (Accel or Gyro) from the 
        Phone's frame into the Vehicle's frame.
        """
        if self.state != AlignmentState.FULLY_ALIGNED:
            return raw_vector # Return raw if uncalibrated
            
        # V_veh = R_p2v * V_phone
        return np.dot(self.R_p2v, raw_vector)

