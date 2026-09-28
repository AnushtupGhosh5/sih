import sys
import os

# Ensure the parent directory is in the path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.alignment.alignment_engine import InVehicleAligner, AlignmentState

def create_rotation_matrix(pitch, roll, yaw):
    """Utility to create a 3D rotation matrix for testing."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def test_alignment():
    print("--- Starting Alignment Test ---")
    
    # 1. Define true orientation of the phone in the vehicle
    # Imagine the phone is mounted haphazardly on a dashboard mount
    true_pitch = np.radians(30) # Tilted 30 degrees up
    true_roll = np.radians(10)  # Leaning 10 degrees right
    true_yaw = np.radians(45)   # Pointing 45 degrees off-center
    
    R_veh_to_phone = create_rotation_matrix(true_pitch, true_roll, true_yaw)
    
    # 2. Initialize the Aligner Engine
    aligner = InVehicleAligner(sampling_rate=10.0)
    print(f"Initial State: {aligner.state.name}")
    
    # 3. Simulate Phase 1: Stationary Vehicle (Extract Gravity)
    print("\n[Simulating] Vehicle is stationary for 3 seconds...")
    g_veh = np.array([0.0, 0.0, 9.81]) # Pure gravity pointing down
    g_phone = R_veh_to_phone @ g_veh   # What the smartphone actually feels
    
    for _ in range(30): # 3 seconds at 10Hz
        noise_accel = np.random.normal(0, 0.01, 3)
        noise_gyro = np.random.normal(0, 0.005, 3)
        aligner.process_frame(g_phone + noise_accel, noise_gyro, gps_speed=0.0)
    
    print(f"Current State: {aligner.state.name}")
    
    # 4. Simulate Phase 2: Forward Acceleration (Extract X-axis)
    print("\n[Simulating] Vehicle accelerates straight forward (GPS denied environment)...")
    # Vehicle accelerating forward at 2.5 m/s^2, while still experiencing gravity
    accel_veh = np.array([2.5, 0.0, 9.81]) 
    accel_phone = R_veh_to_phone @ accel_veh
    
    # We need 10 seconds of data to trigger the PCA since we pass gps_speed=None (Situation 2)
    aligned = False
    for i in range(110): 
        noise_accel = np.random.normal(0, 0.02, 3)
        noise_gyro = np.random.normal(0, 0.005, 3)
        aligned, R_est = aligner.process_frame(accel_phone + noise_accel, noise_gyro, gps_speed=None)
        if aligned:
            break
            
    print(f"Current State: {aligner.state.name}")
    
    # 5. Verify the Transformation
    print("\n--- Verification ---")
    if aligned:
        # Test Case: The vehicle hits a pothole (sharp spike in the Down/Z direction)
        # while driving forward at constant speed.
        test_accel_veh = np.array([0.0, 0.0, 15.0]) # 15.0 m/s^2 on Z axis
        test_accel_phone = R_veh_to_phone @ test_accel_veh # The confusing signal the phone records
        
        # Ask our engine to make sense of the phone signal
        recovered_accel_veh = aligner.transform_reading(test_accel_phone)
        
        print(f"Ground Truth Vehicle Accel : {test_accel_veh}")
        print(f"Raw Phone Sensor Accel     : {np.round(test_accel_phone, 3)}")
        print(f"Engine Recovered Accel     : {np.round(recovered_accel_veh, 3)}")
        
        # Check alignment error
        error = np.linalg.norm(test_accel_veh - recovered_accel_veh)
        print(f"Transformation Error       : {error:.5f}")
        
        if error < 0.2:
            print("\n[SUCCESS] TEST PASSED: The alignment engine successfully reversed the arbitrary phone mounting angle!")
        else:
            print("\n[FAILED] TEST FAILED: Transformation mismatch.")
    else:
        print("\n[FAILED] TEST FAILED: Engine did not reach FULLY_ALIGNED state.")

if __name__ == "__main__":
    test_alignment()

