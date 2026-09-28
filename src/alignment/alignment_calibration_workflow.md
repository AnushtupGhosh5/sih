# Detailed Workflow: In-Vehicle Alignment & Calibration (Smartphone Sensors Only)

The **In-Vehicle Alignment & Calibration** engine dynamically computes the rotation matrix that transforms the smartphone's local coordinate frame $(x, y, z)_{phone}$ into the vehicle's reference coordinate frame $(Forward, Lateral, Down)_{vehicle}$. 

Per the system constraints, this module relies **exclusively on built-in smartphone sensors**. To ensure robustness, the workflow is split to handle environments where GPS is available (Situation 1) and where GPS is denied/unavailable (Situation 2).

---

## 1. System Inputs & Coordinate Frames

### 1.1 Inputs (Smartphone Only)
- **3-Axis Accelerometer:** Core input for gravity and linear acceleration.
- **3-Axis Gyroscope:** Core input for rotation, turning detection, and static detection.
- **3-Axis Magnetometer:** Auxiliary input for orientation relative to magnetic North.
- **GPS / GNSS (If available):** Speed and True Heading (1Hz).

### 1.2 Target Vehicle Frame
- $X_{veh}$: Forward (direction of travel)
- $Y_{veh}$: Right (lateral/passenger side)
- $Z_{veh}$: Down (gravity)

---

## 2. Phase 1: Vertical Alignment (Pitch & Roll / Z-axis)

The first objective is to find the vehicle's true Down vector ($Z_{veh}$) by isolating gravity.

### Situation 1: GPS Readings Available
*   **Motion State Detection:** Check if the vehicle is Stationary or at a Constant Velocity.
    *   *Condition:* $GPS_{speed} \approx 0$ (or constant) **AND** $Variance(Accel) < Threshold_{static}$.
*   **Gravity Extraction:** Apply a Low-Pass Filter (LPF) to the accelerometer. Average the readings to isolate the Gravity vector: $\vec{G}_{phone} = [a_x, a_y, a_z]$.

### Situation 2: GPS Readings Unavailable
*   **Motion State Detection (IMU-Only):** Without GPS, we rely purely on IMU kinematics to detect stationary moments (e.g., stopped at a red light).
    *   *Condition:* $Variance(Accel) < Threshold_{strict}$ **AND** $Variance(Gyro) < Threshold_{strict}$ over a 2-second sliding window.
*   **Gravity Extraction:** Capture $\vec{G}_{phone}$ exactly as above during this strictly detected stationary phase.

### Z-Axis Formulation (Both Situations)
Normalize the gravity vector to establish the vehicle's Down axis in the phone's frame:
$$ \hat{Z}_{veh} = \frac{\vec{G}_{phone}}{||\vec{G}_{phone}||} $$

---

## 3. Phase 2: Longitudinal Alignment (Yaw / X-axis)

**Constraint:** We are given only smartphone sensor acceleration ($\vec{A}_{horizontal}$), and the vehicle's trajectory (linear vs. curved) is initially unknown. 
**Goal:** Compute the vehicle's forward axis ($\hat{X}_{veh}$).

Because the trajectory is unknown, $\vec{A}_{horizontal}$ may be a mix of longitudinal acceleration (driving/braking) and lateral acceleration (centripetal forces during cornering). To accurately compute $\hat{X}_{veh}$, we must classify the trajectory and isolate the longitudinal component.

### Step 3.1: Trajectory Classification (Linear vs. Curved)
Use the gyroscope to determine if the vehicle is turning. 
*   Project the raw Gyroscope vector onto the established Down axis to get the vehicle's True Yaw Rate: 
    $$ \omega_{yaw} = \vec{Gyro}_{phone} \cdot \hat{Z}_{veh} $$
*   **Linear Trajectory:** $|\omega_{yaw}| < Threshold_{turn}$ (Vehicle is driving straight).
*   **Curved Trajectory:** $|\omega_{yaw}| \ge Threshold_{turn}$ (Vehicle is cornering).

### Step 3.2: Compute $\hat{X}_{veh}$ (Situation 1: GPS Readings Available)
*   **Linear Trajectory:** When $GPS_{speed}$ indicates positive acceleration, average the $\vec{A}_{horizontal}$ vector. Since there is no cornering force, $\hat{X}_{veh} = Normalize(\vec{A}_{horizontal})$.
*   **Curved Trajectory:** If alignment must occur during a curve, the horizontal acceleration is skewed by centripetal force.
    1.  *GPS Heading (Primary):* Correlate the GPS True Heading with the Magnetometer to deduce the forward axis directly, bypassing skewed accelerometer data.
    2.  *Kinematic Separation:* Calculate the lateral centripetal acceleration magnitude $a_{lat} = GPS_{speed} \cdot \omega_{yaw}$. Subtract this lateral vector component from $\vec{A}_{horizontal}$ to isolate the pure longitudinal vector, then normalize to get $\hat{X}_{veh}$.

### Step 3.3: Compute $\hat{X}_{veh}$ (Situation 2: GPS Readings Unavailable)
Without GPS, we lack absolute velocity, making it mathematically impossible to accurately calculate and subtract the centripetal acceleration ($a_{lat} = v \cdot \omega_{yaw}$) during a curve.
*   **Curved Trajectory:** **Suspend** axis sampling. Discard all $\vec{A}_{horizontal}$ data while $|\omega_{yaw}| \ge Threshold_{turn}$ to prevent unknown lateral forces from corrupting the forward vector computation.
*   **Linear Trajectory:** **Resume** axis sampling. Collect $\vec{A}_{horizontal}$ data into a rolling buffer (e.g., 10-15 seconds of straight-line driving).
    *   **Principal Component Analysis (PCA):** Apply PCA to this filtered 2D acceleration buffer. Because we strictly filtered out curved trajectories, the variance is entirely due to forward driving and braking.
    *   The **First Principal Component** (eigenvector with the largest eigenvalue) perfectly isolates the longitudinal axis.
    *   **Disambiguation:** Assume the first major acceleration spike post-stop is forward, and set $\hat{X}_{veh}$ to point in that direction along the principal axis.

---

## 4. Phase 3: Lateral Alignment & Transformation Matrix

Once $Z_{veh}$ and $X_{veh}$ are found, this phase is identical for both situations.

### Step 4.1: Y-Axis Formulation
The Lateral (Right) vector is calculated using the cross product to ensure a perfectly orthogonal coordinate system:
$$ \hat{Y}_{veh} = \hat{Z}_{veh} \times \hat{X}_{veh} $$

### Step 4.2: Rotation Matrix Generation
Construct the $3 \times 3$ Rotation Matrix ($R_p^v$):
$$ R_p^v = [\hat{X}_{veh}^T \quad \hat{Y}_{veh}^T \quad \hat{Z}_{veh}^T]^T $$

*Note: $R_p^v$ is typically converted into a Quaternion ($q$) to avoid gimbal lock and improve computational efficiency on the smartphone.*

---

## 5. Phase 4: Dynamic Re-Calibration (Displacement Handling)

If the smartphone falls out of its mount or is picked up, the transformation matrix becomes instantly invalid.

*   **Detection:** Monitor for sudden, uncharacteristic spikes in gyro variance paired with high accelerometer noise.
*   **Reaction (Situation 1 - GPS Available):** Invalidate $R_p^v$. Use GPS speed/heading to maintain navigation continuity while recalculating Phase 1 and 2.
*   **Reaction (Situation 2 - GPS Unavailable):** 
    *   Switch the Dead Reckoning engine into "Coast Mode".
    *   Wait for the IMU variance to drop below the static/stable threshold.
    *   Rapidly re-extract gravity ($\hat{Z}_{veh}$), use the Trajectory Classifier to ensure the vehicle is driving straight ($|\omega_{yaw}| < Threshold_{turn}$), and rely on the PCA method on upcoming linear accelerations to re-establish $\hat{X}_{veh}$.

---

## 6. Module Architecture Summary

```mermaid
graph TD
    A[Smartphone IMU & Mag] --> B{GPS Available?}
    B -->|Yes - Sit 1| C(State: Speed=0/Const)
    B -->|No - Sit 2| D(State: IMU Var < Thresh)
    
    C --> E[Extract Gravity Z_veh]
    D --> E
    
    E --> F[Calc Yaw Rate: Gyro • Z_veh]
    F --> G{Is |Yaw Rate| < Thresh?}
    
    G -->|No: Curved Trajectory| H{GPS Available?}
    H -->|Yes| I(Use GPS Heading or calc a_lat = v * yaw_rate)
    H -->|No| J(Suspend Sampling to prevent skew)
    
    G -->|Yes: Linear Trajectory| K{GPS Available?}
    K -->|Yes| L(Average A_horiz during GPS Accel)
    K -->|No| M(Buffer A_horiz -> Apply PCA)
    
    I --> N[Extract Forward X_veh]
    L --> N
    M --> N
    
    N --> O[Cross Product Y = Z x X]
    O --> P[Generate Rotation Matrix R_p^v]
    
    P --> Q{Phone Dropped/Moved?}
    Q -->|Yes| B
    Q -->|No| R(Transformed Data to AI Engine)
```
