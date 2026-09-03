# Detailed Workflow for Intelligent Dead Reckoning (IDR) System with GNSS Fusion

This document details the end-to-end workflow required to build and deploy the Intelligent Dead Reckoning (IDR) system with GNSS Fusion, based on the project problem statement.

The workflow is divided into two primary phases: Cloud/Desktop Model Training and On-Device/Edge Execution.

## Phase 1: Model Training (Cloud/Desktop Apriori)

The training phase involves processing raw IMU and GNSS data to develop the AI/ML models capable of accurately estimating speed and correcting drift.

### 1. Data Preparation and Preprocessing
*   **Data Sourcing:** Utilize the provided IO-VNBD dataset (Inertial and Odometry Benchmark Dataset) which includes synchronized and unsynchronized vehicle ECU and smartphone sensor data.
*   **Data Cleaning:** Handle missing values, synchronize timestamp data between different sensors (Accelerometer, Gyroscope, Magnetometer, GNSS), and format the data for training.
*   **Feature Engineering:** Extract relevant time-series features from the 10Hz IMU data (and 200Hz FOG data for the edge engine).

### 2. AI Speed & Vibration Filter Training
*   **Model Architecture:** Design a deep-learning (e.g., LSTM, 1D-CNN) or statistical signal-processing model.
*   **Objective:** Train the model to filter out high-frequency road noise, engine idling vibrations, and pothole shocks.
*   **Output:** The model must accurately predict the vehicle's forward velocity and acceleration profiles exclusively from noisy smartphone accelerometer and gyroscope inputs, bypassing the need for OBD-II speedometer feeds.

### 3. GNSS+INS Fusion Engine Training
*   **Model Architecture:** Develop an AI-based sensor fusion algorithm (e.g., AI-enhanced Extended/Unscented Kalman Filter or a purely neural sensor fusion model).
*   **Objective:** Combine GNSS and IMU measurements to mitigate exponential drift errors inherent in consumer-grade IMUs.
*   **Output:** Accurate and continuous position and velocity estimates.

### 4. Model Optimization and Export
*   **Quantization and Compression:** Optimize the trained models for mobile and edge environments to ensure they are lightweight and performant.
*   **Export:** Export the models into a mobile-friendly format (e.g., TensorFlow Lite, ONNX) for the smartphone application, and a suitable format for the standalone edge engine.

---

## Phase 2: On-Device Execution & Inference (Smartphone / Edge)

This phase occurs during real-time deployment (e.g., during the SIH Finale), where the pre-trained models execute inference locally on the smartphone or edge device.

### 1. In-Vehicle Alignment & Calibration
*   **Initialization:** When the application starts, automatically compute the smartphone's pitch, roll, and yaw relative to the vehicle's driving direction.
*   **Adaptability:** Ensure the calibration works regardless of whether the phone is securely mounted on the dashboard or placed loosely in a mobile holder.

### 2. Real-Time Data Ingestion
*   **Sensors:** Continuously receive live inputs from the phone's built-in IMU (Accelerometer, Gyroscope, Magnetometer/Compass) and GNSS receiver.
*   **Edge Engine:** For the edge deployable software engine, ingest data from external IMU sensors (e.g., FOG-based IMU at ~200Hz).

### 3. Real-Time Filtering and Speed Prediction
*   **Inference:** Pass the live, noisy IMU data through the deployed AI Speed & Vibration Filter.
*   **Output:** Obtain clean estimates of the vehicle's forward velocity and acceleration.

### 4. Seamless GNSS Deficit Handling
*   **Monitoring:** Continuously monitor the health and availability of the GNSS signal.
*   **Transition:** Upon detecting a GNSS signal blackout (e.g., entering a tunnel), trigger an instant, sub-millisecond transition from GNSS-aided INS to pure Dead Reckoning mode, and vice-versa when the signal returns.

### 5. Advanced Map-Matching & Kinematic Constraints
*   **Offline Maps:** Utilize an offline map database (e.g., downloaded OpenStreetMap).
*   **Constraint Application:** Apply Non-Holonomic Constraints (NHC) (e.g., a car cannot slide sideways) and overlay the inertial trajectory onto the known road network using algorithms like Hidden Markov Map Matching or Unscented Kalman Filters.
*   **Correction:** Dramatically snap any drifting IMU path back onto the actual road grid.

### 6. Real-Time Navigation Interface (UI)
*   **Display:** Feed the corrected position data to the mobile application's frontend.
*   **User Experience:** Render a smooth, uninterrupted vehicle icon on the map UI, demonstrating seamless navigation even during complete GNSS dropouts.

---

## Performance Benchmarks to Achieve
*   **Dead Reckoning Drift:** Less than 10% of total distance traveled during a GNSS blackout.
    *   *Example 1:* < 5m drift over a 50m GNSS-denied environment in < 1 min.
    *   *Example 2:* < 100m drift over a 1km GNSS-denied environment at 60kmph (e.g., tunnels).
*   **Update Rates:** 10Hz position update rate on the smartphone mobile app, and higher update rates (~200Hz) on the Edge deployable software engine.

