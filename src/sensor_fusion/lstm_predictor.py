"""
LSTM drift predictor module for sensor fusion.

This module wraps a trained LSTM model to predict position and velocity drift
corrections based on recent IMU measurements. It supports both ONNX Runtime
(preferred for edge deployment) and PyTorch JIT as inference backends.
"""

import collections
import warnings
from typing import Optional, Tuple, Deque

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LSTMDriftPredictor:
    """
    Predicts drift corrections using an LSTM model.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = 'onnx',
        seq_len: int = 50,
        input_features: int = 9
    ) -> None:
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the .onnx or .pt model file. If None, runs in passthrough mode.
            backend: The inference backend to use ('onnx' or 'pytorch').
            seq_len: Expected input sequence length.
            input_features: Number of features per timestep.
        """
        self.seq_len = seq_len
        self.input_features = input_features
        self._buffer: Deque[np.ndarray] = collections.deque(maxlen=seq_len)
        self._model_loaded = False
        self._backend = backend.lower()
        self._session = None
        self._model = None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Loads the model using the specified backend."""
        if self._backend == 'onnx':
            if not HAS_ONNX:
                warnings.warn("onnxruntime is not installed. Model not loaded.")
                return
            try:
                self._session = ort.InferenceSession(model_path)
                self._model_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load ONNX model: {e}")
        elif self._backend == 'pytorch':
            if not HAS_TORCH:
                warnings.warn("torch is not installed. Model not loaded.")
                return
            try:
                self._model = torch.jit.load(model_path)
                self._model.eval()
                self._model_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load PyTorch model: {e}")
        else:
            warnings.warn(f"Unsupported backend: {self._backend}")

    def add_sample(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """
        Adds one timestep of IMU data to the ring buffer.
        
        Args:
            accel: Acceleration vector (3,) in m/s^2.
            gyro: Angular velocity vector (3,) in rad/s.
            dt: Time delta since the last sample in seconds.
        """
        accel_norm = float(np.linalg.norm(accel))
        gyro_norm = float(np.linalg.norm(gyro))
        
        feature_vector = np.array([
            accel[0], accel[1], accel[2],
            gyro[0], gyro[1], gyro[2],
            accel_norm, gyro_norm, dt
        ], dtype=np.float32)
        
        self._buffer.append(feature_vector)

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on the accumulated buffer if ready.
        
        Returns:
            A tuple of (position_correction (3,), velocity_correction (3,)).
            Returns zeros if the model is not loaded or the buffer is not full.
        """
        if not self.is_ready():
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            
        # Build input tensor of shape (1, seq_len, input_features)
        input_array = np.stack(self._buffer).astype(np.float32)
        input_batch = np.expand_dims(input_array, axis=0)
        
        try:
            if self._backend == 'onnx':
                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: input_batch})
                output_array = outputs[0][0]  # shape (6,)
            elif self._backend == 'pytorch':
                with torch.no_grad():
                    input_tensor = torch.from_numpy(input_batch)
                    output_tensor = self._model(input_tensor)
                    output_array = output_tensor.numpy()[0]
            else:
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            
            if len(output_array) >= 6:
                pos_corr = output_array[:3]
                vel_corr = output_array[3:6]
            else:
                pos_corr = np.zeros(3, dtype=np.float32)
                vel_corr = np.zeros(3, dtype=np.float32)
                
            return pos_corr, vel_corr
            
        except Exception as e:
            warnings.warn(f"Inference failed: {e}")
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def is_ready(self) -> bool:
        """Checks if the buffer is full and the model is loaded."""
        return self._model_loaded and len(self._buffer) == self.seq_len

    def reset(self) -> None:
        """Clears the internal sample buffer."""
        self._buffer.clear()
