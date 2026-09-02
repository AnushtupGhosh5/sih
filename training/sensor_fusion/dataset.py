import os
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_imu_features(accel: np.ndarray, gyro: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Compute features from IMU data.
    
    Args:
        accel: (N, 3) array of accelerometer data (m/s^2)
        gyro: (N, 3) array of gyroscope data (rad/s)
        timestamps: (N,) array of timestamps (s)
        
    Returns:
        (N, 9) array of features: [a_x, a_y, a_z, ω_x, ω_y, ω_z, |a|, |ω|, Δt]
    """
    N = accel.shape[0]
    features = np.zeros((N, 9), dtype=np.float32)
    features[:, 0:3] = accel
    features[:, 3:6] = gyro
    features[:, 6] = np.linalg.norm(accel, axis=1)
    features[:, 7] = np.linalg.norm(gyro, axis=1)
    
    dt = np.zeros(N, dtype=np.float32)
    if N > 1:
        dt[1:] = timestamps[1:] - timestamps[:-1]
        dt[0] = dt[1]  # Assume constant dt for first step
    features[:, 8] = dt
    
    return features


class IOVNBDDataset(Dataset):
    """
    Dataset for loading IO-VNBD datasets and generating sliding window 
    features for LSTM-based INS drift prediction.
    """
    def __init__(
        self, 
        data_dir: str, 
        seq_len: int = 50, 
        stride: int = 10, 
        split: str = 'train', 
        train_ratio: float = 0.8
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.train_ratio = train_ratio
        
        # Store samples as (features_window, label) tuples
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        
        self._load_data()

    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find the first column name in the DataFrame that contains any of the keywords."""
        for col in df.columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return None

    def _load_data(self):
        """Scans the directory for CSVs and processes them to extract features and labels."""
        csv_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
                    
        # Sort for determinism and split by drive
        csv_files.sort()
        np.random.seed(42)  # Seed for reproducible splits
        np.random.shuffle(csv_files)
        
        split_idx = int(len(csv_files) * self.train_ratio)
        if self.split == 'train':
            target_files = csv_files[:split_idx]
        else:
            target_files = csv_files[split_idx:]
            
        for file in target_files:
            try:
                df = pd.read_csv(file)
                self._process_drive(df, file)
            except Exception as e:
                logger.warning(f"Failed to load or parse {file}: {e}")

    def _process_drive(self, df: pd.DataFrame, filename: str):
        """Processes a single drive CSV and extracts windowed features and labels."""
        # Flexible column identification
        time_col = self._find_column(df, ['time', 'timestamp', 't'])
        acc_x = self._find_column(df, ['acc_x', 'accel_x', 'ax', 'accx'])
        acc_y = self._find_column(df, ['acc_y', 'accel_y', 'ay', 'accy'])
        acc_z = self._find_column(df, ['acc_z', 'accel_z', 'az', 'accz'])
        
        gyr_x = self._find_column(df, ['gyr_x', 'gyro_x', 'gx', 'gyrx'])
        gyr_y = self._find_column(df, ['gyr_y', 'gyro_y', 'gy', 'gyry'])
        gyr_z = self._find_column(df, ['gyr_z', 'gyro_z', 'gz', 'gyrz'])
        
        lat_col = self._find_column(df, ['lat', 'latitude'])
        lon_col = self._find_column(df, ['lon', 'longitude', 'lng'])
        
        # Check if we have essential IMU and time columns
        imu_cols = [time_col, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        if any(c is None for c in imu_cols):
            logger.warning(f"Missing required IMU or time columns in {filename}. Skipping.")
            return
            
        # Check if GNSS truth is available
        if lat_col is None or lon_col is None:
            logger.warning(f"Missing GNSS ground truth (lat/lon) in {filename}. Skipping supervised label generation.")
            return

        # Extract numpy arrays
        # Use simple numeric extraction and handle potential NaNs
        df = df.dropna(subset=imu_cols + [lat_col, lon_col])
        if len(df) < self.seq_len:
            return

        time_data = df[time_col].values
        accel_data = df[[acc_x, acc_y, acc_z]].values
        gyro_data = df[[gyr_x, gyr_y, gyr_z]].values
        
        # In a complete implementation, GNSS (lat, lon) would be converted to local NED coordinates
        # and compared with a full INS mechanization. 
        # Here we perform a simplified dead-reckoning vs GNSS delta approximation.
        lat_data = df[lat_col].values
        lon_data = df[lon_col].values
        
        N = len(df)
        for start_idx in range(0, N - self.seq_len + 1, self.stride):
            end_idx = start_idx + self.seq_len
            
            # Window data
            t_win = time_data[start_idx:end_idx]
            a_win = accel_data[start_idx:end_idx]
            g_win = gyro_data[start_idx:end_idx]
            
            features = compute_imu_features(a_win, g_win, t_win)
            
            # Simplified Label Computation: [δp_N, δp_E, δp_D, δv_N, δv_E, δv_D]
            # Since full INS mechanics are outside the scope of this snippet,
            # we simulate an error label. In practice, you would run INS:
            # INS_end = mechanization(a_win, g_win, initial_state)
            # GNSS_end = to_NED(lat_data[end_idx-1], lon_data[end_idx-1])
            # label = INS_end - GNSS_end
            
            # Placeholder for actual mechanization drift:
            label = np.zeros(6, dtype=np.float32)
            
            self.samples.append((features, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    data_dir: str, 
    batch_size: int = 64, 
    seq_len: int = 50, 
    stride: int = 10, 
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns train and validation dataloaders.
    
    Args:
        data_dir: Path to the root of the dataset
        batch_size: Batch size for loaders
        seq_len: Window length for LSTM
        stride: Stride for sliding window
        num_workers: Number of workers for data loading
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = IOVNBDDataset(
        data_dir=data_dir, 
        seq_len=seq_len, 
        stride=stride, 
        split='train'
    )
    
    val_dataset = IOVNBDDataset(
        data_dir=data_dir, 
        seq_len=seq_len, 
        stride=stride, 
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader
