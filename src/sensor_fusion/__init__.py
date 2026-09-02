"""
GNSS+INS Sensor Fusion Engine.

Provides AI-augmented Error-State Extended Kalman Filter (ES-EKF) based
sensor fusion for seamless navigation during GNSS availability, degraded
signals, and complete GNSS blackouts.

Main entry point:
    from src.sensor_fusion import SensorFusionEngine, FusionEngineConfig

Modules:
    - fusion_engine: Main orchestrator
    - eskf: Error-State Extended Kalman Filter
    - ins_mechanization: Strapdown INS integration
    - gnss_quality: GNSS signal quality assessment
    - nhc: Non-Holonomic Constraints & ZUPT
    - lstm_predictor: LSTM drift prediction (inference)
"""

from .fusion_engine import SensorFusionEngine, FusionEngineConfig, FusionResult
from .eskf import ErrorStateKalmanFilter, ESKFConfig, FusionMode
from .gnss_quality import (
    GNSSQualityAssessor,
    GNSSMeasurement,
    GNSSHealth,
    GNSSQualityResult,
    geodetic_to_ned,
    ned_to_geodetic,
)
from .ins_mechanization import INSMechanization
from .nhc import NonHolonomicConstraints
from .lstm_predictor import LSTMDriftPredictor

__all__ = [
    # Main
    'SensorFusionEngine',
    'FusionEngineConfig',
    'FusionResult',
    # ES-EKF
    'ErrorStateKalmanFilter',
    'ESKFConfig',
    'FusionMode',
    # GNSS
    'GNSSQualityAssessor',
    'GNSSMeasurement',
    'GNSSHealth',
    'GNSSQualityResult',
    'geodetic_to_ned',
    'ned_to_geodetic',
    # INS
    'INSMechanization',
    # NHC
    'NonHolonomicConstraints',
    # LSTM
    'LSTMDriftPredictor',
]
