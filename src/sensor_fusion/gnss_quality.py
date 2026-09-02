"""
GNSS quality assessment module.
Classifies GNSS signal quality and computes adaptive measurement noise.
"""
from enum import Enum
from dataclasses import dataclass
import numpy as np
import math

class GNSSHealth(Enum):
    """Enumeration representing GNSS signal health."""
    GOOD = 1
    DEGRADED = 2
    DENIED = 3

@dataclass
class GNSSMeasurement:
    """Dataclass holding GNSS measurement data."""
    latitude: float
    longitude: float
    altitude: float
    vel_n: float
    vel_e: float
    vel_d: float
    hdop: float
    num_satellites: int
    fix_type: str
    cn0_mean: float
    timestamp: float

@dataclass
class GNSSQualityResult:
    """Dataclass holding the result of a GNSS quality assessment."""
    health: GNSSHealth
    R_adaptive: np.ndarray
    scale_factor: float

class GNSSQualityAssessor:
    """Assesses GNSS signal quality and provides adaptive measurement noise."""
    
    def __init__(self, sigma_pos: float = 2.5, sigma_alt: float = 5.0, 
                 sigma_vel: float = 0.1, hdop_threshold: float = 4.0, 
                 cn0_threshold: float = 25.0, chi2_threshold: float = 16.81):
        """
        Initializes the GNSS quality assessor with base parameters.
        
        Args:
            sigma_pos: Base standard deviation for position (meters).
            sigma_alt: Base standard deviation for altitude (meters).
            sigma_vel: Base standard deviation for velocity (m/s).
            hdop_threshold: Threshold above which HDOP is considered degraded.
            cn0_threshold: Threshold below which CN0 is considered degraded.
            chi2_threshold: Threshold for chi-squared innovation check.
        """
        self.sigma_pos = sigma_pos
        self.sigma_alt = sigma_alt
        self.sigma_vel = sigma_vel
        self.hdop_threshold = hdop_threshold
        self.cn0_threshold = cn0_threshold
        self.chi2_threshold = chi2_threshold

    def _compute_R_base(self) -> np.ndarray:
        """Computes the base measurement noise covariance matrix (6x6)."""
        return np.diag([
            self.sigma_pos ** 2,
            self.sigma_pos ** 2,
            self.sigma_alt ** 2,
            self.sigma_vel ** 2,
            self.sigma_vel ** 2,
            self.sigma_vel ** 2
        ])

    def _compute_scale_factor(self, hdop: float, cn0_mean: float) -> float:
        """
        Computes the scale factor for inflating measurement noise.
        Currently uses quadratic inflation based on HDOP.
        """
        return max(1.0, hdop / 2.0) ** 2

    def assess(self, measurement: GNSSMeasurement) -> GNSSQualityResult:
        """
        Assesses the GNSS measurement and computes adaptive covariance.
        
        Args:
            measurement: The GNSS measurement to assess.
            
        Returns:
            GNSSQualityResult containing health, R_adaptive, and scale_factor.
        """
        R_base = self._compute_R_base()
        
        if measurement.fix_type == 'none' or measurement.num_satellites < 4:
            scale_factor = 1e6
            return GNSSQualityResult(
                health=GNSSHealth.DENIED,
                R_adaptive=R_base * scale_factor,
                scale_factor=scale_factor
            )
            
        scale_factor = self._compute_scale_factor(measurement.hdop, measurement.cn0_mean)
        
        if measurement.hdop >= self.hdop_threshold or measurement.cn0_mean < self.cn0_threshold:
            return GNSSQualityResult(
                health=GNSSHealth.DEGRADED,
                R_adaptive=R_base * scale_factor,
                scale_factor=scale_factor
            )
            
        return GNSSQualityResult(
            health=GNSSHealth.GOOD,
            R_adaptive=R_base,
            scale_factor=1.0
        )

    def innovation_check(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """
        Performs a chi-squared test on the innovation.
        
        Args:
            innovation: The innovation vector (measurement residual).
            S: The innovation covariance matrix.
            
        Returns:
            True if the measurement passes the check (is valid), False otherwise.
        """
        try:
            S_inv = np.linalg.inv(S)
            gamma = innovation.T @ S_inv @ innovation
            return float(gamma) < self.chi2_threshold
        except np.linalg.LinAlgError:
            return False

def geodetic_to_ned(lat: float, lon: float, alt: float, 
                    lat_ref: float, lon_ref: float, alt_ref: float) -> tuple[float, float, float]:
    """
    Converts geodetic coordinates to local NED frame using a flat-earth approximation.
    
    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude in meters.
        lat_ref: Reference latitude in degrees.
        lon_ref: Reference longitude in degrees.
        alt_ref: Reference altitude in meters.
        
    Returns:
        A tuple (n, e, d) representing North, East, Down displacements in meters.
    """
    lat_ref_rad = math.radians(lat_ref)
    dN = (lat - lat_ref) * 111320.0
    dE = (lon - lon_ref) * 111320.0 * math.cos(lat_ref_rad)
    dD = -(alt - alt_ref)
    return float(dN), float(dE), float(dD)

def ned_to_geodetic(n: float, e: float, d: float, 
                    lat_ref: float, lon_ref: float, alt_ref: float) -> tuple[float, float, float]:
    """
    Converts local NED coordinates back to geodetic using a flat-earth approximation.
    
    Args:
        n: North displacement in meters.
        e: East displacement in meters.
        d: Down displacement in meters.
        lat_ref: Reference latitude in degrees.
        lon_ref: Reference longitude in degrees.
        alt_ref: Reference altitude in meters.
        
    Returns:
        A tuple (lat, lon, alt) representing latitude, longitude, altitude.
    """
    lat_ref_rad = math.radians(lat_ref)
    lat = lat_ref + (n / 111320.0)
    lon = lon_ref + (e / (111320.0 * math.cos(lat_ref_rad)))
    alt = alt_ref - d
    return float(lat), float(lon), float(alt)
