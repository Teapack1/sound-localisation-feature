#!/usr/bin/env python3
"""
Common utilities for all sound localization prototypes.
Includes signal processing, geometry, and logging helpers.
"""

import numpy as np
from scipy.signal import butter, sosfilt
import math
from dataclasses import dataclass
from typing import Tuple, Optional
import time

# ============================================================================
# Constants
# ============================================================================

SPEED_OF_SOUND = 343.0  # m/s at 20°C in dry air


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MicGeometry:
    """Microphone positions in 3D space (meters)."""
    positions: np.ndarray  # shape: (n_mics, 3), rows are [x, y, z]
    names: list = None      # Mic identifiers
    
    def __post_init__(self):
        if self.names is None:
            self.names = [f"Mic{i}" for i in range(len(self.positions))]
    
    @property
    def n_mics(self):
        return len(self.positions)
    
    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between mics i and j."""
        return np.linalg.norm(self.positions[i] - self.positions[j])
    
    def pairwise_distances(self):
        """All pairwise distances as dict {(i,j): distance}."""
        distances = {}
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                distances[(i, j)] = self.distance(i, j)
        return distances


@dataclass
class LocalizationResult:
    """Output of a localization estimate."""
    timestamp: float                    # Unix time
    method: str                         # Name of algorithm
    position_xyz: Optional[np.ndarray] = None  # [x, y, z] meters (or None for DOA-only)
    azimuth_deg: Optional[float] = None        # Degrees, 0=reference mic direction
    colatitude_deg: Optional[float] = None     # Degrees, 90=horizontal plane
    confidence: float = 0.0             # 0.0-1.0, higher is better
    mics_used: list = None              # Which mics contributed
    raw_data: dict = None               # Algorithm-specific debug info
    
    def __str__(self):
        parts = [f"[{self.method}]"]
        if self.position_xyz is not None:
            parts.append(f"pos=({self.position_xyz[0]:.2f}, {self.position_xyz[1]:.2f}, {self.position_xyz[2]:.2f})m")
        if self.azimuth_deg is not None:
            parts.append(f"az={self.azimuth_deg:.1f}°")
        if self.colatitude_deg is not None:
            parts.append(f"colat={self.colatitude_deg:.1f}°")
        parts.append(f"conf={self.confidence:.2f}")
        return " ".join(parts)


# ============================================================================
# Signal Processing
# ============================================================================

def design_highpass(fs: int, cutoff_hz: float, order: int = 4) -> Optional[np.ndarray]:
    """Design a high-pass Butterworth filter.
    
    Args:
        fs: Sample rate in Hz
        cutoff_hz: Cutoff frequency in Hz (0 to disable)
        order: Filter order
    
    Returns:
        Second-order sections (sos) array for scipy.signal.sosfilt, or None if cutoff_hz <= 0
    """
    if cutoff_hz <= 0:
        return None
    sos = butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos')
    return sos


def apply_highpass(signal: np.ndarray, sos: Optional[np.ndarray]) -> np.ndarray:
    """Apply high-pass filter if sos is not None."""
    if sos is None:
        return signal
    return sosfilt(sos, signal)


def normalize_signal(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    signal = signal - np.mean(signal)
    energy = np.sqrt(np.mean(signal ** 2)) + eps
    return signal / energy


def extract_window(buffer: np.ndarray, center_idx: int, window_len: int) -> np.ndarray:
    """Extract a centered window from a buffer (handles boundaries).
    
    Args:
        buffer: 1D or 2D array (time, [channels])
        center_idx: Center index in time axis
        window_len: Window length
    
    Returns:
        Window array, zero-padded if near boundaries
    """
    half_win = window_len // 2
    start = center_idx - half_win
    end = center_idx + half_win
    
    if buffer.ndim == 1:
        result = np.zeros(window_len)
        valid_start = max(0, start)
        valid_end = min(len(buffer), end)
        result[valid_start - start : valid_end - start] = buffer[valid_start:valid_end]
    else:
        result = np.zeros((window_len, buffer.shape[1]))
        valid_start = max(0, start)
        valid_end = min(buffer.shape[0], end)
        result[valid_start - start : valid_end - start, :] = buffer[valid_start:valid_end, :]
    
    return result


# ============================================================================
# GCC-PHAT TDOA Estimation (Shared Across Prototypes)
# ============================================================================

def gcc_phat_tdoa(
    x_ref: np.ndarray,
    x_sig: np.ndarray,
    fs: int,
    max_tau: float = None,
    interp: int = 1
) -> Tuple[float, int, float]:
    """
    Estimate TDOA between two signals using GCC-PHAT.
    
    Args:
        x_ref: Reference signal (1D array)
        x_sig: Signal to compare (1D array)
        fs: Sample rate (Hz)
        max_tau: Maximum physically plausible delay (seconds)
        interp: Interpolation factor for finer delay estimates
    
    Returns:
        (tau_s, shift_samp, confidence):
            tau_s: Delay in seconds (positive = x_sig delayed relative to x_ref)
            shift_samp: Delay in samples (original fs)
            confidence: Peak-to-mean ratio (higher = better)
    """
    x_ref = np.asarray(x_ref, dtype=np.float32)
    x_sig = np.asarray(x_sig, dtype=np.float32)
    
    n_fft = 1 << (len(x_ref) + len(x_sig) - 1).bit_length()
    
    # Compute FFT
    X1 = np.fft.rfft(x_ref, n=n_fft)
    X2 = np.fft.rfft(x_sig, n=n_fft)
    
    # Cross-spectrum
    R = X1 * np.conj(X2)
    
    # PHAT weighting: normalize by magnitude
    denom = np.abs(R) + 1e-12
    R_phat = R / denom
    
    # Generalized cross-correlation
    cc = np.fft.irfft(R_phat, n=n_fft)
    
    # Search range
    if max_tau is None:
        max_tau = max(len(x_ref), len(x_sig)) / fs
    max_shift = int(round(max_tau * fs * interp))
    max_shift = max(1, max_shift)
    
    # Circular shift to align lags
    cc_circular = np.concatenate([cc[-max_shift:], cc[:max_shift + 1]])
    
    # Peak detection
    peak_idx = np.argmax(np.abs(cc_circular))
    peak_value = np.abs(cc_circular[peak_idx])
    mean_abs = np.mean(np.abs(cc_circular)) + 1e-12
    confidence = peak_value / mean_abs
    
    # Convert back to sample shift
    shift_interp = peak_idx - max_shift
    tau_s = shift_interp / (fs * interp)
    shift_samp = int(round(tau_s * fs))
    
    return tau_s, shift_samp, confidence


# ============================================================================
# Geometry & Multilateration
# ============================================================================

def tdoa_to_azimuth(
    tdoa: float,
    mic_distance: float,
    speed_of_sound: float = SPEED_OF_SOUND
) -> float:
    """
    Convert TDOA to azimuth angle for linear 2-mic array.
    
    Args:
        tdoa: Time difference of arrival (seconds, positive = sig delayed)
        mic_distance: Distance between microphones (meters)
        speed_of_sound: Speed of sound (m/s)
    
    Returns:
        Azimuth in degrees (-90 to +90, where 0 = perpendicular to mic axis)
    """
    c_tau = speed_of_sound * tdoa
    val = np.clip(c_tau / mic_distance, -1.0, 1.0)
    angle_rad = np.arcsin(val)
    return float(np.degrees(angle_rad))


def tdoa_to_position_1d(
    tdoa: float,
    mic_distance: float,
    speed_of_sound: float = SPEED_OF_SOUND
) -> float:
    """
    Convert TDOA to 1D position along linear 2-mic array.
    
    Assumes mic1 at x=0, mic2 at x=mic_distance. Source on the line.
    
    Args:
        tdoa: Time difference (seconds, positive = mic2 delayed)
        mic_distance: Distance between mics (meters)
        speed_of_sound: Speed of sound (m/s)
    
    Returns:
        Position along mic axis (meters, 0 to mic_distance)
    """
    # x = (d + c*tau) / 2
    x = (mic_distance + speed_of_sound * tdoa) / 2.0
    x = np.clip(x, 0.0, mic_distance)
    return float(x)


def multilaterate_2d(
    mic_geom: MicGeometry,
    tdoa_dict: dict,
    speed_of_sound: float = SPEED_OF_SOUND,
    weights: dict = None
) -> Tuple[np.ndarray, float]:
    """
    Solve 2D position from pairwise TDOAs using least-squares.
    
    Args:
        mic_geom: MicGeometry object
        tdoa_dict: {(i, j): delay_in_seconds} measured delays
        speed_of_sound: Speed of sound (m/s)
        weights: {(i, j): weight} confidence per pair (default: all 1.0)
    
    Returns:
        (position_xy, residual_error)
    """
    if weights is None:
        weights = {k: 1.0 for k in tdoa_dict.keys()}
    
    # Build least-squares problem
    A_list = []
    b_list = []
    w_list = []
    
    for (i, j), tau in tdoa_dict.items():
        w = weights.get((i, j), 1.0)
        w_list.append(w)
        
        # Hyperbolic constraint: distance_i - distance_j = c * tau
        # Linearized near an initial guess
        pos_i = mic_geom.positions[i, :2]
        pos_j = mic_geom.positions[j, :2]
        
        mid = (pos_i + pos_j) / 2.0
        direction = (pos_j - pos_i) / np.linalg.norm(pos_j - pos_i)
        
        A_list.append(direction)
        b_list.append(speed_of_sound * tau)
    
    if len(A_list) == 0:
        return np.array([0.0, 0.0]), float('inf')
    
    A = np.array(A_list)
    b = np.array(b_list)
    w = np.array(w_list)
    
    # Weighted least-squares
    W = np.diag(w)
    try:
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b
        delta = np.linalg.solve(AtWA, AtWb)
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0]), float('inf')
    
    # Use first mic as origin
    pos_xy = mic_geom.positions[0, :2] + delta
    
    # Residual
    residuals = A @ delta - b
    residual_rms = np.sqrt(np.mean(residuals ** 2))
    
    return pos_xy, residual_rms


# ============================================================================
# Logging & Formatting
# ============================================================================

class RateLimiter:
    """Simple rate limiting for terminal output."""
    
    def __init__(self, max_hz: float = 10.0):
        self.min_dt = 1.0 / max(max_hz, 1e-6)
        self.last_time = 0.0
    
    def should_print(self) -> bool:
        now = time.time()
        if now - self.last_time >= self.min_dt:
            self.last_time = now
            return True
        return False


def format_result(result: LocalizationResult) -> str:
    """Pretty-print a localization result."""
    return str(result)
