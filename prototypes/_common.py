"""Common utilities for sound localization prototypes."""

import numpy as np
from scipy import signal
import sounddevice as sd
from typing import Tuple, Optional
import warnings


def get_audio_chunk(
    stream: sd.InputStream,
    frames: int,
    channels: int
) -> np.ndarray:
    """Read audio chunk from stream.
    
    Args:
        stream: sounddevice input stream
        frames: number of frames to read
        channels: number of channels
        
    Returns:
        numpy array shape (frames, channels)
    """
    try:
        data = stream.read(frames, allow_overflow=True)
        return data
    except Exception as e:
        print(f"Warning: audio read error {e}")
        return np.zeros((frames, channels))


def highpass_filter(
    audio: np.ndarray,
    cutoff_hz: float,
    sr: int,
    order: int = 4
) -> np.ndarray:
    """Apply highpass IIR filter.
    
    Args:
        audio: input signal
        cutoff_hz: cutoff frequency
        sr: sample rate
        order: filter order
        
    Returns:
        filtered signal
    """
    if cutoff_hz <= 0:
        return audio
    
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    if normalized_cutoff >= 1.0:
        return np.zeros_like(audio)
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    
    # Apply per-channel if multi-channel
    if audio.ndim == 2:
        filtered = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            filtered[:, ch] = signal.lfilter(b, a, audio[:, ch])
        return filtered
    else:
        return signal.lfilter(b, a, audio)


def gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sr: int,
    interp: int = 1,
    maxlag_samples: Optional[int] = None
) -> Tuple[float, float]:
    """Estimate time difference of arrival using GCC-PHAT.
    
    Args:
        sig1: first signal
        sig2: second signal
        sr: sample rate (Hz)
        interp: interpolation factor for lag estimation
        maxlag_samples: maximum lag to search
        
    Returns:
        (tdoa_seconds, confidence)
    """
    # FFT-based cross-correlation
    n = len(sig1) + len(sig2) - 1
    nfft = 2 ** np.ceil(np.log2(n)).astype(int)
    
    spec1 = np.fft.rfft(sig1, n=nfft)
    spec2 = np.fft.rfft(sig2, n=nfft)
    
    # PHAT weighting: normalize by magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted = (spec1 * np.conj(spec2)) / (np.abs(spec1 * np.conj(spec2)) + 1e-10)
    
    # Inverse FFT
    correlation = np.fft.irfft(weighted, n=nfft)[:len(sig1) + len(sig2) - 1]
    
    # Interpolate for sub-sample resolution
    if interp > 1:
        correlation_interp = signal.resample(correlation, len(correlation) * interp)
        lags = np.arange(-len(sig1) + 1, len(sig2)) / (interp * sr)
    else:
        correlation_interp = correlation
        lags = np.arange(-len(sig1) + 1, len(sig2)) / sr
    
    # Find maximum
    if maxlag_samples is not None:
        center = len(correlation_interp) // 2
        max_idx_offset = maxlag_samples * interp
        search_start = max(0, center - max_idx_offset)
        search_end = min(len(correlation_interp), center + max_idx_offset)
        max_idx = search_start + np.argmax(correlation_interp[search_start:search_end])
    else:
        max_idx = np.argmax(correlation_interp)
    
    # Confidence: peak ratio
    sorted_corr = np.sort(correlation_interp)[::-1]
    confidence = sorted_corr[0] - sorted_corr[1] if len(sorted_corr) > 1 else sorted_corr[0]
    
    tdoa = lags[max_idx]
    
    return tdoa, confidence


def tdoa_to_position(
    tdoa: float,
    mic_distance: float,
    sr: int,
    c: float = 343.0
) -> Tuple[float, str]:
    """Convert TDOA to 1D position.
    
    Assumes mics at 0m and mic_distance m on x-axis.
    Positive TDOA = sound reached mic2 first (source closer to mic2).
    
    Args:
        tdoa: time difference of arrival (seconds, >0 = mic2 first)
        mic_distance: distance between mics (meters)
        sr: sample rate (Hz)
        c: speed of sound (m/s, default 343 at 20°C)
        
    Returns:
        (position_m, solution_type)
        position_m: distance from mic1 (0 to mic_distance)
        solution_type: "normal" or "extrapolated"
    """
    # Distance difference
    dist_diff = tdoa * c
    
    # Solve: d1 - d2 = dist_diff, d1 + d2 = mic_distance
    # Solution: d1 = (mic_distance + dist_diff) / 2
    d1 = (mic_distance + dist_diff) / 2
    
    # Clamp to [0, mic_distance]
    if d1 < 0:
        return 0.0, "extrapolated_before"
    elif d1 > mic_distance:
        return mic_distance, "extrapolated_after"
    else:
        return d1, "normal"


def rms(signal_data: np.ndarray, axis=None) -> float:
    """RMS of signal."""
    return np.sqrt(np.mean(signal_data ** 2, axis=axis))


def db(x: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """Convert to dB."""
    return 20 * np.log10(np.abs(x) / ref + 1e-10)


class ExponentialMovingAverage:
    """EMA smoother for tracking."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


def extract_multichannel_chunk(
    data: np.ndarray,
    channels: list
) -> np.ndarray:
    """Extract specific channels from multichannel data.
    
    Args:
        data: shape (frames, n_channels)
        channels: list of channel indices
        
    Returns:
        shape (frames, len(channels))
    """
    if data.ndim == 1:
        return data[:, np.newaxis]
    return data[:, channels]
