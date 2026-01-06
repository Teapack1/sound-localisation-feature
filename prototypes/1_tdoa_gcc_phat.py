#!/usr/bin/env python3
"""
Prototype 1: Real-time 2-mic sound localization using GCC-PHAT TDOA.

Algorithm: Generalized Cross-Correlation with Phase Transform
- Estimates time difference of arrival (TDOA) between two microphones
- Converts to azimuth angle and 1D position along mic axis
- Lowest CPU cost, most transparent failure modes

Usage:
    python 1_tdoa_gcc_phat.py --list-devices
    python 1_tdoa_gcc_phat.py --device 3 --in-channels 8 --mic1 0 --mic2 1 --mic-distance 4.0
"""

import argparse
import queue
import sys
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

from _common import (
    SPEED_OF_SOUND,
    gcc_phat_tdoa,
    design_highpass,
    normalize_signal,
    RateLimiter,
    LocalizationResult,
    tdoa_to_azimuth,
    tdoa_to_position_1d,
)


@dataclass
class Config:
    """Configuration for 2-mic TDOA localization."""
    device: int
    samplerate: int
    in_channels: int
    mic1: int
    mic2: int
    mic_distance_m: float
    blocksize: int
    window_ms: float
    hop_ms: float
    highpass_hz: float
    confidence_threshold: float
    print_hz: float
    interp: int


def list_devices_and_exit():
    """Print available audio devices and exit."""
    print(sd.query_devices())
    print("\nPick the input device index and pass with --device")
    sys.exit(0)


def main():
    p = argparse.ArgumentParser(
        description="Real-time 2-microphone sound localization via GCC-PHAT TDOA."
    )
    p.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit."
    )
    p.add_argument(
        "--device", type=int, default=None, help="Input device index."
    )
    p.add_argument(
        "--samplerate", type=int, default=48000, help="Sample rate in Hz."
    )
    p.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Number of input channels to open (e.g., 8 for Scarlett).",
    )
    p.add_argument(
        "--mic1", type=int, default=0, help="Index of microphone 1 (0-based)."
    )
    p.add_argument(
        "--mic2", type=int, default=1, help="Index of microphone 2 (0-based)."
    )
    p.add_argument(
        "--mic-distance", type=float, default=4.0, help="Distance between mics in meters."
    )
    p.add_argument(
        "--blocksize", type=int, default=1024, help="Audio callback block size in frames."
    )
    p.add_argument(
        "--window-ms", type=float, default=80.0, help="Analysis window length in ms."
    )
    p.add_argument(
        "--hop-ms", type=float, default=20.0, help="Hop interval in ms."
    )
    p.add_argument(
        "--highpass-hz", type=float, default=120.0, help="High-pass cutoff (0 to disable)."
    )
    p.add_argument(
        "--conf-threshold", type=float, default=3.0, help="Confidence threshold."
    )
    p.add_argument(
        "--print-hz", type=float, default=10.0, help="Max output rate (Hz)."
    )
    p.add_argument(
        "--interp", type=int, default=4, help="GCC-PHAT interpolation factor."
    )
    args = p.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    # Resolve device
    if args.device is None:
        args.device = (
            sd.default.device[0]
            if isinstance(sd.default.device, (list, tuple))
            else sd.default.device
        )
        if args.device is None:
            raise SystemExit(
                "No default input device. Use --list-devices and pass --device."
            )

    # Check channel indices
    devinfo = sd.query_devices(args.device, kind="input")
    max_in = int(devinfo["max_input_channels"])

    if args.in_channels is None:
        args.in_channels = max(args.mic1, args.mic2) + 1

    if args.in_channels > max_in:
        raise SystemExit(
            f"Device only has {max_in} input channels, but --in-channels {args.in_channels} requested."
        )
    if not (0 <= args.mic1 < args.in_channels and 0 <= args.mic2 < args.in_channels):
        raise SystemExit(
            f"Mic indices must be within [0, {args.in_channels - 1}]."
        )

    cfg = Config(
        device=args.device,
        samplerate=args.samplerate,
        in_channels=args.in_channels,
        mic1=args.mic1,
        mic2=args.mic2,
        mic_distance_m=args.mic_distance,
        blocksize=args.blocksize,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        highpass_hz=args.highpass_hz,
        confidence_threshold=args.conf_threshold,
        print_hz=args.print_hz,
        interp=args.interp,
    )

    print(f"2-mic TDOA localizer")
    print(f"  Device: {args.device} ({devinfo['name']})")
    print(f"  Sample rate: {cfg.samplerate} Hz")
    print(f"  Opening {cfg.in_channels} input channels using mic1={cfg.mic1}, mic2={cfg.mic2}")
    print(f"  Mic distance: {cfg.mic_distance_m:.3f} m")
    print(
        f"  Analysis window: {cfg.window_ms:.1f} ms, hop: {cfg.hop_ms:.1f} ms, blocksize: {cfg.blocksize}"
    )
    print(f"  High-pass: {cfg.highpass_hz:.1f} Hz, GCC interp: {cfg.interp}x")
    print(f"  Confidence threshold: {cfg.confidence_threshold:.2f}")
    print("Press Ctrl+C to stop.\n")

    q = queue.Queue(maxsize=200)

    # DSP setup
    hp_sos = design_highpass(cfg.samplerate, cfg.highpass_hz, order=4)

    window_len = int(round(cfg.window_ms * 1e-3 * cfg.samplerate))
    hop_len = int(round(cfg.hop_ms * 1e-3 * cfg.samplerate))
    hop_len = max(1, hop_len)
    window_len = max(window_len, cfg.blocksize)

    max_tau = cfg.mic_distance_m / SPEED_OF_SOUND

    # Buffers
    buf1 = np.zeros(window_len, dtype=np.float32)
    buf2 = np.zeros(window_len, dtype=np.float32)
    pending = np.zeros((0, 2), dtype=np.float32)

    rate_limiter = RateLimiter(cfg.print_hz)

    def callback(indata, frames, timeinfo, status):
        if status:
            pass  # Could log status
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    with sd.InputStream(
        device=cfg.device,
        channels=cfg.in_channels,
        samplerate=cfg.samplerate,
        blocksize=cfg.blocksize,
        dtype="float32",
        callback=callback,
    ):
        print("Streaming...\n")
        try:
            while True:
                block = q.get()
                if block.ndim != 2 or block.shape[1] != cfg.in_channels:
                    continue

                # Extract the two channels of interest
                x1 = block[:, cfg.mic1].astype(np.float32, copy=False)
                x2 = block[:, cfg.mic2].astype(np.float32, copy=False)

                # Accumulate to hop length
                pair = np.stack([x1, x2], axis=1)
                pending = np.concatenate([pending, pair], axis=0)

                # Process hop
                while pending.shape[0] >= hop_len:
                    chunk = pending[:hop_len]
                    pending = pending[hop_len:]

                    # Slide window buffers
                    n = chunk.shape[0]
                    buf1 = np.roll(buf1, -n)
                    buf2 = np.roll(buf2, -n)
                    buf1[-n:] = chunk[:, 0]
                    buf2[-n:] = chunk[:, 1]

                    # Process window
                    x1w = buf1.copy()
                    x2w = buf2.copy()

                    # Optional high-pass
                    if hp_sos is not None:
                        x1w = sosfilt(hp_sos, x1w)
                        x2w = sosfilt(hp_sos, x2w)

                    # Normalize
                    x1w = normalize_signal(x1w)
                    x2w = normalize_signal(x2w)

                    # GCC-PHAT TDOA
                    tau_s, shift_samp, conf = gcc_phat_tdoa(
                        x1w, x2w, fs=cfg.samplerate, max_tau=max_tau, interp=cfg.interp
                    )

                    if conf < cfg.confidence_threshold:
                        continue

                    # Convert to azimuth and 1D position
                    angle_deg = tdoa_to_azimuth(tau_s, cfg.mic_distance_m)
                    x_m = tdoa_to_position_1d(tau_s, cfg.mic_distance_m)

                    # Direction hint
                    direction = "toward_mic1" if tau_s > 0 else "toward_mic2" if tau_s < 0 else "center"

                    if rate_limiter.should_print():
                        print(
                            f"[TDOA] tdoa={tau_s*1e3:7.3f}ms | shift={shift_samp:5d}samp "
                            f"| angle={angle_deg:6.1f}° | x={x_m:5.2f}m | "
                            f"conf={conf:5.2f} | {direction}"
                        )

        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
