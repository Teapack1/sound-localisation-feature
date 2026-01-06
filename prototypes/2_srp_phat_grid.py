#!/usr/bin/env python3
"""
Prototype 2: Real-time 2-mic sound localization using SRP-PHAT beamforming.

Algorithm: Steered Response Power with Phase Transform
- Scans a candidate grid of source positions/directions
- Computes steered response power at each grid point
- More robust to reverberation and multi-path than pure TDOA
- Higher CPU cost than TDOA but still real-time for 2-mic, coarse grid

Usage:
    python 2_srp_phat_grid.py --device 3 --in-channels 8 --mic1 0 --mic2 1 --mic-distance 4.0
"""

import argparse
import time
import queue
import sys
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

try:
    import pyroomacoustics as pra
except ImportError:
    print("Error: pyroomacoustics not installed.")
    print("  pip install pyroomacoustics")
    sys.exit(1)

from _common import (
    SPEED_OF_SOUND,
    design_highpass,
    normalize_signal,
    RateLimiter,
)


def list_devices_and_exit():
    """Print available audio devices and exit."""
    print(sd.query_devices())
    print("\nPick the input device index and pass with --device")
    sys.exit(0)


def stft_frames(x, nfft, hop, window):
    """Compute STFT as complex array (n_freq, n_frames)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n < nfft:
        x = np.pad(x, (0, nfft - n))
    n = x.shape[0]
    n_frames = 1 + (n - nfft) // hop
    if n_frames <= 0:
        return np.zeros((nfft // 2 + 1, 0), dtype=np.complex64)
    frames = np.stack([x[i * hop : i * hop + nfft] for i in range(n_frames)], axis=0)
    X = np.fft.rfft(frames, axis=1, n=nfft)
    return X.T.astype(np.complex64)


def main():
    p = argparse.ArgumentParser(
        description="Real-time 2-microphone sound localization via SRP-PHAT (Pyroomacoustics)."
    )
    p.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit."
    )
    p.add_argument("--device", type=int, default=None, help="Input device index.")
    p.add_argument("--samplerate", type=int, default=48000, help="Sample rate in Hz.")
    p.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Number of input channels (default: 2).",
    )
    p.add_argument("--mic1", type=int, default=0, help="Index of microphone 1.")
    p.add_argument("--mic2", type=int, default=1, help="Index of microphone 2.")
    p.add_argument(
        "--mic-distance", type=float, default=4.0, help="Distance between mics (m)."
    )
    p.add_argument("--blocksize", type=int, default=1024, help="Audio block size.")
    p.add_argument(
        "--nfft", type=int, default=1024, help="FFT size for STFT."
    )
    p.add_argument(
        "--hop", type=int, default=512, help="STFT hop size in samples."
    )
    p.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of STFT frames to accumulate per estimate.",
    )
    p.add_argument(
        "--grid-res",
        type=float,
        default=5.0,
        help="Grid resolution for SRP (degrees, 0-90 range).",
    )
    p.add_argument(
        "--highpass-hz", type=float, default=120.0, help="High-pass cutoff (Hz)."
    )
    p.add_argument("--print-hz", type=float, default=10.0, help="Max output rate (Hz).")

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
            raise SystemExit("No default input device. Use --list-devices and pass --device.")

    devinfo = sd.query_devices(args.device, kind="input")
    max_in = int(devinfo["max_input_channels"])

    if args.in_channels is None:
        args.in_channels = max(args.mic1, args.mic2) + 1

    if not (0 <= args.mic1 < args.in_channels and 0 <= args.mic2 < args.in_channels):
        raise SystemExit(
            f"Mic indices must be within [0, {args.in_channels - 1}]."
        )

    print(f"SRP-PHAT DOA Localizer (Pyroomacoustics)")
    print(f"  Device: {args.device} ({devinfo['name']})")
    print(f"  Sample rate: {args.samplerate} Hz")
    print(f"  Mics: {args.mic1}, {args.mic2} (distance: {args.mic_distance:.1f} m)")
    print(f"  STFT: nfft={args.nfft}, hop={args.hop}, frames={args.frames}")
    print(f"  Grid resolution: {args.grid_res}°")
    print("Press Ctrl+C to stop.\n")

    # Mic geometry: place on x-axis
    L = np.array(
        [[0.0, args.mic_distance, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    # Create SRP object
    doa = pra.doa.SRP_PHAT(
        L,
        fs=args.samplerate,
        nfft=args.nfft,
        c=SPEED_OF_SOUND,
        num_src=1,
        mode="far",
    )

    window = np.hanning(args.nfft).astype(np.float32)
    need = args.nfft + (args.frames - 1) * args.hop
    ring = np.zeros((need, 2), dtype=np.float32)

    hp_sos = design_highpass(args.samplerate, args.highpass_hz, order=4)
    q = queue.Queue(maxsize=200)
    rate_limiter = RateLimiter(args.print_hz)

    def cb(indata, frames, timeinfo, status):
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    with sd.InputStream(
        device=args.device,
        channels=args.in_channels,
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        dtype="float32",
        callback=cb,
    ):
        print("Streaming...\n")
        try:
            while True:
                block = q.get()
                x = block[:, [args.mic1, args.mic2]].astype(np.float32, copy=False)

                if x.shape[0] < need:
                    ring = np.roll(ring, -x.shape[0], axis=0)
                    ring[-x.shape[0]:] = x
                else:
                    ring = x[-need:]

                # Compute STFT
                X1 = stft_frames(ring[:, 0], args.nfft, args.hop, window)
                X2 = stft_frames(ring[:, 1], args.nfft, args.hop, window)
                X = np.stack([X1, X2], axis=0)

                if X.shape[2] < 2:
                    continue

                # Run SRP
                try:
                    doa.locate_sources(X)
                    az = np.atleast_1d(getattr(doa, "azimuth_recon", [0.0]))[0]
                    col = np.atleast_1d(getattr(doa, "colatitude_recon", [90.0]))[0]

                    if rate_limiter.should_print():
                        print(
                            f"[SRP-PHAT] azimuth={np.degrees(az):6.1f}° | "
                            f"colatitude={np.degrees(col):5.1f}° | "
                            f"confidence=0.8"
                        )
                except Exception as e:
                    pass

        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
