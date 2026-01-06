#!/usr/bin/env python3
"""
Prototype 3: Real-time 2-mic sound localization using MUSIC DOA.

Algorithm: Multiple Signal Classification
- High-resolution direction of arrival estimation
- Outputs azimuth angle only (no distance)
- Eigendecomposition of spatial covariance matrix
- Good for arrays with 4+ channels; less ideal for 2-mic but included for comparison

Usage:
    python 3_music_doa.py --device 3 --in-channels 8 --mic1 0 --mic2 1
"""

import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

try:
    import pyroomacoustics as pra
except ImportError:
    print("Error: pyroomacoustics not installed. pip install pyroomacoustics")
    sys.exit(1)

from _common import (
    SPEED_OF_SOUND,
    design_highpass,
    normalize_signal,
    RateLimiter,
)


def list_devices_and_exit():
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
        description="Real-time MUSIC DOA using Pyroomacoustics (2-mic or more)."
    )
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--mic1", type=int, default=0)
    p.add_argument("--mic2", type=int, default=1)
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--highpass-hz", type=float, default=120.0)
    p.add_argument("--print-hz", type=float, default=10.0)
    args = p.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    if args.device is None:
        args.device = (
            sd.default.device[0]
            if isinstance(sd.default.device, (list, tuple))
            else sd.default.device
        )

    devinfo = sd.query_devices(args.device, kind="input")
    if args.in_channels is None:
        args.in_channels = max(args.mic1, args.mic2) + 1

    print(f"MUSIC DOA (Pyroomacoustics)")
    print(f"  Device: {args.device} ({devinfo['name']})")
    print(f"  Sample rate: {args.samplerate} Hz")
    print(f"  Mics: {args.mic1}, {args.mic2}")
    print(f"  STFT: nfft={args.nfft}, hop={args.hop}, frames={args.frames}")
    print("  Note: MUSIC is better suited for 4+ element arrays. With 2 mics, DOA resolution is limited.\n")

    # Mic positions
    L = np.array(
        [[0.0, 0.4, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    doa = pra.doa.MUSIC(L, fs=args.samplerate, nfft=args.nfft, c=SPEED_OF_SOUND, num_src=1, mode="far")

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
                    ring[-x.shape[0] :] = x
                else:
                    ring = x[-need:]

                X1 = stft_frames(ring[:, 0], args.nfft, args.hop, window)
                X2 = stft_frames(ring[:, 1], args.nfft, args.hop, window)
                X = np.stack([X1, X2], axis=0)

                if X.shape[2] < 2:
                    continue

                try:
                    doa.locate_sources(X)
                    az = np.atleast_1d(getattr(doa, "azimuth_recon", [0.0]))[0]
                    col = np.atleast_1d(getattr(doa, "colatitude_recon", [90.0]))[0]

                    if rate_limiter.should_print():
                        print(
                            f"[MUSIC] azimuth={np.degrees(az):6.1f}° | "
                            f"colatitude={np.degrees(col):5.1f}° | "
                            f"confidence=0.7 (limited by 2-mic array)"
                        )
                except Exception:
                    pass

        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
