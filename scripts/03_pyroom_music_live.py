#!/usr/bin/env python3
"""Real-time DOA using pyroomacoustics MUSIC (far-field).

Useful comparison vs SRP-PHAT and GCC-PHAT in your exact room.
"""

import argparse
import math
import sys
import time

import numpy as np
import sounddevice as sd
import pyroomacoustics as pra

SPEED_OF_SOUND = 343.0


def stft_frames(x: np.ndarray, nfft: int, hop: int, window: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] < nfft:
        x = np.pad(x, (0, nfft - x.shape[0]))
    n = x.shape[0]
    n_frames = 1 + (n - nfft) // hop
    if n_frames <= 0:
        return np.zeros((nfft // 2 + 1, 0), dtype=np.complex64)

    frames = np.stack([x[i * hop : i * hop + nfft] * window for i in range(n_frames)], axis=0)
    X = np.fft.rfft(frames, n=nfft, axis=1)
    return X.T.astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--mic1", type=int, default=0)
    ap.add_argument("--mic2", type=int, default=1)
    ap.add_argument("--mic-distance", type=float, default=4.0)
    ap.add_argument("--blocksize", type=int, default=0)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--print-hz", type=float, default=10.0)
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    fs = int(args.samplerate)
    d = float(args.mic_distance)

    L = np.array(
        [
            [0.0, d],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    doa = pra.doa.MUSIC(L, fs=fs, nfft=int(args.nfft), c=SPEED_OF_SOUND, num_src=1, mode="far")

    window = np.hanning(int(args.nfft)).astype(np.float32)
    need = int(args.nfft) + (int(args.frames) - 1) * int(args.hop)
    ring = np.zeros((need, 2), dtype=np.float32)

    min_dt = 1.0 / max(args.print_hz, 1e-6)
    last_print = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal ring
        if status:
            print(status, file=sys.stderr)
        x = indata[:, [args.mic1, args.mic2]].astype(np.float32, copy=False)
        if x.shape[0] >= need:
            ring[:] = x[-need:]
        else:
            ring = np.roll(ring, -x.shape[0], axis=0)
            ring[-x.shape[0] :] = x

    with sd.InputStream(
        device=args.device,
        channels=args.in_channels,
        samplerate=fs,
        dtype="float32",
        blocksize=args.blocksize,
        callback=callback,
    ):
        print("Running pyroomacoustics MUSIC live. Ctrl+C to stop.")
        print(f"device={args.device} fs={fs} ch={args.in_channels} using mic1={args.mic1} mic2={args.mic2} d={d}m")
        try:
            while True:
                time.sleep(int(args.hop) / fs)

                X1 = stft_frames(ring[:, 0], nfft=int(args.nfft), hop=int(args.hop), window=window)
                X2 = stft_frames(ring[:, 1], nfft=int(args.nfft), hop=int(args.hop), window=window)
                X = np.stack([X1, X2], axis=0)

                if X.shape[-1] < 2:
                    continue

                if hasattr(doa, "locate_sources"):
                    doa.locate_sources(X)
                else:
                    doa.locate_source(X)

                az = float(np.atleast_1d(getattr(doa, "azimuth_recon"))[0])

                now = time.time()
                if now - last_print >= min_dt:
                    last_print = now
                    print(f"t={now:.3f}  azimuth={math.degrees(az):6.1f} deg")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
