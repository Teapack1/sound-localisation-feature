#!/usr/bin/env python3
"""
Prototype 4: Real-time 2-mic sound localization using OpenSoundscape TDOA.

Algorithm: TDOA with least-squares multilateration (OpenSoundscape framework)
- Uses OpenSoundscape's SynchronizedRecorderArray and localization API
- Estimates TDOA cross-correlation between channels
- Good for integrated audio lab workflows, but framework overhead adds latency
- PyTorch-based; works well if you're already in that ecosystem

Usage:
    python 4_opensoundscape_tdoa.py --device 3 --in-channels 8 --mic1 0 --mic2 1
"""

import argparse
import queue
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf

try:
    from opensoundscape.localization import SynchronizedRecorderArray
except ImportError:
    print("Error: opensoundscape not installed.")
    print("  pip install opensoundscape[audio]")
    sys.exit(1)

from _common import (
    SPEED_OF_SOUND,
    design_highpass,
    RateLimiter,
)


def list_devices_and_exit():
    print(sd.query_devices())
    print("\nPick the input device index and pass with --device")
    sys.exit(0)


def main():
    p = argparse.ArgumentParser(
        description="Real-time OpenSoundscape TDOA logger (2-mic adapter)."
    )
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--mic1", type=int, default=0)
    p.add_argument("--mic2", type=int, default=1)
    p.add_argument("--mic-distance", type=float, default=4.0)
    p.add_argument("--window-s", type=float, default=1.0, help="Window duration (s)")
    p.add_argument("--hop-s", type=float, default=0.5, help="Hop interval (s)")
    p.add_argument("--cc-threshold", type=float, default=0.01)
    p.add_argument("--cc-filter", type=str, default="phat")
    p.add_argument("--localization-algorithm", type=str, default="leastsquares")
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--print-hz", type=float, default=5.0)
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

    print(f"OpenSoundscape TDOA Logger (2-mic adapter)")
    print(f"  Device: {args.device} ({devinfo['name']})")
    print(f"  Sample rate: {args.samplerate} Hz")
    print(f"  Mics: {args.mic1}, {args.mic2} (distance: {args.mic_distance:.1f} m)")
    print(f"  Window: {args.window_s:.1f}s, Hop: {args.hop_s:.1f}s")
    print(f"  CC threshold: {args.cc_threshold}, filter: {args.cc_filter}")
    print("Press Ctrl+C to stop.\n")

    win = int(round(args.window_s * args.samplerate))
    hop = int(round(args.hop_s * args.samplerate))
    buf = np.zeros((win, 2), dtype=np.float32)
    filled = 0

    rec1_path = "receiver1.wav"
    rec2_path = "receiver2.wav"

    # Receiver array (OpenSoundscape format)
    ar_coords = pd.DataFrame(
        {"x": [0.0, args.mic_distance], "y": [0.0, 0.0]},
        index=[rec1_path, rec2_path],
    )
    array = SynchronizedRecorderArray(ar_coords)

    class_name = "anysound"
    start_ts = datetime.now()

    q = queue.Queue(maxsize=200)
    rate_limiter = RateLimiter(args.print_hz)

    def cb(indata, frames, timeinfo, status):
        nonlocal buf, filled
        x = indata[:, [args.mic1, args.mic2]].astype(np.float32, copy=False)
        if x.shape[0] >= win - filled:
            needed = win - filled
            buf[filled:] = x[:needed]
            filled = win
        else:
            buf[filled : filled + x.shape[0]] = x
            filled += x.shape[0]

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
                time.sleep(args.hop_s)
                if filled < win:
                    continue

                # Write WAVs for each receiver
                sf.write(rec1_path, buf[:, 0], args.samplerate)
                sf.write(rec2_path, buf[:, 1], args.samplerate)

                # Build detections table (OpenSoundscape format)
                end_time = args.window_s
                det = pd.DataFrame(
                    {
                        "file": [rec1_path, rec2_path],
                        "starttime": [0.0, 0.0],
                        "endtime": [end_time, end_time],
                        "class": [class_name, class_name],
                    }
                )
                det["starttimestamp"] = start_ts
                det = det.set_index(["file", "starttime", "endtime", "starttimestamp"])

                # Localize
                try:
                    estimates = array.localize_detections(
                        det,
                        min_n_receivers=2,
                        max_receiver_dist=1000,
                        localization_algorithm=args.localization_algorithm,
                        cc_threshold=args.cc_threshold,
                        cc_filter=args.cc_filter,
                        residual_threshold=None,
                    )

                    if len(estimates) > 0:
                        e = estimates[0]
                        if rate_limiter.should_print():
                            print(
                                f"[OpenSoundscape] TDOA={np.mean(e.tdoas):8.5f}s | "
                                f"CC_max={np.mean(e.ccmaxs):6.3f} | "
                                f"residual_rms={getattr(e, 'residual_rms', 0.0):6.3f}m"
                            )
                except Exception as ex:
                    pass

        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
