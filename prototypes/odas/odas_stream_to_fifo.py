#!/usr/bin/env python3
"""
Stream two microphone channels from Scarlett to ODAS via a FIFO (named pipe).

ODAS will be run separately and read from this FIFO. See odas_run.sh for full setup.

Usage (Terminal 1):
    bash odas_run.sh

Usage (Terminal 2):
    python odas_stream_to_fifo.py --device INDEX --in-channels 8 --mic1 0 --mic2 1
"""

import argparse
import os
import sys
import numpy as np
import sounddevice as sd


def list_devices_and_exit():
    print(sd.query_devices())
    print("\nPick the input device index and pass with --device")
    sys.exit(0)


def main():
    p = argparse.ArgumentParser(
        description="Stream 2 channels from Scarlett to ODAS via FIFO."
    )
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", type=int, default=None, help="Input device index.")
    p.add_argument("--samplerate", type=int, default=16000, help="Sample rate (ODAS expects this).")
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--mic1", type=int, default=0)
    p.add_argument("--mic2", type=int, default=1)
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--fifo", type=str, default="mics.raw")
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

    print(f"Streaming to ODAS via FIFO: {args.fifo}")
    print(f"  Device: {args.device} ({devinfo['name']})")
    print(f"  Sample rate: {args.samplerate} Hz")
    print(f"  Mics: {args.mic1}, {args.mic2}")
    print(f"  Block size: {args.blocksize}")
    print("  Make sure ODAS is running in another terminal (bash odas_run.sh)\n")

    # Open FIFO for writing
    with open(args.fifo, "wb", buffering=0) as f:
        def cb(indata, frames, timeinfo, status):
            x = indata[:, [args.mic1, args.mic2]].astype(np.float32, copy=False)
            # Convert to 16-bit PCM
            pcm = (x * 32767.0).astype(np.int16)
            f.write(pcm.tobytes(order="C"))

        with sd.InputStream(
            device=args.device,
            channels=args.in_channels,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            dtype="float32",
            callback=cb,
        ):
            print("Streaming to ODAS. Press Ctrl+C to stop.\n")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print()


if __name__ == "__main__":
    main()
