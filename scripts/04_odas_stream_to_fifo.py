#!/usr/bin/env python3
"""Stream 2 selected channels from a multichannel audio device into a FIFO as S16LE.

This is meant to feed ODAS (https://github.com/introlab/odas).
ODAS reads from the FIFO and outputs localisation (often as JSON) to terminal / socket.

Usage (example):
  rm -f mics.raw && mkfifo mics.raw
  odaslive -c odas/odas2mic.cfg
  python scripts/04_odas_stream_to_fifo.py --device 2 --in-channels 8 --mic1 0 --mic2 1 --fifo mics.raw
"""

import argparse
import sys

import numpy as np
import sounddevice as sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--mic1", type=int, default=0)
    ap.add_argument("--mic2", type=int, default=1)
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--fifo", type=str, default="mics.raw")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.mic1 == args.mic2:
        raise ValueError("mic1 and mic2 must be different")
    if max(args.mic1, args.mic2) >= args.in_channels:
        raise ValueError("mic index out of range for --in-channels")

    # Open FIFO for writing (will block until ODAS opens it for reading)
    with open(args.fifo, "wb", buffering=0) as f:

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            x = indata[:, [args.mic1, args.mic2]].astype(np.float32, copy=False)
            x = np.clip(x, -1.0, 1.0)
            pcm = (x * 32767.0).astype(np.int16)
            f.write(pcm.tobytes(order="C"))

        with sd.InputStream(
            device=args.device,
            channels=args.in_channels,
            samplerate=int(args.samplerate),
            blocksize=int(args.blocksize),
            dtype="float32",
            callback=callback,
        ):
            print("Streaming to FIFO for ODAS. Ctrl+C to stop.")
            print(f"fifo={args.fifo} fs={args.samplerate} in-ch={args.in_channels} mic1={args.mic1} mic2={args.mic2}")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("\nStopped.")


if __name__ == "__main__":
    main()
