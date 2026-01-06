#!/usr/bin/env python3
"""Record a multichannel WAV for offline analysis.

Example:
  python scripts/record_multichannel_wav.py --device 2 --channels 8 --seconds 10 --out test.wav
"""

import argparse

import sounddevice as sd
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", type=str, default="multich.wav")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    n = int(args.seconds * args.samplerate)
    print(f"Recording {args.seconds}s, {args.channels} ch @ {args.samplerate} Hz -> {args.out}")
    audio = sd.rec(n, samplerate=args.samplerate, channels=args.channels, device=args.device, dtype="float32")
    sd.wait()
    sf.write(args.out, audio, args.samplerate)
    print("Done.")


if __name__ == "__main__":
    main()
