#!/usr/bin/env python3
"""Real-time 2-mic TDOA/DOA demo using GCC-PHAT.

Outputs:
- tau_s: estimated TDOA between mic1 and mic2
- angle_deg: angle relative to the mic axis (far-field assumption)
- conf: peak-to-average metric (higher is better)

This is the most practical "baseline" to understand how TDOA behaves in your room.
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

SPEED_OF_SOUND = 343.0  # m/s


def design_highpass(fs: int, cutoff_hz: float, order: int = 4):
    if cutoff_hz is None or cutoff_hz <= 0:
        return None
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    if wn >= 1.0:
        raise ValueError("highpass cutoff must be < Nyquist")
    return butter(order, wn, btype="highpass", output="sos")


def gcc_phat(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau: float):
    """Return (tau_seconds, conf, cc) using GCC-PHAT.

    tau_seconds sign convention:
      tau > 0 => sig lags refsig (sound arrived earlier at refsig channel)
    """
    sig = np.asarray(sig, dtype=np.float32)
    refsig = np.asarray(refsig, dtype=np.float32)

    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)
    denom = np.abs(R)
    R /= (denom + 1e-12)

    cc = np.fft.irfft(R, n=n)

    max_shift = int(min(max_tau * fs, n // 2))
    # shift correlation so that 0 lag is centered
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    abscc = np.abs(cc)
    shift = int(np.argmax(abscc) - max_shift)
    tau = shift / float(fs)

    peak = float(abscc[max_shift + shift])
    conf = peak / (float(np.mean(abscc)) + 1e-12)

    return tau, conf, cc


@dataclass
class RingBuffer:
    data: np.ndarray
    write_pos: int = 0
    filled: bool = False

    def push(self, x: np.ndarray):
        """Push (frames, channels)."""
        n = x.shape[0]
        L = self.data.shape[0]
        if n >= L:
            self.data[:] = x[-L:]
            self.write_pos = 0
            self.filled = True
            return

        end = self.write_pos + n
        if end <= L:
            self.data[self.write_pos:end] = x
        else:
            k = L - self.write_pos
            self.data[self.write_pos:] = x[:k]
            self.data[: end - L] = x[k:]
        self.write_pos = (self.write_pos + n) % L
        if self.write_pos == 0:
            self.filled = True

    def get_latest(self):
        """Return data in time order (L, channels)."""
        if not self.filled and self.write_pos == 0:
            return self.data.copy()
        L = self.data.shape[0]
        idx = self.write_pos
        return np.concatenate((self.data[idx:], self.data[:idx]), axis=0)


def rms_db(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    v = float(np.mean(x * x) + 1e-12)
    return 10.0 * math.log10(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--mic1", type=int, default=0)
    ap.add_argument("--mic2", type=int, default=1)
    ap.add_argument("--mic-distance", type=float, default=4.0)
    ap.add_argument("--window-ms", type=float, default=120.0)
    ap.add_argument("--hop-ms", type=float, default=20.0)
    ap.add_argument("--blocksize", type=int, default=0, help="0 lets PortAudio choose")
    ap.add_argument("--highpass-hz", type=float, default=0.0)
    ap.add_argument("--min-rms-db", type=float, default=-45.0, help="energy gate")
    ap.add_argument("--print-hz", type=float, default=10.0)
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.mic1 == args.mic2:
        raise ValueError("mic1 and mic2 must be different")
    if max(args.mic1, args.mic2) >= args.in_channels:
        raise ValueError("mic index out of range for --in-channels")

    fs = int(args.samplerate)
    window_n = int(round(args.window_ms * 1e-3 * fs))
    hop_n = int(round(args.hop_ms * 1e-3 * fs))
    if window_n < 256:
        raise ValueError("window too short")
    if hop_n < 64:
        raise ValueError("hop too short")

    rb = RingBuffer(np.zeros((window_n, 2), dtype=np.float32))

    sos = design_highpass(fs, float(args.highpass_hz))

    max_tau = float(args.mic_distance) / SPEED_OF_SOUND

    last_print = 0.0
    min_dt = 1.0 / max(args.print_hz, 1e-6)

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        x = indata[:, [args.mic1, args.mic2]].astype(np.float32, copy=False)
        rb.push(x)

    with sd.InputStream(
        device=args.device,
        channels=args.in_channels,
        samplerate=fs,
        dtype="float32",
        blocksize=args.blocksize,
        callback=callback,
    ):
        print("Running GCC-PHAT live. Ctrl+C to stop.")
        print(f"device={args.device} fs={fs} ch={args.in_channels} using mic1={args.mic1} mic2={args.mic2} d={args.mic_distance}m")
        try:
            while True:
                time.sleep(hop_n / fs)
                buf = rb.get_latest()
                x1 = buf[:, 0]
                x2 = buf[:, 1]

                # energy gate
                lvl = max(rms_db(x1), rms_db(x2))
                if lvl < args.min_rms_db:
                    continue

                if sos is not None:
                    x1f = sosfilt(sos, x1)
                    x2f = sosfilt(sos, x2)
                else:
                    x1f, x2f = x1, x2

                tau_s, conf, _ = gcc_phat(x2f, x1f, fs=fs, max_tau=max_tau)

                # far-field angle relative to axis
                v = (SPEED_OF_SOUND * tau_s) / float(args.mic_distance)
                v = float(np.clip(v, -1.0, 1.0))
                angle_deg = math.degrees(math.asin(v))

                # (optional) 1D position along the mic line segment, assuming source is on the line
                # mic1 at x=0, mic2 at x=d; tau = (d2 - d1)/c, and for x in [0,d]: d2-d1 = d - 2x
                x_line = (float(args.mic_distance) - SPEED_OF_SOUND * tau_s) / 2.0

                now = time.time()
                if now - last_print >= min_dt:
                    last_print = now
                    side = "closer_to_mic1" if tau_s > 0 else "closer_to_mic2" if tau_s < 0 else "center"
                    print(
                        f"t={now:.3f}  tau={tau_s*1e3:+7.3f} ms  angle={angle_deg:+6.1f} deg  {side}  conf={conf:6.2f}  x_line≈{x_line:5.2f} m"
                    )

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
