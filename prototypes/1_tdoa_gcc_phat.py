#!/usr/bin/env python3
"""Real-time 2-mic sound localization using GCC-PHAT TDOA.

Lowest-CPU, lowest-latency approach. Estimates time difference of arrival
using phase-weighted cross-correlation, converts to 1D position.

Best for embedded systems, fast response, relatively clean rooms.

Usage:
    # List devices
    python 1_tdoa_gcc_phat.py --list-devices
    
    # Stream from Scarlett (device 3), channels 0-1, mics 4m apart
    python 1_tdoa_gcc_phat.py --device 3 --in-channels 8 --mic1 0 --mic2 1 --mic-distance 4.0
    
    # With different parameters
    python 1_tdoa_gcc_phat.py --device 3 --in-channels 8 --mic1 0 --mic2 1 \
        --window-ms 120 --hop-ms 20 --highpass-hz 1000 --conf-threshold 5.0
"""

import argparse
import numpy as np
import sounddevice as sd
import sys
from datetime import datetime
from _common import (
    gcc_phat, tdoa_to_position, highpass_filter, rms, db,
    ExponentialMovingAverage, extract_multichannel_chunk
)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 2-mic sound localization with GCC-PHAT TDOA"
    )
    parser.add_argument(
        '--device', type=int, default=None,
        help='Audio device index (or auto-detect)'
    )
    parser.add_argument(
        '--list-devices', action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--in-channels', type=int, default=2,
        help='Number of input channels'
    )
    parser.add_argument(
        '--mic1', type=int, default=0,
        help='Channel index for microphone 1'
    )
    parser.add_argument(
        '--mic2', type=int, default=1,
        help='Channel index for microphone 2'
    )
    parser.add_argument(
        '--mic-distance', type=float, default=1.0,
        help='Distance between mics in meters'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=16000,
        help='Sample rate in Hz'
    )
    parser.add_argument(
        '--window-ms', type=float, default=80,
        help='Window length in milliseconds (longer = smoother but more latency)'
    )
    parser.add_argument(
        '--hop-ms', type=float, default=20,
        help='Hop length in milliseconds (shorter = more frequent updates)'
    )
    parser.add_argument(
        '--highpass-hz', type=float, default=500,
        help='Highpass filter cutoff in Hz (0 = no filter)'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=3.0,
        help='Confidence threshold for valid detections'
    )
    parser.add_argument(
        '--interp', type=int, default=4,
        help='Interpolation factor for TDOA estimation (higher = finer)'
    )
    parser.add_argument(
        '--print-hz', type=float, default=10,
        help='Print updates per second (0 = every frame)'
    )
    parser.add_argument(
        '--smooth', type=float, default=0.3,
        help='EMA alpha for position smoothing (0 = no smoothing)'
    )
    
    args = parser.parse_args()
    
    # List devices
    if args.list_devices:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        sys.exit(0)
    
    # Setup
    sr = args.sample_rate
    window_samples = int(sr * args.window_ms / 1000)
    hop_samples = int(sr * args.hop_ms / 1000)
    
    print(f"\n{'='*80}")
    print(f"GCC-PHAT TDOA 2-Mic Localizer")
    print(f"{'='*80}")
    print(f"Sample rate:        {sr} Hz")
    print(f"Window:             {args.window_ms} ms ({window_samples} samples)")
    print(f"Hop:                {args.hop_ms} ms ({hop_samples} samples)")
    print(f"Highpass:           {args.highpass_hz} Hz")
    print(f"Confidence thresh:  {args.conf_threshold}")
    print(f"Mic distance:       {args.mic_distance} m")
    print(f"Mic 1 / Mic 2:      ch{args.mic1} / ch{args.mic2}")
    print(f"TDOA interp:        {args.interp}x")
    print(f"Position smoothing: alpha={args.smooth}")
    print(f"\nStarting stream (device={args.device})...\n")
    print(f"{'Time':<12} {'TDOA (ms)':<12} {'Pos (m)':<12} {'Conf':<10} {'RMS (dB)':<10}")
    print(f"{'-'*80}")
    
    # Smoothers
    pos_smoother = ExponentialMovingAverage(alpha=args.smooth)
    
    # Open stream
    try:
        with sd.InputStream(
            device=args.device,
            samplerate=sr,
            channels=args.in_channels,
            blocksize=hop_samples,
            latency='low',
            dtype='float32'
        ) as stream:
            
            # Accumulate window
            buffer = np.zeros(window_samples + hop_samples, dtype='float32')
            frame_count = 0
            last_print_frame = 0
            
            while True:
                # Read next hop
                try:
                    chunk = stream.read(hop_samples, allow_overflow=True)
                except Exception as e:
                    print(f"Stream error: {e}", file=sys.stderr)
                    continue
                
                if chunk is None or len(chunk) == 0:
                    continue
                
                # Shift buffer and add new chunk
                buffer[:-hop_samples] = buffer[hop_samples:]
                buffer[-hop_samples:] = chunk[:hop_samples, 0]  # Dummy, will be replaced
                
                # Extract mic channels
                chunk_multi = chunk[:, [args.mic1, args.mic2]]
                buffer_multi = np.vstack([
                    np.zeros((window_samples - hop_samples, 2)),
                    chunk_multi
                ])
                
                frame_count += 1
                
                # Process every print interval
                should_print = (frame_count - last_print_frame) >= (sr / args.hop_ms / args.print_hz) if args.print_hz > 0 else True
                
                if should_print or True:  # Always process
                    # Highpass filter
                    mic1_filtered = highpass_filter(buffer_multi[:, 0], args.highpass_hz, sr)
                    mic2_filtered = highpass_filter(buffer_multi[:, 1], args.highpass_hz, sr)
                    
                    # TDOA estimation
                    tdoa_sec, confidence = gcc_phat(
                        mic1_filtered, mic2_filtered, sr, interp=args.interp
                    )
                    
                    # Convert to position
                    position_m, sol_type = tdoa_to_position(
                        tdoa_sec, args.mic_distance, sr
                    )
                    
                    # Smooth
                    if confidence >= args.conf_threshold:
                        position_m_smooth = pos_smoother.update(position_m)
                    else:
                        position_m_smooth = pos_smoother.value if pos_smoother.value is not None else position_m
                    
                    # RMS
                    rms_mic1 = rms(mic1_filtered)
                    rms_db = db(rms_mic1, ref=1e-5)
                    
                    # Print
                    if should_print:
                        time_str = datetime.now().strftime('%H:%M:%S')
                        tdoa_ms = tdoa_sec * 1000
                        marker = "*" if confidence >= args.conf_threshold else "-"
                        print(
                            f"{time_str:<12} "
                            f"{tdoa_ms:>10.2f} ms "
                            f"{position_m:>10.2f} m "
                            f"{confidence:>8.2f} {marker}  "
                            f"{rms_db:>8.1f} dB"
                        )
                        last_print_frame = frame_count
    
    except KeyboardInterrupt:
        print("\n\nShutdown.")


if __name__ == '__main__':
    main()
