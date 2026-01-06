# Troubleshooting Guide

## Audio Capture Issues

### "No default input device"

```bash
python prototypes/1_tdoa_gcc_phat.py --list-devices
```

Find your Scarlett in the list. It will show:
```
  3: Scarlett 18i8, ALSA (8 in, 2 out)
```

Pass `--device 3` to your script.

### "Device only has X input channels"

```bash
# Wrong:
python 1_tdoa_gcc_phat.py --device 3 --in-channels 8 --mic1 2 --mic2 10

# Right:
python 1_tdoa_gcc_phat.py --device 3 --in-channels 8 --mic1 0 --mic2 1
```

Mic indices are 0-based and must be < in-channels.

### Silent Audio / No Waveform

1. **Check system audio levels:**
   ```bash
   alsamixer
   ```
   - Scarlett channel levels should be ~70-80%
   - Don't clip (watch for red indicators)

2. **Check sounddevice is seeing the Scarlett:**
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

3. **Verify microphone is plugged in** and the XLR cable is not damaged.

4. **Test with a different app:**
   ```bash
   arecord -D hw:CARD=Scarlett18i8 -f S32_LE -c 8 -r 48000 test.wav
   # Play it back
   aplay test.wav
   ```

---

## Localization Issues

### "All Localization Estimates Are Jittery / Random"

**Symptom:**
```
[TDOA] delay=3.456 ms | angle=45.2°  | x=2.15 m | conf=7.32
[TDOA] delay=-1.234 ms | angle=-30.1° | x=3.92 m | conf=2.15
[TDOA] delay=0.567 ms  | angle=5.3°  | x=2.04 m | conf=1.80
```

**Root Causes:**
1. **Too short window** → Noise dominates
2. **Confidence threshold too low** → Accepting weak estimates
3. **Microphones too close together** → Small TDOA errors amplify position errors
4. **Strong background noise** → GCC-PHAT peak is unclear

**Fixes:**
```bash
# Increase window length (smoother but slower)
python 1_tdoa_gcc_phat.py --device 3 --window-ms 120 --hop-ms 30

# Increase confidence threshold (fewer, more reliable estimates)
python 1_tdoa_gcc_phat.py --device 3 --conf-threshold 5.0

# Apply high-pass filter (isolate your target sound)
python 1_tdoa_gcc_phat.py --device 3 --highpass-hz 2000
```

### "Localization Always Wrong (Off by ~1 Meter)"

**Check:**
1. **Mic distance correct?**
   ```bash
   # Measure with a ruler from center of mic 1 to center of mic 2
   python 1_tdoa_gcc_phat.py --device 3 --mic-distance 4.0
   ```

2. **Speed of sound for your temperature?**
   - 20°C: 343 m/s (default)
   - 25°C: 346 m/s
   - 10°C: 337 m/s
   (Doesn't usually matter for 1m error unless you're in extreme conditions)

3. **Mic channels swapped?**
   ```bash
   # Try swapping mic indices
   python 1_tdoa_gcc_phat.py --device 3 --mic1 1 --mic2 0
   ```

### "Soundscape / SRP-PHAT Very Slow or Hanging"

**STFT requires more samples than expected:**

```python
# Try shorter window
--nfft 512 --hop 256 --frames 3

# Instead of
--nfft 2048 --hop 1024 --frames 10
```

### "ODAS: No JSON Output"

1. **FIFO not created?**
   ```bash
   ls -la mics.raw
   ```
   Should show `prw------` (pipe).

2. **ODAS binary not in PATH?**
   ```bash
   which odaslive
   # If nothing, install: sudo apt-get install odas
   ```

3. **Streamer not running?**
   Open second terminal:
   ```bash
   python odas/odas_stream_to_fifo.py --device 3 --in-channels 8
   ```

---

## CPU & Performance

### "High CPU Usage (>50%)"

**Check what's using it:**
```bash
top -p $(pgrep -f 1_tdoa_gcc_phat.py)
```

**Reduce CPU:**
```bash
# Decrease GCC interpolation
--interp 1  # Instead of --interp 4

# Reduce FFT size (SRP-PHAT)
--nfft 512  # Instead of --nfft 2048

# Increase hop interval (fewer updates)
--hop-ms 50  # Instead of --hop-ms 20
```

### "Latency Too High"

Measure end-to-end: clap → output printed on screen

**Expected latency:**
- GCC-PHAT: 50-100 ms
- SRP-PHAT: 100-200 ms

If higher:
```bash
# Reduce window length
--window-ms 60 --hop-ms 10
```

---

## Integration with TensorFlow

### "Ring Buffer Not Staying Synchronized"

**Problem:** Audio callback running faster/slower than expected

**Solution:**
```python
# Check drift
print(f"Ring buffer fill: {ring_buffer_idx / len(ring_buffer) * 100:.1f}%")

# If drifting, use time-based indexing instead of counter:
current_time = time.time()
expected_samples = int((current_time - start_time) * FS) % ring_buffer_len
```

### "Detection Time is Wrong"

When TF fires a detection, the audio has **already passed** the microphone.

**Solution:**
Extract window **back in time**:
```python
# Event detected NOW, but glass clink was 100ms ago
window_back_ms = 100
window_sample_idx = (ring_buffer_idx - int(window_back_ms * FS / 1000)) % len(ring_buffer)
```

---

## Hardware-Specific

### Scarlett 18i8 Issues

**USB dropouts / clicks:**
```bash
# Set USB priority
echo 10 | sudo tee /proc/sys/vm/swappiness

# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance
```

**Channels appear in wrong order:**
```bash
# Test each channel individually
for ch in 0 1 2 3 4 5 6 7; do
    python -c "
import sounddevice as sd, numpy as np
data = sd.rec(int(2 * 48000), channels=8, samplerate=48000, device=3)
sd.wait()
print(f'Channel {$ch} RMS:', np.sqrt(np.mean(data[:, $ch]**2)))
    "
done
```

### Raspberry Pi Performance

**To run on Pi:**
```bash
# Install wheels (much faster than compiling)
sudo pip install numpy pyroomacoustics --only-binary=:all:

# Reduce FFT size
python 1_tdoa_gcc_phat.py --blocksize 512 --window-ms 80 --interp 1

# Monitor CPU
watch -n 1 'ps aux | grep python | grep -v grep'
```

---

## Network / OSC Issues

### OSC Messages Not Arriving

```bash
# Test OSC listener
python -m SimpleHTTPServer 8000

# Check firewall
sudo ufw allow 9000/udp

# Test with nc
nc -u -l 0.0.0.0 9000
```

---

## Getting Help

When filing an issue, include:

1. **Hardware:** Scarlett model, mic distance, OS (Ubuntu version)
2. **Command:**
   ```bash
   python 1_tdoa_gcc_phat.py --device 3 --in-channels 8 --mic1 0 --mic2 1 --mic-distance 4.0 --window-ms 80
   ```
3. **Output sample:** Copy 10 lines of terminal output
4. **Audio test:** Does `arecord` work?
   ```bash
   arecord -D hw:CARD=Scarlett18i8 -f S32_LE -c 8 -r 48000 -d 5 test.wav
   ```

Then share test.wav if you suspect audio issue.
