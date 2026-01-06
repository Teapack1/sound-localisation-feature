# Integration with TensorFlow Detection

## Overview

This guide shows how to wire the localization prototypes into your existing TensorFlow glass-clink detector for **event-gated, real-time inference**.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Scarlett 18i8 (8 channels, 48 kHz, real-time) │
└────────────────┬────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │ Ring Buffer    │
         │ (3-5 seconds)  │ All 8 channels, continuous
         └───────┬────────┘
                 │
           ┌─────┴─────┐
           │           │
     ┌─────▼────┐  ┌──▼──────────┐
     │ TF Model │  │ Localization│
     │ Detection│  │ (Idle state)│
     │ (Always) │  │             │
     └─────┬────┘  └─────────────┘
           │
      [Event detected at time t]
           │
           ▼
     ┌──────────────┐
     │ Extract      │ Cut 100-200 ms around event
     │ Window       │ from ring buffer (all channels)
     │              │
     └──────┬───────┘
            │
       ┌────▼─────────┐
       │ Localization │ Run TDOA, SRP, MUSIC, etc.
       │ (Gated)      │ Only when detection fires
       │              │
       └────┬─────────┘
            │
      ┌─────▼──────────┐
      │ Zone Mapping   │ position → zone ID
      │                │
      └─────┬──────────┘
            │
      ┌─────▼──────────┐
      │ OSC/ArtNet     │ Send to lighting
      │                │
      └────────────────┘
```

## Step 1: Ring Buffer Setup

Keep a continuous rolling buffer of all channels:

```python
import numpy as np
import sounddevice as sd
import queue

# Configuration
RING_BUFFER_SIZE_S = 3.0  # Keep last 3 seconds
N_CHANNELS = 8
FS = 48000

ring_buffer_len = int(RING_BUFFER_SIZE_S * FS)
ring_buffer = np.zeros((ring_buffer_len, N_CHANNELS), dtype=np.float32)
ring_buffer_idx = 0

q = queue.Queue(maxsize=200)

def audio_callback(indata, frames, timeinfo, status):
    """Sounddevice callback: always capture."""
    global ring_buffer_idx
    
    try:
        q.put_nowait(indata.copy())
    except queue.Full:
        pass

# Main loop
with sd.InputStream(
    device=device_index,
    channels=N_CHANNELS,
    samplerate=FS,
    blocksize=1024,
    callback=audio_callback,
    dtype='float32'
):
    print("Capturing multichannel audio...")
    
    while True:
        # Get audio block from callback
        block = q.get()  # shape: (blocksize, N_CHANNELS)
        
        # Update ring buffer
        n = block.shape[0]
        if ring_buffer_idx + n <= ring_buffer_len:
            ring_buffer[ring_buffer_idx:ring_buffer_idx + n] = block
            ring_buffer_idx += n
        else:
            # Wrap around
            remaining = ring_buffer_len - ring_buffer_idx
            ring_buffer[ring_buffer_idx:] = block[:remaining]
            ring_buffer[:n - remaining] = block[remaining:]
            ring_buffer_idx = n - remaining
```

## Step 2: TensorFlow Detection

Run your detection model on a mix of channels:

```python
import tensorflow as tf

# Load your trained model
detector_model = tf.lite.Interpreter(model_path="glass_detector.tflite")
detector_model.allocate_tensors()

input_details = detector_model.get_input_details()
output_details = detector_model.get_output_details()

DETECTION_THRESHOLD = 0.7
DEBOUNCE_TIME_S = 0.5
last_detection_time = 0.0

def run_detection(audio_window):
    """
    audio_window: (samples, N_CHANNELS)
    Returns: (is_glass, probability, class_name)
    """
    # Mix channels (or use specific subset)
    mixed = np.mean(audio_window, axis=1)
    
    # Preprocess for model
    # (Your preprocessing: mel-spec, normalization, etc.)
    mel_spec = compute_mel_spectrogram(mixed, sr=FS)
    mel_spec = np.expand_dims(mel_spec, axis=(0, -1))  # Add batch, channel dims
    
    # Inference
    detector_model.set_tensor(input_details[0]['index'], mel_spec.astype(np.float32))
    detector_model.invoke()
    
    logits = detector_model.get_tensor(output_details[0]['index'])
    prob = tf.nn.softmax(logits)[0]
    
    class_idx = np.argmax(prob)
    class_name = ["silence", "glass_clink", "clap", "knock"][class_idx]
    
    return prob[1] >= DETECTION_THRESHOLD, prob[1], class_name
```

Integrate into main loop:

```python
import time

while True:
    # ... ring buffer update ...
    
    # Every 50 ms, run detection
    if (something_like_frame_count % 2400) == 0:  # 48000 Hz / 20 Hz
        # Extract latest 1 second for detection
        latest_window = ring_buffer  # or last 1 second
        
        is_detected, prob, class_name = run_detection(latest_window)
        
        if is_detected and (time.time() - last_detection_time) > DEBOUNCE_TIME_S:
            detection_timestamp = time.time()
            last_detection_time = detection_timestamp
            
            print(f"[TF DETECTION] {class_name} detected, prob={prob:.2f}")
            
            # Trigger localization (next step)
            localization_triggered = True
            trigger_timestamp_idx = ring_buffer_idx  # Remember buffer position
```

## Step 3: Event-Gated Localization

When detection fires, extract a tight window and localize:

```python
from prototypes._common import (
    gcc_phat_tdoa,
    tdoa_to_azimuth,
    tdoa_to_position_1d,
    normalize_signal,
    design_highpass,
)
from scipy.signal import sosfilt

# Localization config
MIC1_INDEX = 0
MIC2_INDEX = 1
MIC_DISTANCE_M = 4.0
EVENT_WINDOW_MS = 100  # Extract 100 ms around detection
HIGHPASS_HZ = 2000    # For glass clinks

def extract_event_window(ring_buffer, center_idx, duration_ms):
    """Extract window centered on detection time."""
    duration_samples = int(duration_ms * FS / 1000)
    half_win = duration_samples // 2
    
    start_idx = (center_idx - half_win) % len(ring_buffer)
    end_idx = (center_idx + half_win) % len(ring_buffer)
    
    if start_idx < end_idx:
        return ring_buffer[start_idx:end_idx]
    else:
        # Wrapped
        return np.concatenate([
            ring_buffer[start_idx:],
            ring_buffer[:end_idx]
        ])

def localize_event(event_window):
    """
    event_window: (samples, N_CHANNELS)
    Returns: (position_m, azimuth_deg, confidence)
    """
    # Extract channels of interest
    x1 = event_window[:, MIC1_INDEX].astype(np.float32)
    x2 = event_window[:, MIC2_INDEX].astype(np.float32)
    
    # Optional: High-pass filter to isolate glass sound
    hp_sos = design_highpass(FS, HIGHPASS_HZ, order=4)
    x1 = sosfilt(hp_sos, x1)
    x2 = sosfilt(hp_sos, x2)
    
    # Normalize
    x1 = normalize_signal(x1)
    x2 = normalize_signal(x2)
    
    # GCC-PHAT TDOA
    max_tau = MIC_DISTANCE_M / 343.0
    tau_s, shift_samp, confidence = gcc_phat_tdoa(
        x1, x2, fs=FS, max_tau=max_tau, interp=4
    )
    
    # Convert to position and azimuth
    azimuth_deg = tdoa_to_azimuth(tau_s, MIC_DISTANCE_M)
    position_m = tdoa_to_position_1d(tau_s, MIC_DISTANCE_M)
    
    return position_m, azimuth_deg, confidence

# In main loop, when detection fires:
if localization_triggered:
    event_window = extract_event_window(ring_buffer, trigger_timestamp_idx, EVENT_WINDOW_MS)
    
    if event_window.shape[0] > 100:  # Sanity check
        position_m, azimuth_deg, conf = localize_event(event_window)
        
        if conf > 3.0:  # Confidence gate
            print(f"[LOCALIZATION] pos={position_m:.2f}m, az={azimuth_deg:.1f}°, conf={conf:.2f}")
            
            # Proceed to zone mapping
            localization_triggered = False
        else:
            print(f"[LOCALIZATION] Low confidence: {conf:.2f}, rejecting")
            localization_triggered = False
```

## Step 4: Zone Mapping

Convert (x, y) position to a **zone ID** for your venue:

```python
# Define zones for your venue (example: bar)
ZONES = {
    "bar_south": {"x_range": (0, 2), "y_range": (0, 1)},
    "bar_north": {"x_range": (2, 4), "y_range": (0, 1)},
    "sofa_area": {"x_range": (0, 4), "y_range": (1, 3)},
    "terrace": {"x_range": (0, 4), "y_range": (3, 5)},
}

def position_to_zone(position_m, azimuth_deg):
    """
    With only 2 mics (1D localization), map to closest zone.
    """
    # For 2-mic setup, estimate y based on azimuth
    x = position_m
    y = x * np.tan(np.radians(azimuth_deg)) + 1.5  # Heuristic
    
    for zone_name, bounds in ZONES.items():
        x_range, y_range = bounds["x_range"], bounds["y_range"]
        if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
            return zone_name, (x, y)
    
    return "unknown", (x, y)
```

## Step 5: OSC/ArtNet Output

Send zone to lighting system:

```python
import socket

class OSCClient:
    def __init__(self, host="127.0.0.1", port=8000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host
        self.port = port
    
    def send(self, address, *args):
        """Send OSC message."""
        msg = build_osc_message(address, args)
        self.sock.sendto(msg, (self.host, self.port))

def build_osc_message(address, args):
    """Simple OSC message builder."""
    # Simplified; use python-osc library for production
    msg = f"/{address}\0".encode() + b"\0\0\0"
    # ... (truncated, use python-osc)
    return msg

osc_client = OSCClient(host="192.168.1.100", port=9000)  # Lighting rig IP

# When localization succeeds:
zone_name, (x, y) = position_to_zone(position_m, azimuth_deg)
print(f"[OUTPUT] Glass clink in {zone_name} at ({x:.2f}, {y:.2f})")

# Send OSC
osc_client.send("lighting/clink", zone_name, 255, 128, 0)  # Orange flash
```

## Complete Example

See `examples/tf_detection_with_localization.py` for a full, working script.

## Timing & Latency

| Component | Latency | Notes |
|-----------|---------|-------|
| Ring buffer | 0 | Continuous |
| TF detection | 50-100 ms | Runs every 50 ms |
| Detection timestamp | ~30 ms | Window-based |
| Audio extraction | ~0 ms | In-memory |
| Localization (TDOA) | 20-50 ms | FFT + peak finding |
| Zone mapping | <1 ms | Lookup |
| OSC send | <5 ms | Network |
| **Total** | **100-150 ms** | From sound to light |

## Debugging

### No Detections
- Check TF model threshold
- Verify mel-spec preprocessing matches training
- Log raw model outputs

### Localizations Wrong
- Inspect GCC-PHAT correlation plot
- Check mic distance, coordinates are correct
- Try increasing highpass cutoff
- Increase window length if too noisy

### Latency Too High
- Reduce detection window size
- Use faster localization (TDOA vs SRP)
- Lower detection frequency (100 ms → 200 ms)

## Next: Production Hardening

For deployment:
1. **Add robust error handling:** Timeouts, exception catching, logging
2. **Monitor performance:** Log latencies, false-positive rates
3. **Migrate to ODAS (C):** For better reliability
4. **Add watchdog:** Restart if process hangs
5. **Use systemd:** Auto-restart on reboot

See `docs/DEPLOYMENT.md` for production checklist.
