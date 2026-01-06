# Sound Localization Algorithms: Deep Dive

## Overview

This document explains the 5 localization algorithms implemented in this repository, their mathematical foundations, strengths, weaknesses, and when to use each one.

**TL;DR:** Start with **Prototype 1 (GCC-PHAT TDOA)** for most real-world deployments. Layer **Prototype 2 (SRP-PHAT)** if reverberation is problematic. Prototypes 3-5 are research/reference implementations.

---

## 1. GCC-PHAT TDOA (Prototype 1)

### What It Is

**Generalized Cross-Correlation with Phase Transform** estimates the time difference of arrival (TDOA) between two microphone signals using cross-correlation in the frequency domain, with phase weighting.

### How It Works

**Step 1: Frequency Domain**
```
X1(f) = FFT(x1)    # Spectrum of mic1
X2(f) = FFT(x2)    # Spectrum of mic2
R(f) = X1(f) * X2*(f)  # Cross-spectrum
```

**Step 2: PHAT Weighting (Magic!)**
```
R_PHAT(f) = R(f) / |R(f)|  # Normalize by magnitude
```
This step **throws away amplitude information** and keeps only phase. Why?
- If mic1 is near a loud speaker, `|X1|` is huge, `|X2|` is tiny
- Vanilla cross-correlation would be dominated by mic1
- PHAT weighting makes them equally important by normalizing: only the **phase alignment** (timing) matters

**Step 3: Time Domain**
```
cc(lag) = IFFT(R_PHAT)  # Generalized cross-correlation
TDOA = argmax(cc)       # Find peak lag
```

The peak tells you: **by how many samples is mic2 delayed relative to mic1?**

**Step 4: Convert to Position**
```
Speed of sound = 343 m/s
TDOA (seconds) × 343 = distance difference
Multilaterate if you have 3+ mics
```

### Why It's Robust in Noise

**Scenario:** Loud crowd noise at mic1, quiet glass clink at mic2.
- Vanilla correlation: Crowd noise wins (louder signal)
- GCC-PHAT: Throws away "louder"; looks for **aligned sharp edges**
  - Crowd noise: broad, smeared, incoherent across frequencies
  - Glass clink: sharp transient, coherent phase, same moment on all frequencies
  - Result: PHAT peak is sharper for the clink, even if crowd is louder

### Strengths
- **Low CPU:** One FFT per mic pair, simple peak finding. ~5-8% CPU.
- **Transparent:** Easy to debug (inspect TDOA, peak sharpness, PHAT correlation plot)
- **Robust to different mic levels:** Amplitude-agnostic
- **Real-time:** 20-50 ms end-to-end latency

### Weaknesses
- **Reverberation:** Multiple reflections create multiple peaks in correlation. Might lock onto reflection instead of direct sound.
- **Overlapping events:** If two sources overlap in the window, peak might be meaningless
- **Short time windows needed:** To avoid multiple peaks, use 80-150 ms windows

### When to Use
1. **Your first prototype.** Fast to implement, easy to validate.
2. **Clean rooms or outdoor:** Minimal reverberation.
3. **Single, well-separated events:** One glass clink at a time.
4. **Embedded systems:** Raspberry Pi or ARM boards with tight CPU budget.

### Parameters

| Parameter | Typical | Effect |
|-----------|---------|--------|
| window_ms | 80-120 | Longer = more stable but slower. Shorter = faster but noisier. |
| hop_ms | 20-50 | How often to recompute. Lower = more responsive. |
| highpass_hz | 500-2000 | For glass clinks, use 2000 Hz to suppress crowd noise. |
| confidence_threshold | 3.0-5.0 | Reject weak peaks. Higher = fewer false detections. |
| interp | 4-8 | GCC interpolation. Higher = finer delay estimates, more CPU. |

### References
- Knapp & Carter (1976). "The Generalized Correlation Method for Estimation of Time Delay." IEEE Trans. ASSP.
- MathWorks GCC-PHAT: https://www.mathworks.com/help/phased/ref/gccphat.html

---

## 2. SRP-PHAT Beamforming (Prototype 2)

### What It Is

**Steered Response Power with Phase Transform** is a **grid-search beamforming approach**. Instead of computing one TDOA, it scans a grid of candidate **source positions** and ranks them by how much "steered response power" they produce.

### How It Works

**Step 1: Define a Grid**
```
For each candidate position (x, y) in the room:
    Calculate expected arrival times at each mic
    tau[i] = distance_from_source_to_mic[i] / speed_of_sound
```

**Step 2: Steer & Sum**
```
For each grid point (x, y):
    For each mic pair (i, j):
        Compute PHAT-weighted cross-spectrum
        Apply delay (tau[i] - tau[j])
        Weight by confidence (peak sharpness of cross-correlation)
    Sum all contributions → SRP(x, y)
```

**Step 3: Find Peak**
```
Best position = argmax(SRP grid)
```

### Why It's Robust in Reverberation

Consider a reflective room where:
- Direct sound arrives at mic1 first, then mic2
- Reflection arrives 50ms later (bounced off wall)
- Both create peaks in the TDOA correlation

**GCC-PHAT:** Might lock onto the reflection peak, giving wrong TDOA

**SRP-PHAT:** Many mic pairs "vote" on the location
- Pair (mic1, mic2): Direct sound gives peak at true location
- Pair (mic1, mic3): Also peaks at true location (different perspective)
- Pair (mic2, mic3): Also peaks at true location
- A reflection-based location would only be voted for by some pairs
- **Voting:** True location has the most votes → robust

### Strengths
- **Robust in reverb:** Graceful degradation if some pairs are "confused"
- **Handles missing channels:** If one mic fails, others still vote
- **Visual heatmaps:** Can plot SRP grid to see room acoustics
- **Symmetric (DOA + distance):** Unlike MUSIC, you get both direction and distance

### Weaknesses
- **Higher CPU:** Grid search scales as O(grid_size). 2D grid = thousands of candidates.
- **Grid resolution trade-off:** Coarse grid = fast but less accurate; fine grid = slow.
- **Intrinsically slower:** Typically 30-100 ms latency (vs 20-50 ms for TDOA)
- **Parameter tuning:** Grid size, spacing, frequency bands all affect accuracy

### When to Use
1. **Reverberant rooms:** Hotel lobbies, tiled bathrooms, concert halls (RT60 > 0.3 s)
2. **When robustness > speed:** 100 ms latency is acceptable
3. **Multiple mics (4+):** SRP really shines with mic diversity
4. **After validating TDOA:** Use SRP as second opinion / cross-check

### Parameters (Pyroomacoustics)

| Parameter | Typical | Effect |
|-----------|---------|--------|
| nfft | 512-2048 | STFT size. Higher = finer frequency resolution. |
| hop | 256-512 | STFT hop. Together with nfft controls time resolution. |
| frames | 4-10 | Accumulate this many STFT frames. More = stable, slower. |
| grid resolution | 2-10 degrees | For 2D grid. Finer = slower. |

### References
- Salvati et al. (2021). "Pyroomacoustics: A Python Package for Audio Room Simulation and Array Processing Algorithms." ICASSP.
- DiBiase (2000). "Microphone Arrays for Speech Enhancement."

---

## 3. MUSIC DOA (Prototype 3)

### What It Is

**Multiple Signal Classification** is a **high-resolution direction of arrival (DOA)** method. It eigendecomposes the spatial covariance matrix to separate signal subspace from noise subspace.

### How It Works (Intuitive)

**Idea:** Sound sources have structure (correlation across mics). Noise is random (no structure).

**Step 1: Covariance Matrix**
```
R = E[x * x^H]  # Spatial covariance of multichannel input
```

**Step 2: Eigendecompose**
```
[E, lambda] = eig(R)

Large eigenvalues = signal subspace (sources)
Small eigenvalues = noise subspace (random noise)
```

**Step 3: Scan Angles**
```
For each candidate angle θ:
    a(θ) = steering vector (predicted phase difference at angle θ)
    metric = 1 / |a(θ)^H * En|^2  # Distance to noise subspace
    If a(θ) perpendicular to noise → metric is huge (peak!)
```

**Step 4: Find Peaks**
```
DOA peaks = angles where metric is maximum
```

### Why It's High-Resolution

MUSIC can resolve sources separated by very small angles (e.g., 1 degree apart) with a large array. This is because the eigendecomposition captures the full spatial structure.

**GCC-PHAT:** Only uses one or two mic pairs → limited resolution

**MUSIC:** Uses all mics and their spatial relationships → can resolve fine details

### Strengths
- **High angular resolution:** Can separate sources 1-2 degrees apart with 4+ mics
- **Multiple sources:** Can detect 2-3 sources simultaneously
- **Mathematically elegant:** Solid theoretical foundation

### Weaknesses
- **Needs many samples:** Covariance estimation requires long integration windows (100s of milliseconds)
- **DOA only:** Outputs angle, **not distance**. Can't map to (x, y) position without additional info
- **Fragile with model mismatch:** If mic geometry is wrong, breaks badly
- **Needs 4+ mics:** With only 2 mics, angular resolution is ~±45 degrees (useless)
- **Not recommended for your 2-mic setup**

### When to Use
1. **Compact arrays (4+ mics):** Circular, planar geometries with < 1 m spacing
2. **When you need DOA only:** Just which direction is the source?
3. **Low-movement sources:** Recompute DOA every few hundred milliseconds
4. **Not for 2-mic setup:** Included for reference and future 4-mic upgrades

### References
- Schmidt (1986). "Multiple Emitter Location and Signal Parameter Estimation." IEEE Trans. AP.
- Pyroomacoustics DOA module: https://pyroomacoustics.readthedocs.io

---

## 4. ODAS (Prototype 4)

### What It Is

**Open Embedded Audition System** is a **production-grade, real-time C library** for sound source localization, tracking, and separation. It's optimized for embedded devices (Raspberry Pi, BeagleBone) and runs as a separate process.

### How It Works

ODAS pipeline:
```
1. Multichannel audio input (raw PCM from FIFO)
2. Sound source localization (GCCPHAT-based TDOA)
3. Sound source tracking (Kalman filter across time)
4. Separation & beamforming (extract each source cleanly)
5. Post-filtering (noise reduction)
6. JSON output (detected sources + positions + confidence)
```

Core algorithm is **TDOA multilateration**, similar to Prototype 1, but with:
- **Automatic peak detection:** Finds onset of sound
- **Tracking:** Associates peaks across time frames
- **Separation:** Can output clean audio of detected source
- **Production hardening:** Handles edge cases, clipping, dropouts

### Strengths
- **Production-ready:** Used in real robotics systems, tested in the field
- **Separation bonus:** Outputs clean audio of detected sound
- **Low-power:** C implementation, embedded-optimized. Runs on Raspberry Pi ~3-6% CPU
- **Geometry-flexible:** Config file defines mic positions, array shape
- **Mature ecosystem:** Well-documented, active development
- **Real-time:** Built for low-latency (50-150 ms) audio processing

### Weaknesses
- **Configuration-driven:** Less flexible than Python code; hard to tinker
- **Separate process:** IPC overhead (Python ↔ ODAS communication)
- **C library:** Harder to integrate into pure Python pipelines
- **Limited Python bindings:** Need to parse JSON output, no direct Python API
- **DOA angle only:** Doesn't output full 3D position (though can be computed from geometry)

### When to Use
1. **Production deployments:** Venues, smart homes, robots
2. **Low-power embedded:** Raspberry Pi, ARM boards with 500 USD budget
3. **Always-on localization:** Continuous background tracking
4. **When you need separation:** Want clean audio of the detected event
5. **After testing with Prototypes 1-2:** Migrate to ODAS for hardened, optimized version

### How to Integrate with TensorFlow

```
TensorFlow Detection → (if glass detected):
    Extract timestamp from ODAS output JSON
    Find ODAS track at that time
    → Mic position or DOA angle
    → Map to zone
    → Send OSC/ArtNet
```

See `odas/run_odas.sh` and `odas/odas_stream_to_fifo.py` for setup.

### References
- Grondin et al. (2019). "The ODAS Localization System." IEEE ICASSP Workshop.
- GitHub: https://github.com/introlab/odas
- Wiki: https://github.com/introlab/odas/wiki

---

## 5. OpenSoundscape TDOA (Prototype 5)

### What It Is

**OpenSoundscape** is a **Python audio ecology framework** (used for bird song detection, etc.). Prototype 5 uses its localization module to estimate TDOA and multilaterate source positions.

### How It Works

```
1. SynchronizedRecorderArray: Define mic positions
2. Write audio windows to WAV files (OpenSoundscape format)
3. localize_detections(): Compute TDOA via cross-correlation
4. Least-squares solver: Convert TDOA to position
```

Core is **TDOA multilateration**, similar to Prototype 1, but:
- Wrapped in a research-friendly framework
- Tools for validating TDOA cross-correlations
- Integration with PyTorch detection pipelines

### Strengths
- **Research integration:** If you're already in OpenSoundscape ecosystem
- **TDOA validation tools:** Built-in utilities to inspect correlation quality
- **PyTorch-friendly:** Easy to integrate with PyTorch detectors
- **Clean Python API:** No C libraries, all transparent

### Weaknesses
- **Framework overhead:** Slower than raw NumPy/SciPy (100-300 ms latency)
- **File I/O overhead:** Writes WAV files every window (slow on embedded)
- **Less production-hardened:** Research code, not battle-tested like ODAS
- **Not real-time optimized:** Designed for offline audio ecology analysis
- **Overkill for simple 2-mic setup:** Brings in heavy dependencies

### When to Use
1. **Research & development:** Prototyping new SELD architectures
2. **If you're already using OpenSoundscape:** For bird/insect detection
3. **PyTorch + localization:** End-to-end learning integration
4. **Not recommended for production:** Too slow, too much overhead

### References
- OpenSoundscape: https://opensoundscape.org
- GitHub: https://github.com/kitzeslab/opensoundscape

---

## Comparison Table

| Aspect | TDOA | SRP-PHAT | MUSIC | ODAS | OpenSoundscape |
|--------|------|----------|-------|------|----------------|
| **Algorithm** | Cross-correlation (PHAT) | Beamforming grid search | Eigendecomposition | TDOA + tracking | TDOA multilateration |
| **Output** | Position (2D/3D) | Position + direction | Direction only | Position + tracking | Position |
| **Best for 2 mics?** | ✓ Excellent | ✓ Good | ✗ Poor | ✓ Good (production) | ✓ OK (research) |
| **Robustness in reverb** | Moderate | ✓ Excellent | Good | ✓ Excellent | Moderate |
| **CPU (% single core)** | 5-8% | 8-15% | 10-12% | 3-6% (separate) | 8-12% |
| **Latency** | 20-50 ms | 30-100 ms | 30-80 ms | 50-150 ms | 100-300 ms |
| **Real-time** | ✓ | ✓ | ✓ | ✓ | Limited |
| **Separation** | No | No | No | ✓ Yes | No |
| **Language** | Python | Python (pyroomacoustics) | Python | C | Python |
| **Production ready** | ✓ | ✓ | Limited | ✓✓ | No |
| **Learning curve** | Easy | Medium | Hard | Medium | Medium |

---

## Decision Tree

```
"What's your setup?"
|
├─ "2-4 mics, rooms with hard surfaces (reverb)?"
│  ├─ → YES: Start with Prototype 2 (SRP-PHAT)
│  └─ → NO: Start with Prototype 1 (TDOA)
|
├─ "Embedded/low-power (Raspberry Pi)?"
│  ├─ → YES: Use Prototype 4 (ODAS) → production
│  └─ → NO: Prototype 1 or 2
|
├─ "Need to extract clean audio of detected source?"
│  ├─ → YES: Prototype 4 (ODAS only has this)
│  └─ → NO: Any of 1, 2, 3, 5
|
├─ "Already in PyTorch/OpenSoundscape ecosystem?"
│  ├─ → YES: Prototype 5 (integration ease)
│  └─ → NO: Prototype 1 or 2
|
├─ "Compact array (4+ mics close together)?"
│  ├─ → YES: Prototype 3 (MUSIC) for research
│  └─ → NO: Stick with 1, 2, or 4
|
└─ "Ready for production deployment?"
   ├─ → YES: Prototype 4 (ODAS) or hardened Prototype 1
   └─ → NO: Start with Prototype 1 for validation
```

---

## Next Steps

1. **Install and run Prototype 1** with your hardware
2. **Record test data:** Clap, snap fingers, ring bell at known positions
3. **Measure accuracy:** Compare estimated position to actual position
4. **If accuracy OK:** Deploy. If not, try Prototype 2 (SRP-PHAT)
5. **For production:** Migrate to Prototype 4 (ODAS) or hardened version of Prototype 1

For integration with your TensorFlow detector, see `docs/INTEGRATION.md`.
