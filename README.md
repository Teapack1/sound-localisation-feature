# sound-localisation-feature

Minimal, real-time **sound localisation** prototypes for a 2‑microphone test (mics ~4 m apart), intended as a stepping stone toward Sound Detection 2 multi-mic localisation.

> With only 2 microphones you can reliably estimate **TDOA** (time-difference of arrival) and a **1D direction/angle relative to the mic axis**. A unique 2D position in the room is not observable without more microphones or additional assumptions.

## Quick start

### 1) Create venv + install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) List audio devices + pick your input device index

```bash
python scripts/00_list_devices.py
```

### 3) Run the recommended baseline (GCC‑PHAT)

Example: audio interface provides many channels, but you want channel 0 and 1.

```bash
python scripts/01_gcc_phat_live.py \
  --device 2 \
  --in-channels 8 \
  --mic1 0 --mic2 1 \
  --mic-distance 4.0 \
  --samplerate 48000 \
  --highpass-hz 200.0
```

Make sharp sounds between the microphones (clap, knock, click). The script prints TDOA, estimated angle, and a simple confidence metric.

## What’s included

- `scripts/01_gcc_phat_live.py` – recommended, robust TDOA via GCC‑PHAT (NumPy/SciPy).
- `scripts/02_pyroom_srp_phat_live.py` – SRP‑PHAT DOA using **pyroomacoustics**.
- `scripts/03_pyroom_music_live.py` – MUSIC DOA using **pyroomacoustics**.
- `scripts/04_odas_stream_to_fifo.py` + `odas/odas2mic.cfg` – stream your 2 channels into **ODAS** (external C subsystem) and read JSON localisation output.
- `scripts/record_multichannel_wav.py` – record a multichannel WAV for offline debugging.

## Notes on interpreting results

- **Sign convention** in scripts:
  - `tau_s > 0` means “mic2 lags mic1” (sound arrived first at mic1 → source is closer to mic1 side of the axis).
  - `tau_s < 0` means “mic1 lags mic2”.
- Angle estimate is computed as: `angle = asin(c * tau / d)` where `d` is mic spacing.
- If the room is very reverberant, confidence will drop (peak becomes less sharp). Using a high-pass filter often helps for transient sounds.

## ODAS demo (optional)

ODAS is not a Python library; it’s a separate real-time audition system.

1) Install ODAS (Ubuntu): follow ODAS wiki/README.
2) Create FIFO and run ODAS:

```bash
rm -f mics.raw && mkfifo mics.raw
odaslive -c odas/odas2mic.cfg
```

3) In another terminal, stream audio into the FIFO:

```bash
python scripts/04_odas_stream_to_fifo.py --device 2 --in-channels 8 --mic1 0 --mic2 1
```

ODAS terminal should print JSON localisation estimates.
