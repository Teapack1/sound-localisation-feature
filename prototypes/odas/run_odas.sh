#!/usr/bin/env bash

# Run ODAS localization + Python streamer for 2-mic setup
# Usage:
#   Terminal 1: bash run_odas.sh
#   Terminal 2: python odas_stream_to_fifo.py --device INDEX --in-channels 8 --mic1 0 --mic2 1

set -euo pipefail

cd "$(dirname "$0")"

# Remove old FIFO if it exists
rm -f mics.raw

# Create FIFO (named pipe)
mkfifo mics.raw

echo "ODAS setup ready. Created FIFO: mics.raw"
echo ""
echo "Now in another terminal, run:"
echo "  python odas_stream_to_fifo.py --device INDEX --in-channels 8 --mic1 0 --mic2 1"
echo ""
echo "Then ODAS will print JSON localization results below:"
echo ""

# Run ODAS
odaslive -c odas_2mic.cfg
