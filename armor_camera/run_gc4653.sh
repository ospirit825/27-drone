#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/usr/bin/lib:/mnt/system/usr/lib:${LD_LIBRARY_PATH:-}"
exec python3 armor_detector.py --start-csi "$@"
