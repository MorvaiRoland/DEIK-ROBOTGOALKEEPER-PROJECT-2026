#!/usr/bin/env bash
# =============================================================================
# DEIK Robot Foci Kapus – Fő Indító Szkript (run.sh)
# =============================================================================
# Használat:
#   ./run.sh            # Ximea valós kamerákkal
#   ./run.sh --mock     # Mock / Webcam teszt módban
# =============================================================================

set -e

# Azonnali usbfs_memory_mb ellenőrzés és frissítés
if [ -f /sys/module/usbcore/parameters/usbfs_memory_mb ]; then
    CURRENT_VAL=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb)
    if [ "$CURRENT_VAL" != "0" ]; then
        echo 0 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb > /dev/null 2>&1 || true
    fi
fi

# Virtuális környezet aktiválása ha létezik
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Főprogram elindítása
exec python src/main.py "$@"
