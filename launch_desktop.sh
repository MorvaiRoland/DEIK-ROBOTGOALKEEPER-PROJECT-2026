#!/usr/bin/env bash
# =============================================================================
# DEIK Robot Foci Kapus – Asztali Indító Szkript
# =============================================================================

# Projekt gyökérkönyvtára
PROJECT_DIR="/home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026"
cd "$PROJECT_DIR"

# 1. USB memórialimit törlése a XIMEA kamerákhoz (szükség esetén sudo nélkül próbálja)
if [ -f /sys/module/usbcore/parameters/usbfs_memory_mb ]; then
    CURRENT_VAL=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo "0")
    if [ "$CURRENT_VAL" != "0" ]; then
        echo 0 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb > /dev/null 2>&1 || true
    fi
fi

# 2. Python virtuális környezet aktiválása
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# 3. Főprogram elindítása
exec python3 "$PROJECT_DIR/src/main.py" "$@"
