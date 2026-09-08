#!/usr/bin/env python3
"""
HW GPIO Szinkron Verifikáló Script
===================================
Megmondja EGYÉRTELMŰEN hogy a hardveres szinkron működik-e vagy sem.

Futtatás:
    python tools/hw_sync_verify.py

Mit mér:
    - tsSec timestamp különbség a két kamera között (500 frame)
    - Drift (lassan változó eltolódás) vs. Valódi jitter (véletlen eltérés) szétválasztása
    - Standard deviáció a drift eltávolítása után → ez a VALÓDI szinkron minőség

Értelmezés:
    STD < 100 µs  → kiváló HW szinkron
    STD < 500 µs  → jó HW szinkron
    STD < 2 ms    → elfogadható (USB latencia)
    STD > 2 ms    → HW szinkron NEM MŰKÖDIK megfelelően
"""

import sys
import time
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

try:
    from ximea import xiapi
except ImportError:
    print("HIBA: Ximea xiapi nem elérhető. Aktiváld a venv-et!")
    sys.exit(1)

# Konfiguráció (config.yaml alapján)
N_FRAMES        = 500
FRAME_TIMEOUT   = 5000
TARGET_FPS      = 87
EXPOSURE_US     = 4000
SN_MASTER       = "CACAU2517001"
SN_SLAVE        = "CACAU2546000"
GPO_SELECTOR    = "XI_GPO_PORT1"
GPO_MODE        = "XI_GPO_EXPOSURE_ACTIVE"
GPI_SELECTOR    = "XI_GPI_PORT1"
GPI_MODE        = "XI_GPI_TRIGGER"
TRIG_SOURCE     = "XI_TRG_EDGE_RISING"


def open_camera(serial, role):
    cam = xiapi.Camera()
    cam.open_device_by_SN(serial)
    cam.set_exposure(EXPOSURE_US)
    cam.set_imgdataformat("XI_RGB24")
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")
    cam.set_framerate(float(TARGET_FPS))

    if role == "master":
        cam.set_gpo_selector(GPO_SELECTOR)
        cam.set_gpo_mode(GPO_MODE)
        cam.set_trigger_source("XI_TRG_OFF")
        print(f"  MASTER nyitva ({serial}) – GPIO OUT: {GPO_SELECTOR}")
    else:
        cam.set_gpi_selector(GPI_SELECTOR)
        cam.set_gpi_mode(GPI_MODE)
        cam.set_trigger_source(TRIG_SOURCE)
        cam.set_trigger_selector("XI_TRG_SEL_FRAME_START")
        print(f"  SLAVE nyitva ({serial}) – GPIO IN: {GPI_SELECTOR}")
    return cam


def main():
    print("=" * 60)
    print("  XIMEA HW GPIO SZINKRON VERIFIKÁCIÓ")
    print("=" * 60)

    print("\n[1/4] Kamerák megnyitása...")
    try:
        cam_master = open_camera(SN_MASTER, "master")
        cam_slave  = open_camera(SN_SLAVE,  "slave")
    except Exception as e:
        print(f"\n HIBA: {e}")
        sys.exit(1)

    img_master = xiapi.Image()
    img_slave  = xiapi.Image()

    print("\n[2/4] Képszerzés indítása...")
    cam_master.start_acquisition()
    cam_slave.start_acquisition()
    time.sleep(0.5)

    print(f"\n[3/4] {N_FRAMES} frame-pár mérése...")
    raw_deltas = []

    try:
        for i in range(N_FRAMES):
            cam_master.get_image(img_master, timeout=1000)
            ts_master = img_master.tsSec + img_master.tsUSec / 1_000_000.0

            cam_slave.get_image(img_slave, timeout=FRAME_TIMEOUT)
            ts_slave = img_slave.tsSec + img_slave.tsUSec / 1_000_000.0

            raw_deltas.append((ts_master - ts_slave) * 1000.0)

            if (i + 1) % 100 == 0:
                last = raw_deltas[-1]
                print(f"  {i+1}/{N_FRAMES} – raw_delta: {last:.3f} ms")

    except Exception as e:
        print(f"\n HIBA frame {len(raw_deltas)}: {e}")
    finally:
        cam_master.stop_acquisition()
        cam_slave.stop_acquisition()
        cam_master.close_device()
        cam_slave.close_device()

    if len(raw_deltas) < 50:
        print(f"\n Nem elég adat ({len(raw_deltas)} frame)!")
        sys.exit(1)

    print(f"\n[4/4] Analízis ({len(raw_deltas)} frame)...")
    deltas = np.array(raw_deltas)

    # Drift eltávolítás: csúszó átlag
    window = min(50, len(deltas) // 5)
    smoothed = np.convolve(deltas, np.ones(window) / window, mode='valid')
    trim = (len(deltas) - len(smoothed)) // 2
    detrended = deltas[trim:trim + len(smoothed)] - smoothed

    mean_offset_ms = float(np.mean(deltas))
    std_us         = float(np.std(detrended) * 1000.0)
    max_us         = float(np.max(np.abs(detrended)) * 1000.0)
    p95_us         = float(np.percentile(np.abs(detrended), 95) * 1000.0)

    print()
    print("=" * 60)
    print("  EREDMÉNYEK")
    print("=" * 60)
    print(f"  Átlag clock-offset (drift):  {mean_offset_ms:.3f} ms")
    print(f"  STD (drift nélkül):          {std_us:.1f} µs  <- VALÓDI JITTER")
    print(f"  Max jitter (drift nélkül):   {max_us:.1f} µs")
    print(f"  P95 jitter:                  {p95_us:.1f} µs")
    print()

    if std_us < 100:
        verdict = "KITUNO – HW GPIO szinkron tokeletesen mukodik!"
    elif std_us < 500:
        verdict = "JO – HW GPIO szinkron mukodik"
    elif std_us < 2000:
        verdict = "ELFOGADHATO – USB latencia okozza a jittert"
    else:
        verdict = "HIBA – HW szinkron NEM MUKODIK!\n" \
                  "  Ellenorizd: kabelbekotes, config.yaml sync.enabled=true"

    print(f"  ITELET: {verdict}")
    print("=" * 60)

    # ASCII histogram
    print("\n  Jitter eloszlas (drift nelkul, µs):")
    hist_vals = detrended * 1000.0
    bins = [-2000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 2000]
    counts, edges = np.histogram(hist_vals, bins=bins)
    total = max(len(hist_vals), 1)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c / total * 40)
        pct = c / total * 100
        print(f"  {lo:+6.0f}..{hi:+6.0f} µs |{bar:<40}| {pct:4.1f}%")


if __name__ == "__main__":
    main()
