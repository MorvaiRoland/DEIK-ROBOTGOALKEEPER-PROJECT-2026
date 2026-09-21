"""
GPIO MASTER Teszt – Opto-izolált mód
=======================================
Pin 3 (Zöld/OUT1) → XI_GPO_PORT1 → XI_GPO_EXPOSURE_ACTIVE

Futtatás:
    python3 gpio_test_master.py

Közben mérd (V DC állásban):
    MASTER kábel Zöld ér (Pin 3) ↔ Sárga ér (Pin 4 / OUT-GND)
    Elvárt: ~0.9-1.0V DC átlag
"""

import time
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("gpio_master")

try:
    from ximea import xiapi
except ImportError:
    logger.error("Ximea xiAPI nincs telepitve!")
    sys.exit(1)

MASTER_SERIAL = "CACAU2517001"   # Jobb kamera (MASTER)
EXPOSURE_US   = 3000
TARGET_FPS    = 100
TEST_DURATION = 30
GPO_SELECTOR  = "XI_GPO_PORT1"  # Pin 3 (Zold/OUT1) – opto-izolalt kimenet
GPO_MODE      = "XI_GPO_EXPOSURE_ACTIVE"


def main():
    cam = xiapi.Camera()
    img = xiapi.Image()

    logger.info("=" * 55)
    logger.info("GPIO MASTER TESZT – Opto-izolalt mod")
    logger.info("  Pin 3 (Zold/OUT1) = XI_GPO_PORT1")
    logger.info("=" * 55)

    logger.info("MASTER kamera megnyitasa: SN=%s ...", MASTER_SERIAL)
    try:
        cam.open_device_by_SN(MASTER_SERIAL)
    except Exception as exc:
        logger.error("Kamera megnyitasi hiba: %s", exc)
        sys.exit(1)

    try:
        cam.set_imgdataformat("XI_RGB24")
        cam.set_exposure(EXPOSURE_US)
        cam.set_gain(0.0)
        cam.set_limit_bandwidth_mode("XI_OFF")
        cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")
        cam.set_framerate(TARGET_FPS)
        cam.set_trigger_source("XI_TRG_OFF")

        # PORT1 = Pin 3 (Zold/OUT1) – opto-izolalt kimenet
        cam.set_gpo_selector(GPO_SELECTOR)
        cam.set_gpo_mode(GPO_MODE)
        logger.info("GPO beallitva: %s → %s", GPO_SELECTOR, GPO_MODE)
        logger.info("  → Pin 3 (Zold er) ad ki EXPOSURE_ACTIVE jelet")
        logger.info("  → Pin 4 (Sarga er) = OUT-GND referencia")

        cam.start_acquisition()
        logger.info("")
        logger.info("=" * 55)
        logger.info("MOST MERJ MULTIMETERREL (V DC allasban):")
        logger.info("  MASTER kabel Pin 3 (Zold) <-> Pin 4 (Sarga/OUT-GND)")
        logger.info("  Elvart: ~0.9-1.0V DC atlag (87Hz, 3ms expo)")
        logger.info("")
        logger.info("  VAGY olvass GPI szintet a masik terminalban:")
        logger.info("  python3 gpio_read_level.py")
        logger.info("=" * 55)
        logger.info("")

        start = time.perf_counter()
        frame_count = 0
        last_log = start

        while time.perf_counter() - start < TEST_DURATION:
            try:
                cam.get_image(img, timeout=1000)
                frame_count += 1
                now = time.perf_counter()
                if now - last_log >= 2.0:
                    elapsed = now - start
                    logger.info(
                        "  OK %d frame | %.1f FPS | Meg %.0f mp",
                        frame_count, frame_count / elapsed, TEST_DURATION - elapsed
                    )
                    last_log = now
            except Exception as exc:
                logger.warning("Frame hiba: %s", exc)

        total = time.perf_counter() - start
        logger.info("TESZT KESZ: %d frame, %.1f FPS", frame_count, frame_count / total)

    finally:
        try: cam.stop_acquisition()
        except: pass
        try: cam.close_device()
        except: pass
        logger.info("MASTER kamera bezarva.")


if __name__ == "__main__":
    main()
