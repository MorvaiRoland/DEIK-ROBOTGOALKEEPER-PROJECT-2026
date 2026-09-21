"""
GPIO SLAVE Teszt – GPI szint leolvasása
========================================
Ez a szkript CSAK a SLAVE kamerát nyitja meg és figyeli
a Pin 8 (Piros/INOUT1) bemeneti szintjét.

FONTOS: Közben futtasd a tools/gpio_test_master.py szkriptet is
        egy másik terminálban! (MASTER kell a jelhez)

Futtatás (2 terminálban párhuzamosan):
    Terminal 1: python tools/gpio_test_master.py
    Terminal 2: python tools/gpio_test_slave.py

Várható kimenet:
    GPI level: 0 vagy 1 váltakozva (87 Hz-es jel)
    Trigger count növekvő → a SLAVE érzékeli a MASTER jelét!
"""

import time
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("gpio_slave_test")

try:
    from ximea import xiapi
except ImportError:
    logger.error("Ximea xiAPI nincs telepitve!")
    sys.exit(1)

# Konfiguráció
SLAVE_SERIAL  = "CACAU2546000"   # Bal kamera (SLAVE)
GPI_SELECTOR  = "XI_GPI_PORT1"  # Pin 8 (Piros/INOUT1)
GPO_SELECTOR  = "XI_GPO_PORT2"  # Ugyanaz a port, HIGH-Z-re állítjuk
TEST_DURATION = 30               # 30 másodperc teszt


def main():
    cam = xiapi.Camera()
    img = xiapi.Image()

    logger.info("=" * 55)
    logger.info("GPIO SLAVE TESZT – Pin 8 (Piros) jel érzékelése")
    logger.info("=" * 55)
    logger.info("SLAVE sorozatszám: %s", SLAVE_SERIAL)
    logger.info("")
    logger.info("FONTOS: Futtasd párhuzamosan: python tools/gpio_test_master.py")
    logger.info("        és kösd össze a kábeleket (Piros<->Piros, Kek<->Kek)")
    logger.info("")

    logger.info("SLAVE kamera megnyitasa...")
    try:
        cam.open_device_by_SN(SLAVE_SERIAL)
    except Exception as exc:
        logger.error("Kamera megnyitasi hiba: %s", exc)
        logger.error("Ellenőrizd hogy a BAL kamera (SN: %s) csatlakoztatva van!", SLAVE_SERIAL)
        sys.exit(1)

    try:
        cam.set_imgdataformat("XI_RGB24")
        cam.set_exposure(3000)
        cam.set_gain(0.0)
        cam.set_limit_bandwidth_mode("XI_OFF")

        # 1. GPO HIGH-Z – a pin kimenetét ki kell kapcsolni mielőtt inputként használjuk
        try:
            cam.set_gpo_selector(GPO_SELECTOR)
            cam.set_gpo_mode("XI_GPO_HIGH_IMPEDANCE")
            logger.info("GPO PORT2 → HIGH_IMPEDANCE (SLAVE kimenete kikapcsolva)")
        except Exception as exc:
            logger.warning("HIGH-Z beallitas nem sikerult: %s (nem kritikus)", exc)

        # 2. GPI bemenet konfigurálása
        cam.set_gpi_selector(GPI_SELECTOR)
        cam.set_gpi_mode("XI_GPI_TRIGGER")
        logger.info("GPI beallitva: %s → XI_GPI_TRIGGER (Pin 8 bemenet)", GPI_SELECTOR)

        # 3. Trigger forrás beállítása
        cam.set_trigger_source("XI_TRG_EDGE_RISING")
        logger.info("Trigger forrás: XI_TRG_EDGE_RISING")

        # 4. Képszerzés indítása trigger módban
        cam.start_acquisition()

        logger.info("")
        logger.info("=" * 55)
        logger.info("SLAVE FIGYELÉS MEGKEZDVE (30 másodperc):")
        logger.info("  Ha a MASTER fut és a kabel be van kotve,")
        logger.info("  a trigger count 87/mp sebességgel nő!")
        logger.info("=" * 55)
        logger.info("")

        start = time.perf_counter()
        trigger_count = 0
        last_log = start

        while time.perf_counter() - start < TEST_DURATION:
            try:
                # Rövid timeout – ha nincs trigger 200ms alatt, nem blokkolunk sokáig
                cam.get_image(img, timeout=200)
                trigger_count += 1

                now = time.perf_counter()
                if now - last_log >= 2.0:
                    elapsed = now - start
                    rate = trigger_count / elapsed
                    remaining = TEST_DURATION - elapsed

                    if rate > 50:
                        status = "✅ TRIGGER ERKEZETT"
                    elif rate > 10:
                        status = "⚠ GYENGE JEL"
                    else:
                        status = "❌ NINCS JEL"

                    logger.info(
                        "  %s | %d trigger | %.1f/mp | Meg %.0f mp",
                        status, trigger_count, rate, remaining
                    )
                    last_log = now

            except Exception:
                # Timeout = nem jött trigger – logoljuk 2mp-enként
                now = time.perf_counter()
                if now - last_log >= 2.0:
                    elapsed = now - start
                    remaining = TEST_DURATION - elapsed
                    logger.warning(
                        "  ❌ NINCS TRIGGER | %d osszesen | Meg %.0f mp"
                        " | Ellenőrizd a kabelt es a MASTER scriptet!",
                        trigger_count, remaining
                    )
                    last_log = now

        total = time.perf_counter() - start
        logger.info("")
        logger.info("=" * 55)
        if trigger_count > 0:
            logger.info("TESZT KESZ: %d trigger erkezett, %.1f/mp atlag",
                        trigger_count, trigger_count / total)
            if trigger_count / total > 50:
                logger.info("✅ SZINKRONIZACIO MUKODIK! A SLAVE erzekeli a MASTER jelet.")
            else:
                logger.warning("⚠ Gyenge jel – ellenőrizd a kabel erintkezeset!")
        else:
            logger.error("❌ NEM ERKEZETT TRIGGER! Kabel nincs bekotve vagy GPO nem mukodik.")
        logger.info("=" * 55)

    finally:
        try: cam.stop_acquisition()
        except: pass
        try: cam.close_device()
        except: pass
        logger.info("SLAVE kamera bezarva.")


if __name__ == "__main__":
    main()
