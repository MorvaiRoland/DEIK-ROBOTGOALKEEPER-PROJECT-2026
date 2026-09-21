"""
GPI szint közvetlen leolvasása – Opto-izolált mód diagnosztika
================================================================
Figyeli a Pin 5 (Szurke/IN1) = XI_GPI_PORT1 bemenetet.

Futtasd KÖZBEN a python tools/gpio_test_master.py-t egy masik terminalban!
Kabel bekotve: MASTER Pin3(Zold) → SLAVE Pin5(Szurke)
               MASTER Pin4(Sarga) → SLAVE Pin6(Rozsaszin)
               MASTER Pin7(Kek) ↔ SLAVE Pin7(Kek)
"""
import time
import sys
from ximea import xiapi

SLAVE_SERIAL = "CACAU2546000"

cam = xiapi.Camera()
print("SLAVE kamera megnyitasa...")
try:
    cam.open_device_by_SN(SLAVE_SERIAL)
except Exception as e:
    print(f"Hiba: {e}")
    sys.exit(1)

# PORT1 = Pin 5 (Szurke/IN1) – opto-izolalt bemenet
# PORT1 GPI-n nincs szukseges GPO HIGH-Z beallitas (IN1 egyiranyú bemenet)
print("\n--- GPI PORT1 (Pin 5 / Szurke / IN1) beallitasa ---")
try:
    cam.set_gpi_selector("XI_GPI_PORT1")
    cam.set_gpi_mode("XI_GPI_TRIGGER")
    print("GPI_PORT1 → XI_GPI_TRIGGER: OK  (Pin 5, Szurke, opto-izolalt bemenet)")
except Exception as e:
    print(f"GPI beallitas hiba: {e}")

# GPI szint valós idejű leolvasása 10 másodpercig
print("\n--- GPI szint leolvasas (10 masodperc) ---")
print("Ha a MASTER fut es a kabel OK: 0/1 valtozast kell latni!")
print()

levels = []
start = time.perf_counter()
try:
    while time.perf_counter() - start < 10.0:
        try:
            level = cam.get_gpi_level()
            levels.append(level)
            elapsed = time.perf_counter() - start
            bar = "█" * (level * 10) + "░" * ((1 - level) * 10)
            print(f"  t={elapsed:5.1f}s  GPI szint: {level}  {bar}", end="\r")
        except Exception as e:
            print(f"  Leolvasas hiba: {e}")
        time.sleep(0.005)
except KeyboardInterrupt:
    pass

print("\n")
if levels:
    ones  = sum(1 for l in levels if l > 0)
    zeros = sum(1 for l in levels if l == 0)
    pct   = 100 * ones // len(levels)
    print(f"Osszes minta: {len(levels)}")
    print(f"HIGH (1):  {ones:4d}  ({pct}%)")
    print(f"LOW  (0):  {zeros:4d}  ({100-pct}%)")

    if ones > 0 and zeros > 0:
        print(f"\n✅ JEL ERKEZETT! A SLAVE IN1 pin (PORT1) erzekeli a MASTER OUT1 jelet!")
        print(f"   Becsult duty cycle: ~{pct}% (elvart ~26%)")
        print(f"   → Futtasd: python tools/gpio_test_slave.py a trigger teszthez!")
    elif ones == 0:
        print(f"\n❌ Mindig LOW – az opto-izolator nem kap aram ellatest.")
        print(f"   Megoldas: kell egy kulso 5V + ~1kΩ ellenallas a korbe,")
        print(f"   VAGY hasznad a szoftver szinkront (sync.enabled: false)")
    else:
        print(f"\n⚠ Mindig HIGH – lebego bemenet vagy mas problema")
else:
    print("❌ Nem sikerult szintet olvasni!")

cam.close_device()
print("Kesz.")
