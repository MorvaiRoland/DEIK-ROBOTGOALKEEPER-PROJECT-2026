# 🛠 Hardver és GPIO Diagnosztikai Eszközök (Hardware Diagnostics)

Ez a mappa a **DEIK Robot Foci Kapus** sztereó kamerarendszerének (Ximea MC023CG-SY-UB) és a GPIO hardveres szinkronizációjának ellenőrzésére, beállítására és hibakeresésére szolgáló diagnosztikai szkripteket tartalmazza.

---

## 📋 Eszközök Áttekintése

| Szkript | Cél és Funkció | Használat |
|---|---|---|
| [`hw_sync_verify.py`](file:///home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026/tools/hw_sync_verify.py) | **Szinkron Minőség Mérés**: Megméri a két kamera timestampjei közötti eltérést (jitter & drift), és kiértékeli a HW szinkron működését. | `python tools/hw_sync_verify.py` |
| [`gpio_read_level.py`](file:///home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026/tools/gpio_read_level.py) | **GPI Bemenet Olvasás**: Valós időben monitorozza a SLAVE Pin 5 (Szürke / IN1) opto-izolált bemeneti szintjét. | `python tools/gpio_read_level.py` |
| [`gpio_test_master.py`](file:///home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026/tools/gpio_test_master.py) | **Master Trigger Generátor**: A MASTER Pin 3 (Zöld / OUT1) kimenetén periodikus expozíciós trigger jelet ad ki. | `python tools/gpio_test_master.py` |
| [`gpio_test_slave.py`](file:///home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026/tools/gpio_test_slave.py) | **Slave Trigger Vevő**: Teszteli, hogy a SLAVE kamera képes-e a bejövő hardveres él-trigger hatására képeket rögzíteni. | `python tools/gpio_test_slave.py` |
| [`multimeter_test.py`](file:///home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026/tools/multimeter_test.py) | **DC Szint Mérés**: A MASTER GPO portjait fix 3.3V magas szintre állítja kézi multiméteres feszültségméréshez. | `python tools/multimeter_test.py` |
| [`test_bidir_ports.py`](file:///home/student/Dokumentumok/DEIK-ROBOTGOALKEEPER-PROJECT-2026/tools/test_bidir_ports.py) | **Kétirányú Port Teszt**: Ellenőrzi a Master és Slave közötti kétirányú GPIO vonalak átvitelét (HIGH/LOW szintek). | `python tools/test_bidir_ports.py` |

---

## 🔌 Kábel Bekötési Referencia (CBL-702-8P-SYNC-5M0)

| Pin | Szín | Funkció | Leírás |
|---|---|---|---|
| **Pin 1** | Fekete | GND | Tápellátás földelés |
| **Pin 2** | Fehér | PWR | Külső tápellátás (+12V...+24V) |
| **Pin 3** | Zöld | OUT1 (GPO1) | Opto-izolált kimenet (Master EXPOSURE_ACTIVE) |
| **Pin 4** | Sárga | OUT-GND | Opto-izolált kimenet referencia földelés |
| **Pin 5** | Szürke | IN1 (GPI1) | Opto-izolált bemenet (Slave trigger bemenet) |
| **Pin 6** | Rózsaszín | IN-GND | Opto-izolált bemenet referencia földelés |
| **Pin 7** | Kék | ISO-GND | Izolált közös föld |
| **Pin 8** | Piros | INOUT1 | Bidirekcionális TTL I/O (PORT2) |

---

## 🚀 Tipikus Hibakeresési Folyamat

### 1. Opto-izolált HW Szinkron Tesztelése:
Nyiss két külön terminált ugyanazon a gépen:
```bash
# 1. Terminál: Trigger jel adása a Master kamerával
python tools/gpio_test_master.py

# 2. Terminál: Szint figyelése a Slave kamerán
python tools/gpio_read_level.py
```
* **Sikeres jelátvitel esetén**: A `gpio_read_level.py` 0 és 1 szinteket lát felváltva (~26% duty cycle).
* **Mindig 0 esetén**: Az opto-izolátor nem kap elegendő feszültséget / áramot, külső felhúzó ellenállás vagy táp szükséges.

### 2. Valódi Szinkron Pontosság Verifikálása:
```bash
python tools/hw_sync_verify.py
```
* **STD < 100 µs**: Kiváló hardveres szinkron (mindkét kamera pontosan azonos időpillanatban exponál).
* **STD < 500 µs**: Jó hardveres szinkron.
* **STD > 2 ms**: A hardveres trigger nem működik, a rendszer szoftveres szinkronra esik vissza.
