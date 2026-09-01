# DEIK Robot Foci Kapus / DEIK Robot Goalkeeper

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![PyTorch CUDA](https://img.shields.io/badge/GPU-PyTorch%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/Inference-TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)

---

### Nyelvválasztás / Language Selection
* [🇭🇺 Magyar nyelvű leírás](#-magyar-dokumentáció)
* [🇬🇧 English Documentation](#-english-documentation)
* [👥 Szerzők & Készítők / Authors](#-szerzők--authors)

---

## 🇭🇺 Magyar Dokumentáció

### 📌 Projektáttekintés

A **DEIK Robot Foci Kapus** egy valós idejű, nagy sebességű optikai labdakövető és trajektória-előrejelző rendszer robot foci kapus mechanikák vezérléséhez. A rendszer két darab ipari **Ximea MC023CG-SY-UB** sztereó kamerával figyeli a pályát, és valós időben (100+ FPS) számítja ki a kapu síkjában ($Z=0$) a labda várható becsapódási pontját, idejét és a kapu eltalálásának valószínűségét.

#### Főbb jellemzők és képességek:
* **Kétkamerás Ximea illesztés**: Ipari Global Shutter érzékelők (Sony IMX174, 2.3 MP, max. 165 FPS), dedikált USB3 sávszélesség-kezelés és szinkronizált képgyűjtés.
* **GPU-gyorsított AI detektálás**: TensorRT `.engine` (YOLOv8n / YOLOv10n / RT-DETR) modellek batch=2 GPU inferenciával NVIDIA CUDA architektúrán (>100 FPS).
* **Szín- és mozgásszűrés**: HSV narancssárga színvalidáció (`OrangeBallFilter`) kifejezetten a mintás (pl. Kipsta 4/5) focilabdaspecifikus detektáláshoz, valamint Lucas-Kanade optikai folyam (`OpticalFlowTracker`) követés.
* **Többszintű Kalman-szűrés**: Kameránkénti 2D Kalman-szűrő (`KalmanTracker`) a mérési zaj elnyomására, valamint 3D térbeli Kalman-szűrő a trajektória simítására.
* **3D Sztereó Háromszögelés & Mono Depth Fallback**: Milliméter pontos 3D térbeli koordináta-meghatározás kalibrációs mátrixok alapján, illetve egykamerás mélységbecslés (`MonoDepthEstimator`) arra az esetre, ha a labda csak az egyik kamerában látható.
* **Fizika-alapú pálya-előrejelzés**: Parabolikus mozgásmodell gravitációval ($g=9.81 \text{ m/s}^2$) és aerodinamikai légellenállási tényezővel ($C_d = 0.47$).
* **PyQt6 Grafikus Felület**:
  * Élő dual kamera feed SVG overlay elemekkel (detektálási dobozok, középpontok, 2D/3D vektorok, FPS számlálók).
  * **GoalView Widget**: A kapu 2D síkú valós idejű grid vizualizációja becsapódási pontokkal, konfidencia-zónákkal és lövési előzményekkel.
  * **ActuatorControlWidget**: Aktuátor/szervo tesztelő és vezérlő panel manuális felülbírálással, preset pozíciókkal és E-STOP (vészleállító) funkcióval.
  * **AnalyticsView Dashboard**: Valós idejű grafikonok (sebesség, magasság, Z-mélység), 2D lövési hőtérkép (Heatmap), szektoros eloszlási mutatók és CSV/HTML riport exportálás.
  * **CalibrationDialog**: Interaktív, többlépcsős sztereó kalibrációs varázsló élő sakktábla sarokdetektálási visszajelzéssel.
  * **SplashScreen**: Indítási hardver- és környezeti diagnosztikai ellenőrzés (Health Check).
* **Rugalmas futtatási módok**: GUI mód, Mock/Webcam fejlesztési mód (`--mock`), és korlátozott erőforrású vagy szerver környezetekhez tervezett Headless mód (`--no-gui`).

---

### 🛠 Hardver és Geometriai Specifikációk

| Komponens / Paraméter | Érték / Leírás |
|---|---|
| **Kamerák** | 2× Ximea MC023CG-SY-UB (USB 3.0) |
| **Szenzor** | Sony IMX174 CMOS, Global Shutter, 2.3 MP |
| **Felbontás & FPS** | Natív: 1936 × 1216 @ 100 FPS (max 165 FPS) |
| **Lencsék** | Fujifilm CF8ZA-1S (8 mm, C-Mount, f/1.8 – f/4.0) |
| **Pixeltávolság / Fókusz** | 5.86 µm pixelméret, $f_{px} \approx 1365.2 \text{ px}$ |
| **Kamera Pozíciók** | Bal: $X = -1070\text{ mm}$, Jobb: $X = +1070\text{ mm}$, Magasság: $Y = 2900\text{ mm}$ |
| **Baseline (Kameratáv)** | $\approx 2140\text{ mm}$ – $2238\text{ mm}$ |
| **Kapu Mérete** | $4000\text{ mm} \times 2000\text{ mm}$ ($X \in [-2000, 2000]$, $Y \in [0, 2000]$) |
| **Lövő Távolság** | $Z = 8000\text{ mm}$ (8 méter a kapu síkjától) |
| **Labda** | 4-es / 5-ös méretű focilabda ($\approx 210-220\text{ mm}$ átmérő, $\approx 340-430\text{ g}$) |
| **Inferencia GPU** | NVIDIA GeForce RTX 3050 6GB (TensorRT CUDA) |

---

### 📂 Projekt Struktúra

```
DEIK-ROBOTGOALKEEPER-PROJECT-2026/
├── README.md                          ← Rendszer dokumentáció (Ez a fájl)
├── requirements.txt                   ← Python függőségek listája
├── setup.sh                           ← Automatikus környezeti telepítő szkript
├── run.sh                             ← Rendszerindító szkript
├── config/
│   └── config.yaml                    ← Fő konfigurációs fájl ⚙️
├── src/
│   ├── main.py                        ← Belépési pont (GUI & Headless) 🚀
│   ├── camera/                        ← Kamera kezelő modulok
│   │   ├── base_camera.py             ← Absztrakt kamera interfész
│   │   ├── ximea_camera.py            ← Ximea xiAPI wrapper & sávszélesség kezelő
│   │   ├── mock_camera.py             ← Webcam / videó / szintetikus teszt kamera
│   │   ├── camera_manager.py          ← Dual kamera szinkronizáló és koordinátor
│   │   └── camera_utils.py            ← Kamera segédfüggvények
│   ├── detection/                     ← Detektálás és 2D követés
│   │   ├── ball_detector.py           ← YOLO / RT-DETR TensorRT detektor
│   │   ├── kalman_tracker.py          ← 2D Kalman-szűrő (per-kamera)
│   │   └── optical_flow_tracker.py    ← Lucas-Kanade optikai folyam követő
│   ├── stereo/                        ← 3D Sztereó látórendszer
│   │   ├── triangulator.py            ← 3D sztereó háromszögelés
│   │   └── mono_depth_estimator.py    ← Egykamerás mélységbecslő fallback
│   ├── calibration/                   ← Kalibráció segédmodulok
│   │   └── alignment_helper.py        ← Sakktábla pozíció és dőlésigazító
│   ├── prediction/                    ← Fizikai pálya-előrejelzés
│   │   └── trajectory_predictor.py    ← 3D Kalman + aerodinamikai fizikai modell
│   └── gui/                           ← PyQt6 Grafikus Felület
│       ├── main_window.py             ← Fő ablak és vezérlő logika
│       ├── goal_view.py               ← Kapu 2D grid vizualizátor widget
│       ├── actuator_widget.py         ← Aktuátor szervo tesztelő & E-STOP panel
│       ├── analytics_view.py          ← Grafikonok, 2D hőtérkép és CSV/HTML riport
│       ├── calibration_dialog.py      ← Sztereó kalibrációs varázsló ablak
│       ├── splash_screen.py           ← Indító ablak Health Check diagnosztikával
│       └── theme.py                   ← Sötét / Világos témakezelő
├── scripts/                           ← Diagnosztikai és kalibrációs szkriptek
│   ├── calibrate_stereo.py            ← Interaktív parancssori sztereó kalibráló
│   ├── test_cameras.py                ← Kamera kapcsolat és FPS teszt
│   ├── download_model.py              ← AI modell letöltő
│   └── setup_usbfs_memory.sh          ← Linux USBFS memórialimit beállító (2048 MB)
├── data/
│   ├── calibration/                   ← Kalibrációs fájlok (stereo_calibration.npz)
│   └── recordings/                    ← Mentett videók és felvételek
└── tests/                             ← Pytest unit tesztek
    ├── test_trajectory.py             ← Pálya-előrejelzés tesztek
    ├── test_mono_depth.py             ← Monokuláris mélységtesztek
    ├── test_kalman_tracker.py         ← 2D Kalman-szűrő tesztek
    ├── test_optical_flow.py           ← Optikai folyam tesztek
    ├── test_orange_ball_filter.py     ← Narancssárga szín szűrő tesztek
    └── test_alignment_helper.py       ← Kalibrációs igazítási tesztek
```

---

### 💻 Telepítés és Beállítás

#### 1. Automatikus telepítés (Ajánlott)
```bash
git clone <repository_url>
cd DEIK-ROBOTGOALKEEPER-PROJECT-2026
chmod +x setup.sh scripts/setup_usbfs_memory.sh
./setup.sh
```

#### 2. Linux USBFS memóriabuffer növelése (Ximea dual-camera használatához kötelező!)
A két nagy sebességű Ximea kamera egyidejű működtetéséhez növelni kell az USB lefoglalt memóriát:
```bash
sudo ./scripts/setup_usbfs_memory.sh
```

#### 3. Kézi telepítés
```bash
# Virtuális környezet létrehozása
python3 -m venv venv
source venv/bin/activate

# PyTorch telepítése CUDA támogatással (RTX GPU-hoz)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Függőségek telepítése
pip install -r requirements.txt

# YOLO / TensorRT modell letöltése
python scripts/download_model.py
```

#### 4. Ximea Linux SDK telepítése
1. Töltsd le a [Ximea Linux Software Package](https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package) csomagot.
2. Telepítés:
   ```bash
   tar -xzf ximea_linux_sp.tgz
   cd package
   sudo ./install
   ```
3. Python API telepítése a virtuális környezetbe:
   ```bash
   cd /opt/XIMEA/api/Python/v3
   python3 setup.py install
   ```

---

### 🚀 Indítás és Használat

#### Normál üzemmód (Valódi Ximea kamerákkal + Full GUI)
```bash
source venv/bin/activate
python src/main.py
```
*vagy az indítószkripttel:*
```bash
./run.sh
```

#### Fejlesztői / Teszt üzemmód (Mock/Webcam kamera)
Ha nincsenek csatlakoztatva a Ximea kamerák:
```bash
python src/main.py --mock
```

#### Fejléc nélküli (Headless) üzemmód
Szerveres futtatáshoz vagy GUI nélküli mérésekhez:
```bash
python src/main.py --mock --no-gui
```

#### Egyedi konfigurációs fájl és naplózási szint megadása
```bash
python src/main.py --config config/custom_config.yaml --log-level DEBUG
```

---

### 🎯 Sztereó Kalibrálás

A 3D háromszögelés pontosságához a kamerákat első használat előtt kalibrálni kell.

#### Kalibráció menete a Grafikus Felületen (GUI):
1. Indítsd el az alkalmazást: `python src/main.py`
2. Készíts elő egy **70 mm-es négyzetméretű**, **12×9 mezős** (11×8 belső sarok) sakktáblát.
3. Kattints a menüben a **Sztereó Kalibráció** gombra.
4. Gyűjts össze legalább 20-30 éles képpárt különböző távolságokból (0.5m – 4m) és szögekből.
5. Futtasd a kalibrációt a varázslóban. A rendszer automatikusan elmenti az eredményt a `data/calibration/stereo_calibration.npz` fájlba.

#### Kalibráció parancssorból:
```bash
python scripts/calibrate_stereo.py
```
* **Billentyűk**: `SPACE` = Képpár rögzítése, `c` = Kalibráció futtatása, `q` = Kilépés.
* **Cél RMSE érték**: $< 1.0 \text{ px}$ (elfogadható), $< 0.5 \text{ px}$ (kiváló).

---

### ⚙️ Konfiguráció (`config/config.yaml`)

A főbb paraméterek közvetlenül a `config/config.yaml` fájlban módosíthatók:
* `camera.type`: `"ximea"` vagy `"mock"`
* `camera.fps`: Célzott képfrissítés (pl. `100`)
* `camera.exposure_time_us`: Zársebesség ($\mu s$, pl. `3000`)
* `detection.model_path`: A TensorRT/YOLO modell fájl elérési útja (`models/yolov8n.engine`)
* `detection.confidence_threshold`: Detektálási küszöb (alapértelmezett: `0.15`)
* `prediction.drag_coefficient`: Légellenállási tényező (alapértelmezett: `0.0005`)
* `geometry.baseline_mm`: Kamerák közötti fizikai távolság mm-ben (alapértelmezett: `2140.0`)

---

### 🧪 Diagnosztika és Tesztelés

```bash
# Kamera kapcsolat, sorozatszámok és FPS tesztelése
python scripts/test_cameras.py

# Mock kamerateszt képablak megjelenítésével
python scripts/test_cameras.py --mock --show-frames

# Unit tesztek futtatása (pytest)
pytest tests/ -v
```

---

<br/>

---

## 🇬🇧 English Documentation

### 📌 Project Overview

The **DEIK Robot Goalkeeper** project is a high-speed real-time optical ball detection, 3D tracking, and trajectory prediction system designed to control a robotic goalkeeper mechanism. Utilizing two industrial **Ximea MC023CG-SY-UB** stereo cameras, the system tracks a football in 3D space at 100+ FPS and predicts its precise impact coordinates, time-to-impact, and goal probability on the goal plane ($Z=0$).

#### Key Features & Capabilities:
* **Dual Ximea Camera Acquisition**: Synchronized capture using industrial Sony IMX174 Global Shutter sensors (2.3 MP, up to 165 FPS), custom USB3 bandwidth limiters, and `usbfs_memory` allocation.
* **TensorRT GPU Acceleration**: Optimized AI inference (`.engine` models for YOLOv8n / YOLOv10n / RT-DETR) running batch=2 processing on NVIDIA CUDA hardware at >100 FPS.
* **Color & Motion Validation**: Integrated HSV orange color filter (`OrangeBallFilter`) tuned for patterned footballs (e.g. Kipsta size 4/5) and Lucas-Kanade Optical Flow tracker (`OpticalFlowTracker`).
* **Multi-Stage Kalman Filtering**: Dual 2D Kalman filters (`KalmanTracker`) for per-camera noise reduction and a 3D spatial Kalman filter for smooth trajectory estimations.
* **Stereo 3D Triangulation & Monocular Depth Fallback**: Millimeter-accurate 3D coordinate calculation via stereo calibration matrices, with single-camera depth estimation (`MonoDepthEstimator`) when object visibility is degraded.
* **Physics-Based Trajectory Prediction**: Real-time parabolic motion solver including gravitational acceleration ($g=9.81 \text{ m/s}^2$) and quadratic aerodynamic drag ($C_d = 0.47$).
* **PyQt6 Visualization Suite**:
  * Dual live camera feeds with SVG HUD overlays (bounding boxes, centroids, 2D/3D vectors, real-time FPS).
  * **GoalView Widget**: Interactive 2D goal plane grid showing projected impact points, confidence rings, and shot history.
  * **ActuatorControlWidget**: Hardware actuator/servo control and test panel featuring manual overrides, preset target positions, and emergency stop (E-STOP).
  * **AnalyticsView Dashboard**: Real-time performance analytics (ball speed, altitude, depth profile), 2D shot heatmap, goal sector breakdown, and CSV/HTML report export.
  * **CalibrationDialog**: Interactive step-by-step stereo calibration wizard with live chessboard feedback.
  * **SplashScreen**: Automated startup hardware & system health diagnostic check.
* **Versatile Execution Modes**: Full PyQt6 GUI mode, Mock/Webcam development mode (`--mock`), and headless server execution (`--no-gui`).

---

### 🛠 Hardware & Geometry Specifications

| Parameter / Component | Value / Description |
|---|---|
| **Cameras** | 2× Ximea MC023CG-SY-UB (USB 3.0) |
| **Sensor** | Sony IMX174 CMOS, Global Shutter, 2.3 MP |
| **Resolution & FPS** | Native: 1936 × 1216 @ 100 FPS (up to 165 FPS max) |
| **Lenses** | Fujifilm CF8ZA-1S (8 mm, C-Mount, f/1.8 – f/4.0) |
| **Pixel Pitch / Focal** | 5.86 µm pixel size, $f_{px} \approx 1365.2 \text{ px}$ |
| **Camera Positions** | Left: $X = -1070\text{ mm}$, Right: $X = +1070\text{ mm}$, Height: $Y = 2900\text{ mm}$ |
| **Baseline Distance** | $\approx 2140\text{ mm}$ – $2238\text{ mm}$ |
| **Goal Dimensions** | $4000\text{ mm} \times 2000\text{ mm}$ ($X \in [-2000, 2000]$, $Y \in [0, 2000]$) |
| **Shooting Distance** | $Z = 8000\text{ mm}$ (8 meters from goal line) |
| **Football Target** | Size 4 / Size 5 ball ($\approx 210-220\text{ mm}$ diameter, $\approx 340-430\text{ g}$) |
| **Inference Hardware** | NVIDIA GeForce RTX 3050 6GB (TensorRT CUDA) |

---

### 💻 Quick Start & Execution

```bash
# Clone repository
git clone <repository_url>
cd DEIK-ROBOTGOALKEEPER-PROJECT-2026

# Automatic setup
chmod +x setup.sh scripts/setup_usbfs_memory.sh
./setup.sh

# Increase USB memory buffer for Ximea dual cameras (Required)
sudo ./scripts/setup_usbfs_memory.sh

# Run application with Ximea cameras
python src/main.py

# Run in Mock mode (webcam / test video)
python src/main.py --mock

# Run in Headless mode (no GUI)
python src/main.py --mock --no-gui
```

---

### 🧪 Testing & Diagnostics

```bash
# Diagnostic test for Ximea cameras and frame rate
python scripts/test_cameras.py

# Run automated test suite
pytest tests/ -v
```

---

<br/>

---

## 👥 Szerzők & Készítők / Authors

### **Debreceni Egyetem – Informatikai Kar (DEIK)**
**DEIK Robot Foci Kapus Projekt 2026 / DEIK Robot Goalkeeper Project 2026**

* 👨‍💻 **Morvai Roland** – *BSc Mérnökinformatikus* (DEIK)
* 👨‍💻 **Rácz Donát** – *BSc Mérnökinformatikus* (DEIK)

---
*Copyright © 2026 DEIK Robot Goalkeeper Team. All Rights Reserved.*
