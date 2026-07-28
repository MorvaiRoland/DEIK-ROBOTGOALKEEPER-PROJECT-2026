# DEIK Robot Foci Kapus – Rendszer Dokumentáció

## Projektleírás

Valós idejű focilabda detektáló és trajektória előrejelző rendszer robot foci kapushoz.
Két Ximea MC023CG-SY-UB sztereó kamerával 3D-ben követi a labda röppályáját és meghatározza a kapu síkján a becsapódási pontot.

---

## Hardver Specifikációk

| Komponens | Specifikáció |
|---|---|
| Kamera | Ximea MC023CG-SY-UB × 2 |
| Szenzor | Sony IMX174, Global Shutter, 2.3 MP |
| Max FPS | 165 FPS (1936 × 1216) |
| Lencse | Fujifilm CF8ZA-1S, 8mm, f/1.8, C-Mount |
| Fókusztávolság (px) | ~1365 px |
| Csatlakozás | EP-USB3HybridcableU-20 (20m hybrid kábel) |
| GPU | NVIDIA RTX 3050 6GB |
| Kamera elrendezés | Bal: X=-2450mm, Jobb: X=+2450mm, Y=2800mm |
| Baseline | 4900 mm |
| Kapu mérete | 4000 mm × 2000 mm |
| Lövőtávolság | 8000 mm |
| Labda | 5-ös méret (~220mm átmérő) |

---

## Projekt Struktúra

```
DEIK-ROBOTGOALKEEPER-PROJECT-2026/
├── README.md                          ← Ez a fájl
├── requirements.txt                   ← Python függőségek
├── setup.sh                           ← Automatikus telepítő
├── .gitignore
├── config/
│   └── config.yaml                    ← Összes rendszerparaméter ⚙️
├── src/
│   ├── main.py                        ← Belépési pont 🚀
│   ├── camera/
│   │   ├── base_camera.py             ← Absztrakt kamera interfész
│   │   ├── ximea_camera.py            ← Ximea MC023CG-SY-UB wrapper
│   │   ├── mock_camera.py             ← Webcam/videó teszt kamera
│   │   └── camera_manager.py          ← Dual camera koordinátor
│   ├── detection/
│   │   ├── ball_detector.py           ← YOLOv10n + ByteTrack detektor
│   │   └── kalman_tracker.py          ← 2D Kalman szűrő (per-kamera)
│   ├── stereo/
│   │   ├── calibration.py             ← [jövőbeni] Kalibrálás manager
│   │   └── triangulator.py            ← Sztereó 3D háromszögelés
│   ├── prediction/
│   │   └── trajectory_predictor.py    ← Fizika-alapú pálya előrejelzés
│   └── gui/
│       ├── main_window.py             ← PyQt6 főablak
│       └── goal_view.py               ← Kapu vizualizátor widget
├── scripts/
│   ├── calibrate_stereo.py            ← Sztereó kalibrálás (interaktív)
│   ├── test_cameras.py                ← Kamera diagnosztika
│   └── download_model.py              ← YOLOv10n modell letöltő
├── models/
│   └── yolov10n.pt                    ← [letöltendő] YOLO modell
├── data/
│   ├── calibration/                   ← Kalibrálási adatok (.npz)
│   └── recordings/                    ← DVR felvételek
└── tests/
    └── test_trajectory.py             ← Unit tesztek
```

---

## Telepítés

### 1. Alaptelepítés
```bash
git clone <repo_url>
cd DEIK-ROBOTGOALKEEPER-PROJECT-2026
chmod +x setup.sh
./setup.sh
```

### 2. Kézi telepítés (ha setup.sh nem fut)
```bash
python3 -m venv venv
source venv/bin/activate

# PyTorch CUDA (RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Többi csomag
pip install -r requirements.txt

# Modell letöltés
python scripts/download_model.py
```

### 3. Ximea SDK telepítése
```bash
# Letöltés: https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package
tar -xzf ximea_linux_sp.tgz
cd package
sudo ./install

# Python API
cd /opt/XIMEA/api/Python/v3
sudo python3 setup.py install
```

---

## Indítás

### Normál mód (Ximea kamerákkal)
```bash
source venv/bin/activate
python src/main.py
```

### Teszt mód (webcam/szintetikus kép)
```bash
python src/main.py --mock
```

### Headless mód (GUI nélkül)
```bash
python src/main.py --mock --no-gui
```

---

## Kalibrálás (KÖTELEZŐ a valódi kamerákhoz!)

A rendszer első indítása előtt el kell végezni a sztereó kalibrálást:

### Szükséges eszközök
- Nyomtatott sakktábla: **9 × 6 belső sarok**, **30 mm négyzetméret**
- Jó, egyenletes megvilágítás

### Kalibrálás menete
```bash
python scripts/calibrate_stereo.py
```

**Vezérlők:**
- `SPACE` → Képpár mentése (ha mindkét kamerában látja a sakktáblát)
- `c` → Kalibrálás futtatása (min. 20 képpár után)
- `q` → Kilépés

**Tippek:**
- Tarts 20-30 különböző szöget és távolságot (0.5m – 3m)
- RMSE < 1.0 px = jó kalibrálás; < 0.5 px = kiváló

---

## Modell: YOLOv10n

### Miért YOLOv10n?
| Tulajdonság | Érték |
|---|---|
| Architektúra | CNN, anchor-free, NMS-mentes |
| Sebesség (RTX 3050, 1280px) | ~200 FPS |
| Sebesség (CPU only) | ~55 FPS |
| mAP50 (COCO) | 38.5 |
| Modell méret | ~7 MB |

### Custom modell tanítás
Ha a COCO pre-trained modell nem elég pontos, saját adatokkal lehet tanítani:

1. Képgyűjtés: `python scripts/collect_training_data.py` *(hamarosan)*
2. Annotálás: [Roboflow](https://roboflow.com) online eszközzel
3. Tanítás:
   ```bash
   yolo train model=yolov10n.pt data=data.yaml epochs=100 imgsz=1280
   ```
4. Frissítés a `config/config.yaml`-ban:
   ```yaml
   detection:
     model_path: "models/custom_soccer_ball.pt"
     ball_class_id: 0   # Custom modellnél 0
   ```

---

## Fizikai modell

### Koordináta-rendszer
```
         Y (felfelé)
         ↑
         │    Z (pálya felé)
         │   ↗
         │  /
─────────│──────────── X (jobbra)
     Kapu közép
```

### Trajektória előrejelzés
A rendszer a következő erőket veszi figyelembe:
- **Gravitáció**: g = 9810 mm/s² (lefelé)
- **Légellenállás**: F_d = ½ × ρ × C_d × A × v² (mozgással ellentétes)

Labda paraméterei:
- Átmérő: 220 mm, Tömeg: 430 g
- C_d ≈ 0.47 (gömb)

---

## Diagnosztika

```bash
# Kamera kapcsolat és FPS teszt
python scripts/test_cameras.py

# Mock kamerával (Ximea nélkül)
python scripts/test_cameras.py --mock --show-frames

# Unit tesztek
python -m pytest tests/ -v
```

---

## Konfiguráció

A `config/config.yaml` fájl tartalmazza az összes beállítást.
Legfontosabb paraméterek:

```yaml
camera:
  type: "ximea"           # "ximea" | "mock"
  fps: 100                # Célzott frame rate
  exposure_time_us: 3000  # Zársebesség

detection:
  model_path: "models/yolov10n.pt"
  confidence_threshold: 0.4
  device: "cuda:0"        # RTX 3050 GPU

geometry:
  baseline_mm: 4900.0     # Két kamera távolsága
  focal_length_px: 1365.2 # Fujifilm CF8ZA-1S becsült érték
```

---

## Fejlesztési terv

- [x] Kamera réteg (Ximea + Mock)
- [x] YOLOv10n detektálás + ByteTrack
- [x] Sztereó háromszögelés
- [x] Fizika-alapú trajektória előrejelzés
- [x] PyQt6 GUI
- [x] Sztereó kalibrálás szkript
- [ ] Custom YOLO modell tanítás (saját labda adatokon)
- [ ] TensorRT export (tovább növeli a GPU sebességet)
- [ ] Robot kapus kommunikáció (mikor a hardware elérhető)

---

## Szerzők

**Debreceni Egyetem, Informatikai Kar**  
DEIK Robot Foci Kapus Projekt 2026
