# 📋 DEIK Robot Foci Kapus – Projekt Összefoglaló és Rendszerspecifikáció
**Céges megbeszélésre / Partner találkozóra előkészített szakmai dokumentum**

---

### ℹ️ Dokumentum Információk
* **Projekt neve**: DEIK Robot Foci Kapus (DEIK Robot Goalkeeper Project 2026)
* **Intézmény**: Debreceni Egyetem – Informatikai Kar (DEIK)
* **Fejlesztők / Szerzők**: Morvai Roland (*BSc Mérnökinformatikus*), Rácz Donát (*BSc Mérnökinformatikus*)
* **Dátum**: 2026. szeptember 8.
* **Verzió**: v1.3 (Aktív szoftvermodulok, hardveres szinkronizáció és megvilágítás fejlesztési fázisban)

---

## 1. 🎯 Végrehajtói Összefoglaló (Executive Summary)

A **DEIK Robot Foci Kapus** egy valós idejű, nagy sebességű optikai 3D labdakövető, trajektória-előrejelző és robotkapus-aktuátor vezérlő szoftverrendszer. A rendszer célja, hogy a pályáról rálőtt nagy sebességű labdákat (akár 80–100 km/h) a repülés első töredékmásodpercében (0.1–0.3 másodperc alatt) 3D térben beazonosítsa, kiszámítsa a labda fizikai repülési pályáját, és megadja a kapu síkjában ($Z=0$) a várható becsapódási koordinátákat ($X, Y$), a becsapódás idejét ($t_{\text{impact}}$), valamint a kapu eltalálásának valószínűségét.

### Jelenlegi Projektstátusz Rövid Áttekintése:
* **Elkészült Szoftveres Alapok**: A teljes szoftveres pipeline készen áll: TensorRT GPU detektálás, 3D sztereó háromszögelés, aerodinamikai pálya-előrejelzés és a komplett PyQt6 grafikus felület az aktuátor tesztelő panellel és E-STOP-pal.
* **Jelenlegi Fejlesztési Fókusz (Kihívások)**:
  1. **Hardveres Képszinkronizáció**: A kamerák hardveres GPIO triggerelése még nem működik stabilan, ennek beállításán és hibaelhárításán dolgozunk. Hardveres szinkron hiányában a kamerák közötti fáziseltérés miatt a 3D háromszögelés jelenleg eltolódott / pontatlan értékekkel számol.
  2. **Megvilágítás / Fényviszonyok**: Várjuk a dedikált lámpákat a tesztkörnyezethez. Megfelelő világítás hiányában a képzaj és az alulexponáltság miatt az optikai AI detektáció néha bizonytalan.
* **Következő Mérföldkő**: A szinkronizáció és világítás rendezése után következik a fizikai robotkapus mechanizmus és aktuátorok bekötése, valamint a vezérlés élesítése.

---

## 2. 🛠 Hardver és Geometriai Specifikációk

A rendszer a `config/config.yaml` fájlban rögzített fizikai parametrizáció alapján működik:

| Komponens / Paraméter | Hardveres & Geometriai Specifikáció | Szakmai Megjegyzés & Aktuális Státusz |
|---|---|---|
| **Sztereó Kamerák** | **2× XIMEA MC023CG-SY-UB** (USB 3.0) | Ipari kamerák robusztus alumínium házban |
| **Képérzékelő Szenzor** | **Sony IMX174**, Global Shutter, 2.3 MP | Nincs mozgási torzulás (Rolling Shutter artifact mentes) |
| **Felbontás & Képfrissítés**| Natív 1936 × 1216 pixel; Target: **100 FPS** (Max 165 FPS) | 10ms képkocka-időköz |
| **Optika / Lencse** | **2× Fujifilm CF8ZA-1S** (8mm, f/1.8 – f/4.0, C-Mount) | Ipari mélységélesség és alacsony torzítás ($f_{px} \approx 1365.2\text{ px}$) |
| **Adatkábelek** | **2× EP-USB3HybridcableU-20** (20m USB3 hibrid optikai) | Hosszú távú stabil adatátvitel veszteség nélkül |
| **Hardveres Szinkronizáció**| **CBL-702-8P-SYNC-5M0** (8-pin GPIO kábel) | ⚠️ **Fejlesztés alatt**: A trigger jel még nem működik stabilan (jelenleg szoftveres szinkron fut) |
| **Számítási Kapacitás** | **NVIDIA GeForce RTX 3050 6GB GPU** | CUDA & TensorRT `.engine` GPU acceleráció |
| **Baseline (Kameratáv)** | **2369.4 mm** | Sztereó kalibrációval kimért fizikai távolság |
| **Kamera Pozíciók** | Bal: $X = -1070\text{ mm}$, Jobb: $X = +1070\text{ mm}$ | Magasság: $Y = 2570\text{ mm}$, Z-offset: $Z = 583\text{ mm}$ (gólvonal mögött) |
| **Kamera Dőlésszög** | **43° Pitch** (lefelé döntve) | Lefedettség a kapu és a 10m-es lövőzóna között |
| **Kapu Mérete** | **4000 mm × 2000 mm** (Szélesség × Magasság) | Standard kapus tesztkeret ($X \in [-2000, +2000]$, $Y \in [0, 2000]$) |
| **Lövőtávolság** | **10000 mm (10 méter)** | A lövő pont távolsága a kapu síkjától |
| **Célobjektum / Labda** | **Kipsta 4-es / 5-ös méretű focilabda** | 4-es: 210 mm átmérő (340g), 5-ös: 220 mm átmérő (430g) |

---

## 3. 💻 Technológiai Stakk és Szoftveres Architektúra

### Technológiai Összetevők:
* **Programozási Nyelv**: Python 3.10+ / Python 3.12
* **Gépi Látás & AI**: OpenCV 4.x, PyTorch CUDA, NVIDIA TensorRT (YOLOv8n / YOLOv10n / RT-DETR modellek batch=2 inferenciával)
* **Kamera SDK**: Ximea `xiAPI` Python 3 kiterjesztés + Linux USBFS kernel buffer kezelés (2048 MB lefoglalás)
* **Szűrők és Követés**:
  * `OrangeBallFilter`: HSV színtartomány alapú narancssárga színvalidáció a mintás Kipsta labdák téves detektálásának kiszűrésére.
  * `OpticalFlowTracker`: Lucas-Kanade optikai folyam a 2D mozgásvektorok azonnali frissítésére két AI detektálás között.
  * `KalmanTracker`: Kameránkénti 2D Kalman-szűrő a mérések zajmentesítésére.
  * `Spatial3DKalman`: 3D térbeli Kalman-szűrő a háromszögelt 3D pontsorozat simítására.
* **3D Látórendszer & Fizikai Trajektória**:
  * Sztereó háromszögelés intrinsic ($K_1, K_2$) és extrinsic ($R, T$) kalibrációs mátrixokkal.
  * `MonoDepthEstimator`: Egykamerás fallback mélységbecslő a labda ismert fizikai átmérője alapján.
  * Aerodinamikai Parabolikus Modell: Gravitáció ($g=9.81\text{ m/s}^2$) és kvadratikus légellenállás ($C_d = 0.47, \rho = 1.225\text{ kg/m}^3$) differenciálegyenletének valós idejű megoldása.
* **Grafikus Felület (GUI)**: PyQt6 modularizált architektúra, sötét/világos témakezelővel (`theme.py`), élő HUD overlay-jel.
* **Tesztelési Keretrendszer**: Pytest automatizált tesztcsomag (27/27 sikeres teszteset).

---

## 4. 📂 Projekt Struktúra

```
DEIK-ROBOTGOALKEEPER-PROJECT-2026/
├── README.md                          ← Rendszer dokumentáció (Magyar és Angol)
├── PROJECT_SUMMARY_MEETING.md         ← Ez a céges tárgyalási dokumentum
├── config/
│   └── config.yaml                    ← Rendszerparaméterek és beállítások ⚙️
├── src/
│   ├── main.py                        ← Alkalmazás belépési pont (GUI & Headless) 🚀
│   ├── camera/                        ← Kamera alrendszer
│   │   ├── base_camera.py             ← Absztrakt kamera interfész
│   │   ├── ximea_camera.py            ← Ximea xiAPI wrapper, bandwidth & GPIO szinkronizációs logika
│   │   ├── mock_camera.py             ← Webcam / szintetikus videó tesztkamera
│   │   ├── camera_manager.py          ← Dual kamera szinkronizáció és vezérlő
│   │   └── camera_utils.py            ← Linux USBFS memóriakezelő segédfüggvények
│   ├── detection/                     ← Detektálás és 2D követés
│   │   ├── ball_detector.py           ← YOLO / TensorRT GPU detektor + OrangeBallFilter
│   │   ├── kalman_tracker.py          ← 2D Kalman-szűrő (per-kamera)
│   │   └── optical_flow_tracker.py    ← Lucas-Kanade optikai folyam követő
│   ├── stereo/                        ← 3D Sztereó látórendszer
│   │   ├── triangulator.py            ← 3D sztereó háromszögelő modul
│   │   └── mono_depth_estimator.py    ← Monokuláris mélységbecslő fallback
│   ├── calibration/                   ← Sztereó kalibráció és beállítás
│   │   └── alignment_helper.py        ← Kamera fizikai dőlés- és pozíció igazító
│   ├── prediction/                    ← Fizikai pálya-előrejelző engine
│   │   └── trajectory_predictor.py    ← 3D Kalman + aerodinamikai fizikai modell
│   └── gui/                           ← PyQt6 Grafikus Felület
│       ├── main_window.py             ← Fő ablak és vezérlő logika
│       ├── goal_view.py               ← Kapu 2D grid vizualizáció és becsapódási pontok
│       ├── actuator_widget.py         ← Aktuátor szervo tesztelő & E-STOP panel
│       ├── analytics_view.py          ← Grafikonok, 2D hőtérkép és CSV/HTML exportálás
│       ├── calibration_dialog.py      ← Interaktív sztereó kalibrációs varázsló
│       ├── splash_screen.py           ← Indítási hardver diagnosztikai ellenőrző (Health Check)
│       └── theme.py                   ← Sötét / Világos témakezelő
├── scripts/                           ← Segédszkriptek
│   ├── calibrate_stereo.py            ← Parancssori sztereó kalibráló
│   ├── test_cameras.py                ← Kamera kapcsolat és FPS diagnosztikai teszt
│   ├── download_model.py              ← TensorRT / YOLO model letöltő
│   └── setup_usbfs_memory.sh          ← Linux USBFS memórialimit (2048 MB) szkript
├── data/
│   ├── calibration/                   ← Kalibrációs adatok (stereo_calibration.npz)
│   └── recordings/                    ← Mentett tesztvideók és mérések
├── tests/                             ← Pytest unit teszt csomag (27 teszteset)
├── gpio_test_master.py                ← Hardveres GPIO Master trigger hibakereső szkript
├── gpio_test_slave.py                 ← Hardveres GPIO Slave trigger hibakereső szkript
└── setup.sh / run.sh                  ← Telepítő és indító szkriptek
```

---

## 5. 🟢 Mi van készen? (Elért Eredmények és Működő Modulok)

1. **Ipari Kamera Illesztés & USB3 Sávszélesség Optimálás**:
   * A két Ximea MC023CG-SY-UB kamera stabil 100 FPS adatfolyama biztosított a 20 méteres hibrid USB3 kábeleken keresztül.
   * Számítógépes kernel szinten beállításra került a 2048 MB-os USBFS memóriabuffer (`setup_usbfs_memory.sh`).

2. **Kamera Szoftveres Vezérlő & Fallback Architektúra**:
   * Dual-kamera képlekérő és feldolgozó keretrendszer (`CameraManager`).
   * Szoftveres szinkronizáció aktív, amíg a hardveres GPIO trigger beállítása és hibaelhárítása tart.

3. **GPU-Gyorsított AI Detektálás & Narancs Színszűrés**:
   * NVIDIA TensorRT `.engine` inferencia sztereó batch=2 feldolgozással.
   * HSV `OrangeBallFilter` integráció a mintás Kipsta 4/5 focilabda felismeréséhez.

4. **3D Sztereó Háromszögelés & Monokuláris Fallback Algoritmusok**:
   * Sztereó kalibráción alapuló 3D háromszögelő modul.
   * Egykamerás mélységbecslés (`MonoDepthEstimator`) ha a labda kitakarásba kerül.

5. **Aerodinamikai Pálya-előrejelző Engine**:
   * 3D spatial Kalman-szűrővel kombinált fizikai parabolikus + légellenállás modell.
   * Becsült becsapódási hely ($X, Y$), idő ($t_{\text{impact}}$) és gól-valószínűség számítás.

6. **Teljes PyQt6 Felhasználói és Analitikai GUI Suite**:
   * Élő dual kamera nézet HUD overlay-jel.
   * **GoalView Widget**: Kapusík 2D hálózat becsapódási zónákkal.
   * **ActuatorControlWidget**: Aktuátor/szervo vezérlő és tesztelő panel E-STOP (vészleállító) gombbal.
   * **AnalyticsView**: Valós idejű analitikai grafikonok és exportálás.
   * **CalibrationDialog & SplashScreen**: Interaktív sztereó kalibrációs varázsló és diagnosztika.

7. **Szoftveres Teszteltség**:
   * 27 automatizált Pytest unit teszt (100%-os zöld lefutás).

---

## 6. 🟡 Mi van még hátra? (Jelenlegi Kihívások & Ütemterv)

Az éles működéshez a következő lépéseken dolgozunk szigorú egymásutániságban:

```
┌─────────────────────────────────────────────────────────┐
│ 1. Hardveres GPIO Szinkronizáció Hibaelhárítása         │  ← CURRENT FOCUS
│    (Master Out -> Slave Trigger jel bemérése)           │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Megvilágítás / Lámpák Beállítása                     │  ← CURRENT FOCUS
│    (Alulexponáltság és képzaj megszüntetése)            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 3D Kalibráció & Mérések Validálása                   │
│    (Pontos 3D sztereó háromszögelés stabil fénynél)     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Robotkapus Aktuátorok Bekötése & Vezérlése          │
│    (Fizikai motorok csatlakoztatása és tesztelése)      │
└─────────────────────────────────────────────────────────┘
```

### Részletes Feladatleírás:

1. **Hardveres Képszinkronizáció (GPIO Trigger) Beállítása (Jelenlegi Fókusz)**:
   * **Jelenlegi probléma**: A kamerák hardveres trigger jele még nem működik megfelelően. Ezen dolgozunk jelenleg (Master GPO output $\rightarrow$ Slave GPI input).
   * **Hatás a mérésre**: Hardveres szinkron hiányában a kamerák képrögzítése között apró fázis-eltolódás van. Gyorsan mozgó labdánál ez azt eredményezi, hogy a két kamera eltérő időpillanatban látja a labdát, emiatt a 3D háromszögelő algoritmus pontatlan kalibrációs/pozíció értékekkel számol.

2. **Megvilágítás / Lámpák Beállítása (Várakozás a lámpákra)**:
   * **Jelenlegi probléma**: Várjuk a dedikált lámpákat a tesztelési terület megvilágításához.
   * **Hatás a detektációra**: A tesztkörnyezet jelenleg sötét/alulexponált. Alacsony fénynél a kamerákon képzaj keletkezik, a zársebesség lassulhat, ami miatt a gyorsan mozgó labda elmosódik (motion blur), és az AI detektáció néha pontatlanná válik.

3. **3D Mérések és Kalibráció Újra-validálása**:
   * A hardveres trigger működése és a lámpák felszerelése után a sztereó kalibrációt újra lefutatjuk. Ezzel elerjük a mm-pontos, stabil 3D pozícióbecslést.

4. **Robotkapus Bekötése és Aktuátor Vezérlés Élesítése**:
   * Miután a 3D detektálás és szinkronizált mérés stabilan és pontosan működik, ezt követően kötjük be a fizikai robotkapus mozgató mechanizmusát és az aktuátor vezérlőt (CAN bus / RS485 / Serial / UDP).

---

## 7. 🗣 Megbeszélési Pontok a Céges Találkozóra (Discussion Agenda)

A holnapi megbeszélésen a következő kérdések egyeztetése javasolt a partnerrel:

1. **Megvilágítás és Helyszíni Feltételek**:
   * Milyen világítási infrastruktúra áll rendelkezésre a tesztpályán / csarnokban? Kérhető-e dedikált megvilágítás a kapu előtti 10 méteres zónára?

2. **Hardveres Szinkronizáció és Kamera Kábelezés**:
   * Tájékoztatás a Ximea GPIO trigger kábelezés jelenlegi hibaelhárítási állapotáról.

3. **Aktuátor Kommunikációs Protokoll**:
   * Milyen interfészt és protokolt vár el a robotkapus fizikai mozgatását végző motorvezérlő (pl. RS485 / Serial / UDP Ethernet / CAN-bus)?
   * Milyen maximális válaszidő (látencia) fogadható el az aktuátor oldalán?

4. **Fizikai Vészleállító (E-STOP) Integráció**:
   * A GUI-n meglévő E-STOP gomb mellé szükséges-e hardveres kapcsolódás (relé / GPIO kimenet) a fizikai aktuátor tápellátásának megszakítására?

5. **Közös Tesztelési Ütemterv**:
   * Hardveres trigger + lámpák beállítása után mikor indítható a fizikai robotkapus bekötése és az első közös éles lövésteszt?

---

*DEIK Robot Goalkeeper Project 2026 – Debreceni Egyetem Informatikai Kar*
