#!/usr/bin/env bash
# =============================================================================
# DEIK Robot Foci Kapus – Automatikus Telepítő Szkript
# =============================================================================
# Futtatás: chmod +x setup.sh && ./setup.sh
#
# Ez a szkript elvégzi:
#   1. Python virtuális környezet létrehozása
#   2. Alap Python csomagok telepítése
#   3. PyTorch CUDA 12.1 telepítése (RTX 3050-hez)
#   4. YOLOv10n modell letöltése
#   5. Könyvtárstruktúra ellenőrzése
# =============================================================================

set -e  # Ha bármely parancs hibával zárul, leállunk

# --- Színkódok a terminálkimenethez ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DEIK Robot Foci Kapus – Projekt Telepítő           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# --- 1. Virtuális környezet ---
echo -e "${YELLOW}[1/5] Python virtuális környezet létrehozása...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtuális környezet létrehozva: ./venv${NC}"
else
    echo -e "${GREEN}✓ Virtuális környezet már létezik${NC}"
fi

# Aktiválás
source venv/bin/activate
echo -e "${GREEN}✓ Virtuális környezet aktiválva${NC}"

# --- 2. Pip frissítése ---
echo -e "${YELLOW}[2/5] pip frissítése...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip naprakész${NC}"

# --- 3. Alap csomagok telepítése ---
echo -e "${YELLOW}[3/5] Alap Python csomagok telepítése...${NC}"
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Alap csomagok telepítve${NC}"

# --- 4. PyTorch CUDA telepítése (RTX 3050, CUDA 12.1) ---
echo -e "${YELLOW}[4/5] PyTorch CUDA 12.1 telepítése (RTX 3050)...${NC}"
echo -e "${BLUE}   Ez néhány percig tarthat (kb. 2.5 GB letöltés)...${NC}"
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
echo -e "${GREEN}✓ PyTorch CUDA telepítve${NC}"

# --- 5. YOLOv10n modell letöltése ---
echo -e "${YELLOW}[5/5] YOLOv10n modell letöltése...${NC}"
if [ ! -f "models/yolov10n.pt" ]; then
    mkdir -p models
    python3 scripts/download_model.py
    echo -e "${GREEN}✓ Modell letöltve: models/yolov10n.pt${NC}"
else
    echo -e "${GREEN}✓ Modell már megvan: models/yolov10n.pt${NC}"
fi

# --- Könyvtárstruktúra létrehozása ---
echo -e "${YELLOW}Adatkönyvtárak létrehozása...${NC}"
mkdir -p data/calibration
mkdir -p data/recordings
mkdir -p models
echo -e "${GREEN}✓ Könyvtárstruktúra kész${NC}"

# --- CUDA ellenőrzés ---
echo ""
echo -e "${YELLOW}GPU ellenőrzés:${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  ✓ CUDA elérhető: {name} ({mem:.1f} GB VRAM)')
else:
    print('  ⚠ CUDA nem elérhető – CPU módban fog futni')
"

# --- Ximea SDK ellenőrzés ---
echo -e "${YELLOW}Ximea SDK ellenőrzés:${NC}"
python3 -c "
try:
    from ximea import xiapi
    print('  ✓ Ximea xiAPI elérhető')
except ImportError:
    print('  ⚠ Ximea xiAPI nem található – mock kamera módban futtatható')
    print('    Telepítés: https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package')
"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Telepítés kész! Indítás:                           ║${NC}"
echo -e "${GREEN}║                                                      ║${NC}"
echo -e "${GREEN}║   source venv/bin/activate                           ║${NC}"
echo -e "${GREEN}║   python src/main.py                                 ║${NC}"
echo -e "${GREEN}║                                                      ║${NC}"
echo -e "${GREEN}║   Kamera teszt:                                      ║${NC}"
echo -e "${GREEN}║   python scripts/test_cameras.py                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
