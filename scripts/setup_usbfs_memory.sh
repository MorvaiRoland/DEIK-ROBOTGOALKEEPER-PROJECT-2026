#!/usr/bin/env bash
# =============================================================================
# DEIK Robot Foci Kapus – Végleges USBFS Memória Konfiguráló Szkript
# =============================================================================
# Ez a szkript beállítja az usbfs_memory_mb=0 értéket véglegesen a Linux
# rendszerben, így minden indításkor automatikusan 0 lesz és a Python kód
# jelszó nélkül is frissítheti.
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  USBFS Memória Automatikus Konfigurálása (Ximea)     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Beállítás az/etc/modprobe.d/usbfs.conf fájlban (Indításkori kernel beállítás)
echo -e "${YELLOW}[1/3] Modprobe konfiguráció létrehozása (/etc/modprobe.d/usbfs.conf)...${NC}"
echo "options usbcore usbfs_memory_mb=0" | sudo tee /etc/modprobe.d/usbfs.conf > /dev/null
echo -e "${GREEN}✓ Modprobe konfiguráció mentve.${NC}"

# 2. Sudoers beállítás jelszómentes állításhoz
echo -e "${YELLOW}[2/3] Sudoers jelszómentes jogosultság beállítása (/etc/sudoers.d/usbfs_memory)...${NC}"
ACTUAL_USER="${SUDO_USER:-$USER}"
echo "$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/module/usbcore/parameters/usbfs_memory_mb" | sudo tee /etc/sudoers.d/usbfs_memory > /dev/null
sudo chmod 0440 /etc/sudoers.d/usbfs_memory
echo -e "${GREEN}✓ Sudoers szabály beállítva ($ACTUAL_USER felhasználóhoz).${NC}"

# 3. Jelenlegi érték azonnali frissítése
echo -e "${YELLOW}[3/3] usbfs_memory_mb beállítása 0-ra a jelenlegi munkamenetben...${NC}"
if [ -f /sys/module/usbcore/parameters/usbfs_memory_mb ]; then
    echo 0 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb > /dev/null
    echo -e "${GREEN}✓ Jelenlegi érték frissítve: $(cat /sys/module/usbcore/parameters/usbfs_memory_mb) MB${NC}"
fi

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN} SIKER! A rendszer mostantól minden indításnál        ${NC}"
echo -e "${GREEN} automatikusan 0-ra állítja az usbfs memóriát.        ${NC}"
echo -e "${GREEN} Többé nem kell kézzel megadnod az echo 0 parancsot!   ${NC}"
echo -e "${GREEN}======================================================${NC}"
