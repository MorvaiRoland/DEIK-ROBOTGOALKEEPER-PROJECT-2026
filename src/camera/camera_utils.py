"""
DEIK Robot Foci Kapus – Kamera Segéd segédprogramok (Camera Utils)
==================================================================

Ez a modul Linux USB rendszerszintű paraméterek (pl. usbfs_memory_mb)
automatikus ellenőrzését és konfigurálását látja el.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

USBFS_SYSFS_PATH = "/sys/module/usbcore/parameters/usbfs_memory_mb"


def ensure_usbfs_memory_mb(target_mb: int = 0) -> bool:
    """
    Ellenőrzi és automatikusan beállítja a Linux USBFS memórialimitet.

    A Ximea MC023CG és más ipari USB3 kamerák 100+ FPS sebességű működéséhez
    a rendszer usbfs memórialimitének 0-nak (korlátlan / dinamikus) kell lennie.

    Args:
        target_mb: A cél értéke (alapértelmezett: 0)

    Returns:
        True ha sikeres vagy már be van állítva, False ha sikertelen.
    """
    if sys.platform != "linux" or not os.path.exists(USBFS_SYSFS_PATH):
        return True

    try:
        with open(USBFS_SYSFS_PATH, "r") as f:
            current_val = int(f.read().strip())

        if current_val == target_mb:
            logger.info("✓ USBFS memórialimit (usbfs_memory_mb) rendben: %d MB", current_val)
            return True

        logger.warning(
            "USBFS memórialimit (usbfs_memory_mb=%d) nem megfelelő. Automatizált beállítás %d MB-ra...",
            current_val, target_mb
        )

        # 1. Kísérlet: Közvetlen fájlírás (ha root joggal fut a folyamat)
        try:
            with open(USBFS_SYSFS_PATH, "w") as f:
                f.write(str(target_mb))
            logger.info("✓ usbfs_memory_mb sikeresen átállítva %d MB-ra (közvetlen írás).", target_mb)
            return True
        except PermissionError:
            pass

        # 2. Kísérlet: Jelszómentes sudo tee (sudoers beállítás alapján)
        cmd_sudo_n = ["sudo", "-n", "tee", USBFS_SYSFS_PATH]
        res_n = subprocess.run(cmd_sudo_n, input=f"{target_mb}\n".encode(), capture_output=True)
        if res_n.returncode == 0:
            logger.info("✓ usbfs_memory_mb sikeresen átállítva %d MB-ra (jelszómentes sudo).", target_mb)
            return True

        # 3. Kísérlet: Interaktív sudo / pkexec
        cmd_sh = f"echo {target_mb} | sudo tee {USBFS_SYSFS_PATH}"
        res_sh = subprocess.run(cmd_sh, shell=True, capture_output=True)
        if res_sh.returncode == 0:
            logger.info("✓ usbfs_memory_mb sikeresen átállítva %d MB-ra (sudo).", target_mb)
            return True

        logger.error(
            "Nem sikerült automatikusan átállítani a usbfs_memory_mb értéket! "
            "Futtasd a 'scripts/setup_usbfs_memory.sh' szkriptet."
        )
        return False

    except Exception as exc:
        logger.warning("Hiba a usbfs_memory_mb ellenőrzése során: %s", exc)
        return False
