"""
DEIK Robot Foci Kapus – Munkamenet Perzisztencia (Session Manager)
=================================================================

Automatikusan menti a lövési adatokat JSON Lines formátumban.
Minden lövés egy önálló JSON sor, ami hatékony append-only írást
és egyszerű visszaolvasást tesz lehetővé.

Fájlok helye: data/sessions/session_YYYYMMDD_HHMMSS.jsonl
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Lövési munkamenet kezelő.

    Automatikusan ment minden lövési eseményt egy JSONL fájlba,
    és képes korábbi munkameneteket visszatölteni.
    """

    def __init__(self, session_dir: str = "data/sessions"):
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._session_file = self._session_dir / f"session_{ts}.jsonl"
        self._shots: List[dict] = []
        self._session_start = time.time()
        logger.info("Munkamenet indítva: %s", self._session_file)

    def save_shot(self, record: dict) -> None:
        """Hozzáfűz egy lövési rekordot a munkamenet fájlhoz."""
        record_with_ts = {
            "session_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **record,
        }
        self._shots.append(record_with_ts)
        try:
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_with_ts, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Munkamenet mentési hiba: %s", e)

    def load_session(self, path: str) -> List[dict]:
        """Visszatölt egy korábbi munkamenetet a megadott fájlból."""
        shots: List[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        shots.append(json.loads(line))
            logger.info("Munkamenet betöltve: %s (%d lövés)", path, len(shots))
        except Exception as e:
            logger.error("Munkamenet betöltési hiba: %s", e)
        return shots

    def get_latest_session(self) -> Optional[str]:
        """Visszaadja a legutóbbi munkamenet fájl elérési útját."""
        files = sorted(self._session_dir.glob("session_*.jsonl"))
        return str(files[-1]) if files else None

    def get_session_summary(self) -> Dict[str, object]:
        """Visszaadja az aktuális munkamenet összesítő adatait."""
        elapsed_s = time.time() - self._session_start
        elapsed_min = elapsed_s / 60.0
        n = len(self._shots)
        in_goal = sum(1 for s in self._shots if s.get("in_goal", False))
        speeds = [s.get("speed_kmh", 0.0) for s in self._shots if s.get("speed_kmh", 0.0) > 0]
        return {
            "shot_count": n,
            "in_goal": in_goal,
            "elapsed_min": elapsed_min,
            "avg_speed_kmh": sum(speeds) / len(speeds) if speeds else 0.0,
            "max_speed_kmh": max(speeds) if speeds else 0.0,
            "session_file": str(self._session_file),
        }

    def clear(self) -> None:
        """Törli az aktuális munkamenet memória-beli adatait (a fájl megmarad)."""
        self._shots.clear()
        # Új fájlt indítunk
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._session_file = self._session_dir / f"session_{ts}.jsonl"
        self._session_start = time.time()
        logger.info("Munkamenet nullázva, új fájl: %s", self._session_file)

    @property
    def session_file(self) -> str:
        return str(self._session_file)

    @property
    def shot_count(self) -> int:
        return len(self._shots)
