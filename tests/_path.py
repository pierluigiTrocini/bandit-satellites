"""Aggiunge src/ (e scripts/) al percorso di import: i test si lanciano da qualunque cartella."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
