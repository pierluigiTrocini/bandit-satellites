"""Ricava la mappa di base del viewer da Natural Earth (pubblico dominio).

Produce in viewer/web/static/:
  - earth.jpg  : "Natural Earth I with Shaded Relief, Water, and Drainage" 50m, ridotta a 4096x2048
                 (proiezione equirettangolare, lon -180..180, lat 90..-90)
  - coast.json : contorni delle terre emerse (ne_50m_land), liste di [lon, lat]

Fonte: https://www.naturalearthdata.com (termini: https://www.naturalearthdata.com/about/terms-of-use/).
I file sorgente (~90 MB) si scaricano una volta in outputs/cache/ (cartella ignorata da git).
Uso: uv run python scripts/build_basemap.py
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

RASTER_URL = "https://naturalearth.s3.amazonaws.com/50m_raster/NE1_50M_SR_W.zip"
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "outputs" / "cache"
OUT = ROOT / "viewer" / "web" / "static"
Image.MAX_IMAGE_PIXELS = None


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("scarico", url)
        with urllib.request.urlopen(url) as r:
            dest.write_bytes(r.read())
    return dest


def thin(ring: np.ndarray, step: float = 0.3) -> np.ndarray:
    """Tiene un vertice solo se dista almeno `step` gradi dall'ultimo tenuto (e l'ultimo punto)."""
    keep = [0]
    for i in range(1, len(ring) - 1):
        if np.hypot(*(ring[i] - ring[keep[-1]])) >= step:
            keep.append(i)
    keep.append(len(ring) - 1)
    return ring[keep]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    zpath = fetch(RASTER_URL, CACHE / "NE1_50M_SR_W.zip")
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".tif"))
        im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
    print("sorgente:", im.size)
    im.resize((4096, 2048), Image.LANCZOS).save(OUT / "earth.jpg", quality=80, optimize=True)

    land = json.loads(fetch(LAND_URL, CACHE / "ne_50m_land.geojson").read_text())
    out = []
    for feat in land["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for ring in poly:  # anello esterno + eventuali buchi (laghi interni)
                r = thin(np.asarray(ring, dtype=float))
                if len(r) >= 8:  # scarta isolette minuscole
                    out.append(np.round(r, 2).tolist())
    (OUT / "coast.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"contorni: {len(out)} polilinee, {sum(map(len, out))} punti")


if __name__ == "__main__":
    main()
