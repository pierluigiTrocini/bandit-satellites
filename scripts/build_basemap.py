"""Ricava la mappa di base del viewer dall'immagine inclusa in BSK-RL.

Produce in viewer/web/static/:
  - earth.jpg  : immagine ridotta a 4096x2048 (proiezione equirettangolare, lon -180..180, lat 90..-90)
  - coast.json : contorni terra/mare ricavati dalla stessa immagine (liste di [lon, lat])

I contorni sono approssimati (soglia di colore sull'immagine): nessuna fonte esterna.
Uso: uv run python scripts/build_basemap.py
"""

import json
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from PIL import Image
from scipy.ndimage import gaussian_filter

import bsk_rl

SRC = Path(bsk_rl.__file__).parent / "_dat" / "world.200407.3x5400x2700.jpg"
OUT = Path(__file__).resolve().parent.parent / "viewer" / "web" / "static"
Image.MAX_IMAGE_PIXELS = None


def land_score(im: Image.Image) -> np.ndarray:
    """1 = terra, 0 = mare: il mare e' blu scuro (B domina su R e G)."""
    a = np.asarray(im, dtype=float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ocean = (b > r * 1.25) & (b > g * 1.05) & (a.max(-1) < 0.75)
    return gaussian_filter((~ocean).astype(float), 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.open(SRC).convert("RGB")
    print("sorgente:", im.size)

    im.resize((4096, 2048), Image.LANCZOS).save(OUT / "earth.jpg", quality=80, optimize=True)

    small = im.resize((1440, 720), Image.LANCZOS)
    score = land_score(small)
    h, w = score.shape
    lon = np.linspace(-180, 180, w)
    lat = np.linspace(90, -90, h)
    lines = contour_generator(lon, lat, score).lines(0.5)
    out = []
    for ln in lines:
        if len(ln) < 40:  # scarta laghi e isolette minuscole
            continue
        ln = ln[:: 3]  # sottocampiona (~0.75 deg)
        out.append(np.round(ln, 2).tolist())
    (OUT / "coast.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"contorni: {len(out)} polilinee, {sum(map(len, out))} punti")


if __name__ == "__main__":
    main()
