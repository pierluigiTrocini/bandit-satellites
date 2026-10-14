"""Richieste di imaging a runtime: sorgenti (file, flusso Poisson) e target BSK-RL dinamici.

Una richiesta e' un dizionario {id, lat, lon, name, elevation_m, priority, arrival_s, expires_s|None} (lo stesso
formato normalizzato di `config_io.load_requests`). Le sorgenti restituiscono, per un istante t, le richieste
arrivate fino a t, in ordine di arrivo.

Riproducibilita': ogni quantita' casuale ha un flusso proprio derivato dal seed (`config_io.rng_stream`) e ogni
richiesta consuma sempre lo stesso numero di estrazioni da ciascun flusso, quindi la sequenza e' identica a
parita' di seed qualunque sia la politica (e il momento in cui la si legge).
"""

import numpy as np
from bsk_rl.scene.targets import Target
from bsk_rl.utils.orbital import lla2ecef

from config_io import rng_stream

R_M = 6378136.6  # [m] raggio usato da BSK-RL per i target


class RequestTarget(Target):
    """Target BSK-RL con identita' stabile: `id` e' l'identificativo della richiesta (quello di BSK-RL contiene
    `id(self)` e cambia a ogni esecuzione)."""

    def __init__(self, req):
        super().__init__(req["name"], lla2ecef(req["lat"], req["lon"], R_M + req.get("elevation_m", 0.0)),
                         req["priority"])
        self._id = req["id"]
        self.arrival_s, self.expires_s = req["arrival_s"], req["expires_s"]


class FileSource:
    """Richieste da un elenco (requests/v1)."""

    def __init__(self, requests):
        self.items, self.i = sorted(requests, key=lambda r: r["arrival_s"]), 0

    def pop_until(self, t):
        out = []
        while self.i < len(self.items) and self.items[self.i]["arrival_s"] <= t:
            out.append(self.items[self.i])
            self.i += 1
        return out


class PoissonSource:
    """Arrivi di Poisson a tasso costante (per ora), posizione uniforme sulla sfera, priorita' uniforme in (min, max],
    scadenza opzionale a durata fissa. `initial_count` richieste sono gia' presenti a t = 0."""

    def __init__(self, seed, spec):
        self.rate = spec["rate_per_hour"] / 3600.0
        self.pmin, self.pmax = spec["priority"]["min"], spec["priority"]["max"]
        self.validity = spec["validity_s"]
        self.arr, self.space, self.prio = (rng_stream(seed, k) for k in ("arrivals", "space", "priority"))
        self.n, self.initial = 0, spec["initial_count"]
        self.next_t = self._draw_gap() if self.initial == 0 else 0.0

    def _draw_gap(self):
        return float(self.arr.exponential(1.0 / self.rate))

    def _make(self, t):
        self.n += 1
        lat = float(np.degrees(np.arcsin(self.space.uniform(-1.0, 1.0))))
        lon = float(self.space.uniform(-180.0, 180.0))
        prio = float(self.pmax - (self.pmax - self.pmin) * self.prio.random())      # in (min, max]
        rid = f"G{self.n:06d}"
        return dict(id=rid, lat=round(lat, 4), lon=round(lon, 4), name=rid, elevation_m=0.0, priority=round(prio, 4),
                    arrival_s=round(t, 3), expires_s=None if self.validity is None else round(t + self.validity, 3))

    def pop_until(self, t):
        out = []
        while self.n < self.initial:                      # arretrato iniziale, tutto a t = 0
            out.append(self._make(0.0))
            if self.n == self.initial:
                self.next_t = self._draw_gap()
        while self.next_t <= t:
            out.append(self._make(self.next_t))
            self.next_t += self._draw_gap()
        return out


def make_source(seed, targets, requests=None):
    """Sorgente di richieste dal blocco `targets` dello scenario (modalita' stream o file); None per random."""
    if targets["mode"] == "stream":
        return PoissonSource(seed, targets)
    if targets["mode"] == "file":
        if requests is None:
            raise ValueError("mode 'file': manca l'elenco delle richieste (requests/v1)")
        return FileSource(requests)
    return None


def requests_file(requests, name="richieste generate"):
    """Elenco di richieste nel formato requests/v1 (per salvare un flusso generato e rileggerlo)."""
    keep = ("id", "lat", "lon", "name", "elevation_m", "priority", "arrival_s", "expires_s")
    out = []
    for r in requests:
        item = dict(id=r["id"], target=dict(lat=r["lat"], lon=r["lon"], name=r["name"], elevation_m=r["elevation_m"]),
                    priority=r["priority"], arrival_s=r["arrival_s"])
        if r["expires_s"] is not None:
            item["expires_s"] = r["expires_s"]
        out.append(item)
    return dict(schema="requests/v1", name=name, requests=out)
