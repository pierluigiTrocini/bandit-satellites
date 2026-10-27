"""Modello dei satelliti: tempo di transizione (manovra + stabilizzazione) e budget di risorse.

Nel paper (Wang et al., 2020) il tempo di transizione fra due task consecutivi e' DeltaT = DeltaV + DeltaS:
DeltaV = tempo di manovra d'assetto, DeltaS = tempo di stabilizzazione; entrambi sono *parametri noti*.
Qui DeltaV e' un profilo di manovra «bang-bang» con velocita' e accelerazione angolari massime, funzione
dell'angolo fra le due direzioni di puntamento (linee di vista inerziali): e' monotona, concava e subadditiva.
Come nel paper, la transizione (5) si impone fra task *consecutivi* dello stesso satellite.

I valori di default sono neutri/illustrativi: NON rappresentano un satellite reale e vanno calibrati.
"""

import math
from dataclasses import dataclass, field

from .geometry import Orbit

D2R = math.pi / 180.0


@dataclass(frozen=True)
class SlewModel:
    max_rate: float = 1.0 * D2R        # [rad/s]
    max_accel: float = 0.5 * D2R       # [rad/s^2]
    settle_s: float = 5.0              # DeltaS [s], dovuto a ogni cambio di puntamento

    def delta_v(self, theta):
        """Tempo di manovra per un angolo theta [rad] (scalare o array): accelera, (crociera), decelera."""
        import numpy as np
        th = np.abs(np.asarray(theta, dtype=float))
        w, a = self.max_rate, self.max_accel
        short = 2.0 * np.sqrt(th / a)                  # triangolare: non raggiunge la velocita' massima
        long = th / w + w / a                          # trapezoidale
        return np.where(th <= w * w / a, short, long)

    def transition(self, theta):
        """DeltaT = DeltaV + DeltaS."""
        return self.delta_v(theta) + self.settle_s

    @property
    def max_transition(self):
        return float(self.transition(math.pi))


@dataclass(frozen=True)
class Resources:
    """Budget per ciclo orbitale (o per orizzonte): memoria (12) ed energia (13) del paper."""
    memory_capacity_gbit: float = 20.0     # M^C
    data_rate_mbps: float = 800.0          # M^I: dati prodotti per secondo di osservazione
    energy_capacity_wh: float = 5.0        # E^C
    imaging_power_w: float = 100.0         # E^I
    maneuver_power_w: float = 200.0        # E^M

    def memory(self, duration_s):
        return self.data_rate_mbps * 1e-3 * duration_s                 # Gbit

    def energy_imaging(self, duration_s):
        return self.imaging_power_w * duration_s / 3600.0              # Wh

    def energy_maneuver(self, dv_s):
        return self.maneuver_power_w * dv_s / 3600.0                   # Wh


@dataclass(frozen=True)
class SatSpec:
    name: str
    orbit: Orbit
    agility: str = "agile"                 # "agile": OTW libera dentro la VTW; "none": osservazione sull'intera VTW, nessuna manovra
    observation_s: float = 2.0             # d: durata di un'osservazione (modello agile)
    min_elevation_deg: float = 40.0        # soglia di visibilita' (equivale a un off-nadir massimo)
    slew: SlewModel = field(default_factory=SlewModel)
    resources: Resources = field(default_factory=Resources)

    @property
    def agile(self):
        return self.agility == "agile"
