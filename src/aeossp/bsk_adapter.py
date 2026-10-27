"""Adattatore: la traiettoria propagata da BSK-RL come fornitore di posizioni (`PositionProvider`).

Non importa Basilisk ne' BSK-RL: usa per «duck typing» l'oggetto `sat.trajectory` di un satellite BSK-RL
(`TrajectorySimulator`), cioe' gli stessi dati che `scripts/ground.py::WindowEngine` usa per le finestre di accesso:
  - `trajectory.times`, `_r_BN_N_log` (posizione inerziale), `_J20002Pfix_log` (matrice inerziale -> planet-fixed).
Cosi' il livello `Instance` puo' essere costruito sulla stessa orbita di BSK-RL (`build_instance(cfg, providers=[BskTrajectory(sat)])`)
e confrontato con il propagatore analitico `geometry.KeplerJ2`.
"""

import numpy as np
from scipy.interpolate import interp1d


class BskTrajectory:
    """Fornitore di posizioni dalla traiettoria di un satellite BSK-RL. t in secondi dall'epoca dello scenario."""

    def __init__(self, sat, horizon):
        self.sat, self.horizon = sat, horizon
        tr = sat.trajectory
        tr.extend_to(horizon + 3 * tr.dt)
        self.t = np.asarray(tr.times, dtype=float)
        self.r_eci = np.array(tr._r_BN_N_log, dtype=float)
        pn = np.array(tr._J20002Pfix_log, dtype=float)                   # (n, 3, 3): inerziale -> planet-fixed
        self.r_ecef = np.einsum("nij,nj->ni", pn, self.r_eci)
        th = np.unwrap(np.arctan2(pn[:, 0, 1], pn[:, 0, 0]))              # angolo di rotazione terrestre (attorno a z)
        self._eci = interp1d(self.t, self.r_eci, kind="cubic", axis=0, fill_value="extrapolate")
        self._ecef = interp1d(self.t, self.r_ecef, kind="cubic", axis=0, fill_value="extrapolate")
        self._rot = interp1d(self.t, th, kind="linear", fill_value="extrapolate")
        self._pn = interp1d(self.t, pn.reshape(len(pn), 9), kind="linear", axis=0, fill_value="extrapolate")

    def eci(self, t):
        return self._eci(np.atleast_1d(np.asarray(t, dtype=float)))

    def ecef(self, t):
        return self._ecef(np.atleast_1d(np.asarray(t, dtype=float)))

    def rotation(self, t):
        return self._rot(np.asarray(t, dtype=float))

    def ecef_to_eci(self, r, t):
        """Inversa della matrice inerziale -> planet-fixed di BSK-RL (interpolata e riortogonalizzata)."""
        r = np.atleast_2d(r)
        pn = self._pn(np.broadcast_to(np.asarray(t, dtype=float), (len(r),))).reshape(-1, 3, 3)
        u, _, vt = np.linalg.svd(pn)
        pn = u @ vt
        return np.einsum("nji,nj->ni", pn, r)
