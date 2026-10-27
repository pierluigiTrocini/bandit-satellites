"""Geometria: propagazione orbitale, sistemi di riferimento e finestre di accesso.

Il resto del pacchetto vede la geometria solo attraverso un *fornitore di posizioni* (`PositionProvider`):
un oggetto con `eci(t)`, `ecef(t)` e `ecef_to_eci(r, t)` (t = secondi dall'epoca; metri; forma (n, 3)).
Ne esistono tre:
  - `NumericalJ2`  integrazione numerica di gravita' centrale + J2 da elementi *osculanti* (come BSK-RL);
  - `KeplerJ2`     soluzione analitica: orbita kepleriana con deriva secolare J2, elementi interpretati come *medi*
                   (veloce, ma il periodo differisce dallo 0,1% circa da una propagazione osculante);
  - `bsk_adapter.BskTrajectory`   traiettoria propagata da BSK-RL (gravita' completa + resistenza).
Le finestre di accesso (`find_windows`) lavorano su un qualunque fornitore.

Convenzioni: ECI = J2000 (come BSK-RL); ECEF = Rz(GMST) * P * r_J2000 con P la precessione IAU 1976 all'epoca
(la nutazione, ~17", e' ignorata); terra sferica, elevazione calcolata sulla sfera di raggio RE.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq

MU = 3.98600436e14                 # [m^3/s^2] come BSK-RL
RE = 6378136.6                     # [m] raggio equatoriale (BSK-RL: 6378.1366 km)
J2 = 1.08262668e-3
OMEGA_E = 7.2921150e-5             # [rad/s] velocita' di rotazione terrestre
SIDEREAL_YEAR_S = 365.2422 * 86400.0
D2R = math.pi / 180.0
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- tempo

def parse_epoch(text):
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def julian_date(dt):
    """Data giuliana (UTC ~ UT1) di un datetime con fuso."""
    return (dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 2451545.0


def gmst_rad(dt):
    """Tempo siderale medio di Greenwich [rad] (IAU 1982)."""
    d = julian_date(dt) - 2451545.0
    T = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T * T - T ** 3 / 38710000.0
    return math.radians(deg % 360.0)


def mean_sun_ra_deg(dt):
    """Ascensione retta del Sole medio [deg] (formula a bassa precisione, la stessa di config_io)."""
    d = (dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    return (280.460 + 0.9856474 * d) % 360.0


# ---------------------------------------------------------------- orbita

def sso_inclination_deg(alt_km):
    """Inclinazione per cui la precessione J2 e' di 360 gradi all'anno (orbita circolare)."""
    a = RE + alt_km * 1e3
    n = math.sqrt(MU / a ** 3)
    c = -(TWO_PI / SIDEREAL_YEAR_S) / (1.5 * J2 * (RE / a) ** 2 * n)
    if abs(c) > 1:
        raise ValueError(f"orbita sun-synchronous impossibile a {alt_km} km di quota")
    return math.degrees(math.acos(c))


def sso_raan_deg(dt, ltdn_hours):
    """RAAN [deg] per cui il nodo discendente passa alle `ltdn_hours` ora solare media locale."""
    ltan = (ltdn_hours + 12.0) % 24.0
    return (mean_sun_ra_deg(dt) + (ltan - 12.0) * 15.0) % 360.0


@dataclass(frozen=True)
class Orbit:
    """Elementi kepleriani all'epoca (angoli in radianti, semiasse in metri)."""
    a: float
    e: float
    inc: float
    raan: float
    argp: float
    nu: float          # anomalia vera

    @property
    def mean_motion(self):
        return math.sqrt(MU / self.a ** 3)

    @property
    def period_s(self):
        return TWO_PI / self.mean_motion

    @staticmethod
    def from_spec(spec, epoch):
        """Da un blocco `orbit` nello stesso formato di constellation/v1 (angoli in gradi)."""
        kind = spec["type"]
        alt = float(spec["altitude_km"])
        if kind == "keplerian":
            inc, raan = float(spec["inclination_deg"]), float(spec["raan_deg"])
        elif kind == "sun_synchronous":
            inc, raan = sso_inclination_deg(alt), sso_raan_deg(epoch, float(spec["ltdn_hours"]))
        else:
            raise ValueError(f"tipo di orbita sconosciuto: {kind!r}")
        return Orbit(a=RE + alt * 1e3, e=float(spec.get("eccentricity", 0.0)), inc=inc * D2R, raan=raan * D2R,
                     argp=float(spec.get("arg_perigee_deg", 0.0)) * D2R, nu=float(spec.get("true_anomaly_deg", 0.0)) * D2R)


def precession_matrix(dt):
    """Matrice di precessione IAU 1976 (J2000 -> media dell'epoca `dt`)."""
    T = (julian_date(dt) - 2451545.0) / 36525.0
    asec = math.pi / 648000.0
    zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T ** 3) * asec
    z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T ** 3) * asec
    th = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T ** 3) * asec

    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])           # rotazione di sistema (passiva) attorno a z

    def ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, -s], [0, 1.0, 0], [s, 0, c]])

    return rz(-z) @ ry(th) @ rz(-zeta)


class Frames:
    """ECI (J2000) <-> ECEF all'epoca data: ECEF = Rz(theta0 + OMEGA_E t) * P * r_J2000."""

    def __init__(self, epoch):
        self.epoch, self.theta0, self.P = epoch, gmst_rad(epoch), precession_matrix(epoch)

    def rotation(self, t):
        return self.theta0 + OMEGA_E * np.asarray(t, dtype=float)

    def eci_to_ecef(self, r, t):
        return eci_to_ecef(np.atleast_2d(r) @ self.P.T, self.rotation(t))

    def ecef_to_eci(self, r, t):
        return ecef_to_eci(r, self.rotation(t)) @ self.P

    @property
    def pole(self):
        """Direzione dell'asse polare (della data) espressa in J2000."""
        return self.P[2]


class KeplerJ2:
    """Propagatore analitico: moto kepleriano con le derive secolari di RAAN, argomento del perigeo e
    anomalia media dovute a J2. Niente resistenza atmosferica ne' termini periodici: valido per ore o
    giorni a quote LEO (l'errore rispetto a un propagatore completo si misura con `scripts/validate_aeossp.py`).

    Fornitore di posizioni: `eci(t)`, `ecef(t)`, `ecef_to_eci(r, t)`; t in secondi dall'epoca.
    Gli elementi sono *medi*; il piano orbitale e' trattato come se l'asse polare fosse l'asse z di J2000
    (la differenza, ~0.35 gradi di precessione, e' un effetto del secondo ordine sulla deriva)."""

    def __init__(self, orbit, epoch, j2=True):
        self.orbit, self.epoch, self.j2 = orbit, epoch, j2
        self.frames = Frames(epoch)
        o = orbit
        n, p = o.mean_motion, o.a * (1 - o.e ** 2)
        k = J2 * (RE / p) ** 2 if j2 else 0.0
        c = math.cos(o.inc)
        self.raan_dot = -1.5 * k * n * c
        self.argp_dot = 0.75 * k * n * (5 * c * c - 1)
        self.m_dot = n * (1 + 0.75 * k * math.sqrt(1 - o.e ** 2) * (3 * c * c - 1))
        E0 = 2 * math.atan2(math.sqrt(1 - o.e) * math.sin(o.nu / 2), math.sqrt(1 + o.e) * math.cos(o.nu / 2))
        self.m0 = E0 - o.e * math.sin(E0)
        self.theta0 = self.frames.theta0

    def eci(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        o = self.orbit
        M = (self.m0 + self.m_dot * t) % TWO_PI
        E = M.copy() if o.e < 0.8 else np.full_like(M, math.pi)
        for _ in range(30):                                    # Newton su E - e sin E = M
            d = (E - o.e * np.sin(E) - M) / (1 - o.e * np.cos(E))
            E -= d
            if np.max(np.abs(d)) < 1e-13:
                break
        xp, yp = o.a * (np.cos(E) - o.e), o.a * math.sqrt(1 - o.e ** 2) * np.sin(E)
        raan, argp = o.raan + self.raan_dot * t, o.argp + self.argp_dot * t
        cO, sO, cw, sw = np.cos(raan), np.sin(raan), np.cos(argp), np.sin(argp)
        ci, si = math.cos(o.inc), math.sin(o.inc)
        x = (cO * cw - sO * sw * ci) * xp + (-cO * sw - sO * cw * ci) * yp
        y = (sO * cw + cO * sw * ci) * xp + (-sO * sw + cO * cw * ci) * yp
        z = (sw * si) * xp + (cw * si) * yp
        return np.stack([x, y, z], axis=1)

    def rotation(self, t):
        return self.frames.rotation(t)

    def ecef(self, t):
        return self.frames.eci_to_ecef(self.eci(t), t)

    def ecef_to_eci(self, r, t):
        return self.frames.ecef_to_eci(r, t)

    def orbit_phase(self, t):
        """Argomento di latitudine medio [rad, non ridotto]: cresce di 2 pi a ogni orbita (conteggio dei cicli)."""
        t = np.asarray(t, dtype=float)
        return self.m0 + self.orbit.argp + (self.m_dot + self.argp_dot) * t


class NumericalJ2:
    """Integrazione numerica di gravita' centrale + J2 a partire da elementi *osculanti* all'epoca (come BSK-RL),
    in J2000, con l'asse polare inclinato della precessione. Niente resistenza ne' armoniche superiori.
    Fornitore di posizioni: `eci(t)`, `ecef(t)`, `ecef_to_eci(r, t)`."""

    def __init__(self, orbit, epoch, horizon, j2=True, rtol=1e-11):
        from scipy.integrate import solve_ivp
        self.orbit, self.epoch, self.frames = orbit, epoch, Frames(epoch)
        o = orbit
        p = o.a * (1 - o.e ** 2)
        xp, yp = p * math.cos(o.nu) / (1 + o.e * math.cos(o.nu)), p * math.sin(o.nu) / (1 + o.e * math.cos(o.nu))
        vp, vq = -math.sqrt(MU / p) * math.sin(o.nu), math.sqrt(MU / p) * (o.e + math.cos(o.nu))
        cO, sO, cw, sw, ci, si = math.cos(o.raan), math.sin(o.raan), math.cos(o.argp), math.sin(o.argp), math.cos(o.inc), math.sin(o.inc)
        R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci], [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci], [sw * si, cw * si]])
        r0, v0 = R @ np.array([xp, yp]), R @ np.array([vp, vq])
        pole = self.frames.pole if j2 else np.array([0.0, 0.0, 1.0])
        k = 1.5 * J2 * MU * RE ** 2 if j2 else 0.0

        def f(_, y):
            r = y[:3]
            rn = np.linalg.norm(r)
            zp = float(r @ pole)
            a = -MU * r / rn ** 3 - k / rn ** 5 * ((1 - 5 * zp ** 2 / rn ** 2) * r + 2 * zp * pole)
            return np.concatenate([y[3:], a])

        self.sol = solve_ivp(f, (0.0, float(horizon)), np.concatenate([r0, v0]), method="DOP853", rtol=rtol, atol=1e-6, dense_output=True)
        self.horizon = float(horizon)

    def eci(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.sol.sol(np.clip(t, 0.0, self.horizon)).T[:, :3]

    def rotation(self, t):
        return self.frames.rotation(t)

    def ecef(self, t):
        return self.frames.eci_to_ecef(self.eci(t), t)

    def ecef_to_eci(self, r, t):
        return self.frames.ecef_to_eci(r, t)


def eci_to_ecef(r, theta):
    """Ruota di -theta attorno a z. r: (n, 3), theta: (n,) o scalare."""
    r, theta = np.atleast_2d(r), np.broadcast_to(np.asarray(theta, dtype=float), (len(np.atleast_2d(r)),))
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1], r[:, 2]], axis=1)


def ecef_to_eci(r, theta):
    r, theta = np.atleast_2d(r), np.broadcast_to(np.asarray(theta, dtype=float), (len(np.atleast_2d(r)),))
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]], axis=1)


def lla_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """Terra sferica di raggio RE (+ quota)."""
    la, lo, R = np.asarray(lat_deg) * D2R, np.asarray(lon_deg) * D2R, RE + np.asarray(alt_m)
    return np.stack([R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la)], axis=-1)


def ecef_to_latlon(r):
    r = np.atleast_2d(r)
    return np.degrees(np.arcsin(r[:, 2] / np.linalg.norm(r, axis=1))), np.degrees(np.arctan2(r[:, 1], r[:, 0]))


# ---------------------------------------------------------------- accesso

def elevation(r_sat, r_loc):
    """Elevazione [rad] del satellite visto da luoghi fissi. r_sat: (n, 3), r_loc: (m, 3) -> (m, n)."""
    d = r_sat[None, :, :] - r_loc[:, None, :]
    up = r_loc / np.linalg.norm(r_loc, axis=1, keepdims=True)
    return np.arcsin(np.clip(np.einsum("mnk,mk->mn", d, up) / np.linalg.norm(d, axis=2), -1.0, 1.0))


def min_elevation_from_offnadir(max_offnadir_deg, alt_m):
    """Elevazione minima [deg] equivalente a un angolo off-nadir massimo (orbita circolare, Terra sferica)."""
    s = (RE + alt_m) / RE * math.sin(max_offnadir_deg * D2R)
    if s >= 1.0:
        return 0.0
    return math.degrees(math.acos(s))


def find_windows(provider, r_loc, min_elev_rad, t0, t1, step=5.0):
    """Finestre [apertura, chiusura] (s) in cui l'elevazione e' >= soglia, per ciascun luogo ECEF.

    Campionamento ogni `step` s e raffinamento degli estremi con una ricerca di zeri (stesso metodo di
    `scripts/ground.py::WindowEngine`). Finestre piu' corte di `step` possono mancare.
    r_loc: (m, 3) in ECEF; min_elev_rad: scalare o (m,). Restituisce una lista di m liste di (a, c)."""
    r_loc = np.atleast_2d(r_loc)
    me = np.broadcast_to(np.asarray(min_elev_rad, dtype=float), (len(r_loc),))
    ts = np.append(np.arange(t0, t1, step), t1)
    el = elevation(provider.ecef(ts), r_loc) - me[:, None]
    out = []
    for m in range(len(r_loc)):
        ok = el[m] >= 0
        if not ok.any():
            out.append([])
            continue

        def f(t, m=m):
            return float(elevation(provider.ecef(np.array([t])), r_loc[m:m + 1])[0, 0] - me[m])

        edges = np.flatnonzero(ok[1:] != ok[:-1])
        opens, closes = ([ts[0]] if ok[0] else []), []
        for j in edges:
            root = brentq(f, ts[j], ts[j + 1], xtol=1e-3)
            (opens if ok[j + 1] else closes).append(root)
        if ok[-1]:
            closes.append(ts[-1])
        out.append([(float(a), float(c)) for a, c in zip(opens, closes) if c - a > 0.1])
    return out
