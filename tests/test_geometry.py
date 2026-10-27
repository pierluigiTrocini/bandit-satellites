"""Test di integrita' del livello geometrico: ogni controllo ha un riferimento analitico indipendente."""
import math
import unittest

import _path  # noqa: F401
import numpy as np

from aeossp import geometry as G

EPOCH = G.parse_epoch("2025-03-21T00:00:00Z")


def circ(alt_km, inc_deg, raan_deg=0.0, nu_deg=0.0):
    return G.Orbit(a=G.RE + alt_km * 1e3, e=0.0, inc=inc_deg * G.D2R, raan=raan_deg * G.D2R, argp=0.0, nu=nu_deg * G.D2R)


class TestTime(unittest.TestCase):
    def test_gmst_j2000(self):
        # valore di riferimento: GMST alle 12:00 UTC del 2000-01-01 = 280.46061837 gradi
        g = G.gmst_rad(G.parse_epoch("2000-01-01T12:00:00Z"))
        self.assertAlmostEqual(math.degrees(g), 280.46061837, places=6)

    def test_sidereal_day(self):
        # un punto fisso nello spazio inerziale torna nello stesso punto ECEF dopo un giorno siderale
        k = G.KeplerJ2(circ(500, 55), EPOCH)
        r = np.array([[7e6, 1e6, 2e6]])
        sidereal = 2 * math.pi / G.OMEGA_E
        a = G.eci_to_ecef(r, k.rotation(0.0))
        b = G.eci_to_ecef(r, k.rotation(sidereal))
        np.testing.assert_allclose(a, b, atol=1e-3)


class TestOrbit(unittest.TestCase):
    def test_period_and_closure(self):
        o = circ(500, 55)
        k = G.KeplerJ2(o, EPOCH, j2=False)
        T = 2 * math.pi * math.sqrt(o.a ** 3 / G.MU)
        self.assertAlmostEqual(o.period_s, T, places=6)
        r = k.eci(np.array([0.0, T]))
        np.testing.assert_allclose(r[0], r[1], atol=1e-3)           # senza J2 l'orbita si chiude esattamente

    def test_circular_radius_and_latitude(self):
        k = G.KeplerJ2(circ(786, 98.5), EPOCH)
        t = np.linspace(0, 5 * 6000, 4000)
        r = k.eci(t)
        np.testing.assert_allclose(np.linalg.norm(r, axis=1), G.RE + 786e3, rtol=1e-9)
        lat, _ = G.ecef_to_latlon(k.ecef(t))
        self.assertAlmostEqual(np.max(np.abs(lat)), 180 - 98.5, delta=0.1)     # la latitudine massima e' 180 - i

    def test_vis_viva_eccentric(self):
        o = G.Orbit(a=8e6, e=0.1, inc=0.9, raan=0.3, argp=0.7, nu=0.2)
        k = G.KeplerJ2(o, EPOCH, j2=False)
        t = np.array([0.0, 1000.0, 3000.0, 7000.0])
        r = k.eci(t)
        h = 1.0  # velocita' per differenze finite centrali
        v = (k.eci(t + h) - k.eci(t - h)) / (2 * h)
        energy = 0.5 * np.sum(v ** 2, axis=1) - G.MU / np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(energy, -G.MU / (2 * o.a), rtol=1e-6)

    def test_sun_synchronous_precession(self):
        # a 786 km l'inclinazione sun-sincrona e' ~98.6 e la RAAN deriva di 360 gradi/anno (0.9856 gradi/giorno)
        inc = G.sso_inclination_deg(786)
        self.assertAlmostEqual(inc, 98.6, delta=0.15)
        k = G.KeplerJ2(circ(786, inc), EPOCH)
        self.assertAlmostEqual(math.degrees(k.raan_dot) * 86400, 360.0 / 365.2422, delta=0.005)

    def test_ltdn_matches_config_io(self):
        # RAAN dalla LTDN: stessa formula di scripts/config_io.py (se disponibile)
        import config_io
        for ltdn in (10.5, 22.0):
            self.assertAlmostEqual(G.sso_raan_deg(EPOCH, ltdn), config_io.sso_raan_deg(EPOCH, ltdn), places=9)
        self.assertAlmostEqual(G.sso_inclination_deg(786), config_io.sso_inclination_deg(786), places=9)


class TestWindows(unittest.TestCase):
    def test_overhead_pass_duration(self):
        """Orbita equatoriale, stazione sul piano orbitale: durata analitica del passaggio centrale."""
        alt, el_min = 500e3, 10.0 * G.D2R
        o = circ(alt / 1e3, 0.0)
        k = G.KeplerJ2(o, EPOCH, j2=False)
        r0 = k.ecef(np.array([600.0]))[0]                         # stazione sotto la traccia, a 600 s
        lat, lon = G.ecef_to_latlon(r0[None, :])
        station = G.lla_to_ecef(0.0, lon[0])[None, :]
        wins = G.find_windows(k, station, el_min, 0.0, 6000.0, step=2.0)[0]
        self.assertGreaterEqual(len(wins), 1)
        lam = math.acos(G.RE * math.cos(el_min) / (G.RE + alt)) - el_min       # semi-angolo geocentrico
        rel_rate = o.mean_motion - G.OMEGA_E                                    # moto relativo satellite-stazione
        expected = 2 * lam / rel_rate
        a, c = next(w for w in wins if w[0] < 600 < w[1])
        self.assertAlmostEqual(c - a, expected, delta=1.0)
        self.assertAlmostEqual((a + c) / 2, 600.0, delta=2.0)                   # centrata sul passaggio zenitale

    def test_edges_are_threshold_crossings(self):
        k = G.KeplerJ2(circ(500, 55, 40, 10), EPOCH)
        st = G.lla_to_ecef(45.0, 10.0)[None, :]
        el_min = 5 * G.D2R
        wins = G.find_windows(k, st, el_min, 0.0, 3 * 86400.0)[0]
        self.assertGreater(len(wins), 4)
        for a, c in wins:
            e = G.elevation(k.ecef(np.array([a, (a + c) / 2, c])), st)[0]
            self.assertAlmostEqual(e[0], el_min, delta=1e-4)           # agli estremi l'elevazione vale la soglia
            self.assertAlmostEqual(e[2], el_min, delta=1e-4)
            self.assertGreater(e[1], el_min)                            # dentro e' sopra

    def test_offnadir_conversion(self):
        # un off-nadir massimo di 0 corrisponde al solo zenit (elevazione 90), e cresce al crescere dell'angolo
        self.assertAlmostEqual(G.min_elevation_from_offnadir(0.0, 500e3), 90.0, places=9)
        a, b = G.min_elevation_from_offnadir(20, 500e3), G.min_elevation_from_offnadir(40, 500e3)
        self.assertGreater(a, b)
        # coerente con la geometria: da quota h, off-nadir 20 gradi => elevazione ~ 64.6 (Terra sferica)
        self.assertAlmostEqual(a, math.degrees(math.acos((G.RE + 500e3) / G.RE * math.sin(20 * G.D2R))), places=9)


if __name__ == "__main__":
    unittest.main()
