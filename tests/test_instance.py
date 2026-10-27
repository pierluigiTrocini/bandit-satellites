"""Test di integrita' di configurazione, modello di transizione e istanza."""
import copy
import itertools
import math
import unittest

import _path  # noqa: F401
import numpy as np

from aeossp import geometry as G
from aeossp.config import ConfigError, load_config
from aeossp.instance import build_instance
from aeossp.model import SlewModel

D2R = math.pi / 180


def base_cfg(**over):
    cfg = {
        "schema": "model/v1", "seed": 4, "horizon_hours": 6, "downlink": "none",
        "satellite_defaults": {"agility": "agile", "observation_s": 2, "min_elevation_deg": 50},
        "walker": {"sats": 2, "planes": 2, "phasing": 1, "altitude_km": 500, "inclination_deg": 55},
        "targets": {"mode": "random", "count": 60, "n_max": 1},
        "discretization": {"step_s": 30},
    }
    cfg.update(over)
    return cfg


class TestSlew(unittest.TestCase):
    def setUp(self):
        self.s = SlewModel(max_rate=1.0 * D2R, max_accel=0.5 * D2R, settle_s=5.0)

    def test_zero_and_analytic(self):
        self.assertEqual(float(self.s.delta_v(0.0)), 0.0)
        th = 0.5 * D2R                                            # manovra triangolare: 2*sqrt(theta/alpha)
        self.assertAlmostEqual(float(self.s.delta_v(th)), 2 * math.sqrt(th / self.s.max_accel), places=12)
        th = 90 * D2R                                             # trapezoidale: theta/omega + omega/alpha
        self.assertAlmostEqual(float(self.s.delta_v(th)), th / self.s.max_rate + self.s.max_rate / self.s.max_accel, places=12)

    def test_continuity_monotone_subadditive(self):
        w, a = self.s.max_rate, self.s.max_accel
        tc = w * w / a                                            # angolo di raccordo
        self.assertAlmostEqual(float(self.s.delta_v(tc - 1e-9)), float(self.s.delta_v(tc + 1e-9)), places=6)
        x = np.linspace(0, math.pi, 400)
        d = self.s.delta_v(x)
        self.assertTrue(np.all(np.diff(d) >= -1e-12))             # monotona
        rng = np.random.default_rng(0)
        a1, a2 = rng.uniform(0, math.pi / 2, 500), rng.uniform(0, math.pi / 2, 500)
        self.assertTrue(np.all(self.s.delta_v(a1 + a2) <= self.s.delta_v(a1) + self.s.delta_v(a2) + 1e-9))   # subadditiva


class TestConfig(unittest.TestCase):
    def test_example_loads(self):
        cfg = load_config(_path.ROOT / "configs" / "aeossp.example.json")
        self.assertEqual(len(cfg["sats"]), 2)
        self.assertEqual(cfg["downlink"], "none")

    def test_errors_are_collected(self):
        bad = base_cfg(foo=1, horizon_hours=-3)
        bad["targets"]["mode"] = "boh"
        with self.assertRaises(ConfigError) as cm:
            load_config(bad)
        txt = " | ".join(cm.exception.errors)
        self.assertIn("foo", txt)
        self.assertIn("horizon_hours", txt)
        self.assertIn("targets.mode", txt)

    def test_downlink_stations_is_reserved(self):
        with self.assertRaises(ConfigError) as cm:
            load_config(base_cfg(downlink="stations"))
        self.assertIn("riservato", " ".join(cm.exception.errors))

    def test_offnadir_and_elevation_exclusive(self):
        c = base_cfg()
        c["satellite_defaults"]["max_offnadir_deg"] = 30
        with self.assertRaises(ConfigError):
            load_config(c)

    def test_walker_geometry(self):
        cfg = load_config(base_cfg(walker={"sats": 6, "planes": 3, "phasing": 1, "altitude_km": 500, "inclination_deg": 55}))
        raans = sorted({round(math.degrees(s.orbit.raan)) for s in cfg["sats"]})
        self.assertEqual(raans, [0, 120, 240])
        self.assertEqual(len(cfg["sats"]), 6)


class TestInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(base_cfg())
        cls.inst = build_instance(cls.cfg)

    def test_reproducible_and_seed_sensitive(self):
        again = build_instance(load_config(base_cfg()))
        np.testing.assert_array_equal(self.inst.t0, again.t0)
        np.testing.assert_array_equal(self.inst.target, again.target)
        other = build_instance(load_config(base_cfg(seed=5)))
        self.assertNotEqual(len(other.t0), 0)
        self.assertFalse(len(other.t0) == len(self.inst.t0) and np.array_equal(other.target, self.inst.target))

    def test_targets_independent_of_satellites(self):
        # i flussi casuali dei target non dipendono dai satelliti: stessi target con 1 o 2 satelliti
        one = build_instance(load_config(base_cfg(walker={"sats": 1, "planes": 1, "altitude_km": 500, "inclination_deg": 55})))
        np.testing.assert_array_equal(one.targets.lat, self.inst.targets.lat)
        np.testing.assert_array_equal(one.targets.priority, self.inst.targets.priority)

    def test_tasks_inside_visibility(self):
        i = self.inst
        self.assertGreater(i.n, 100)
        el_min = np.array([s.min_elevation_deg for s in i.sats])[i.sat] * D2R
        self.assertTrue(np.all(i.elevation >= el_min - 1e-3))                        # visibili al centro dell'osservazione
        self.assertTrue(np.all(i.t0 >= i.vtw_start - 1e-9) and np.all(i.t1 <= i.vtw_end + 1e-9))
        np.testing.assert_allclose(i.t1 - i.t0, 2.0)                                  # durata d fissa (agile)

    def test_ordering_and_slices(self):
        i = self.inst
        for k, (lo, hi) in enumerate(i.sat_slice):
            self.assertTrue(np.all(i.sat[lo:hi] == k))
            self.assertTrue(np.all(np.diff(i.t0[lo:hi]) >= 0))
        self.assertEqual(sum(hi - lo for lo, hi in i.sat_slice), i.n)

    def test_unit_vectors_and_offnadir(self):
        i = self.inst
        np.testing.assert_allclose(np.linalg.norm(i.los0, axis=1), 1.0, atol=1e-9)
        # off-nadir coerente con la quota: sin(eta) = RE/(RE+h) * cos(el)
        h = np.array([s.orbit.a for s in i.sats])[i.sat]
        np.testing.assert_allclose(np.sin(i.offnadir), G.RE / h * np.cos(i.elevation), atol=2e-3)

    def test_transition_matches_formula(self):
        i = self.inst
        lo, hi = i.sat_slice[0]
        p, q = np.arange(lo, lo + 40), np.arange(lo + 40, lo + 80)
        ang = np.arccos(np.clip(np.sum(i.los1[p] * i.los0[q], axis=1), -1, 1))
        np.testing.assert_allclose(i.transition(p, q), i.sats[0].slew.transition(ang))
        self.assertTrue(np.all(i.transition(p, q) >= i.sats[0].slew.settle_s))

    def test_conflicts_consistent_with_compatible(self):
        i = self.inst
        P, Q = i.conflicts()
        self.assertGreater(len(P), 0)
        self.assertFalse(np.any(i.compatible(P, Q)))                                 # ogni coppia elencata e' davvero in conflitto
        # e viceversa, a campione: nessuna coppia in conflitto manca (stesso satellite, vicine nel tempo)
        lo, hi = i.sat_slice[0]
        have = set(zip(P.tolist(), Q.tolist()))
        rng = np.random.default_rng(1)
        for p in rng.integers(lo, hi - 30, 150):
            for q in range(p + 1, min(hi, p + 30)):
                if not i.compatible(p, q):
                    self.assertIn((int(p), int(q)), have)

    def test_cycle_and_class_ranges(self):
        i = self.inst
        self.assertEqual(i.n_cycles, math.ceil(6 * 3600 / i.cycle_s))
        self.assertTrue(np.all((i.cycle >= 0) & (i.cycle < i.n_cycles)))
        self.assertTrue(np.all(i.cls == i.targets.cls[i.target]))
        self.assertTrue(np.all(i.targets.cls < (180 // 30) * (360 // 30)))

    def test_stream_respects_arrival_and_expiry(self):
        cfg = load_config(base_cfg(targets={"mode": "stream", "initial_count": 10, "arrival": {"type": "poisson", "rate_per_hour": 20},
                                            "validity_hours": 1.5}))
        inst = build_instance(cfg)
        T = inst.targets
        self.assertGreater(len(T), 20)
        self.assertTrue(np.all(inst.t0 >= T.arrival[inst.target] - 1e-9))
        self.assertTrue(np.all(inst.t1 <= T.expires[inst.target] + 1e-9))
        self.assertTrue(np.all(np.diff(T.arrival[10:]) >= 0))

    def test_ceos_has_no_transitions(self):
        c = base_cfg()
        c["satellite_defaults"]["agility"] = "none"
        inst = build_instance(load_config(c))
        self.assertGreater(inst.n, 20)
        self.assertEqual(float(inst.transition(0, 1)), 0.0)
        # una sola OTW per VTW, uguale all'intera finestra
        self.assertEqual(len(np.unique(inst.vtw)), inst.n)
        np.testing.assert_allclose(inst.t0, inst.vtw_start)
        np.testing.assert_allclose(inst.t1, inst.vtw_end)

    def test_budget_scope_horizon_key(self):
        c = base_cfg(budgets={"scope": "horizon"})
        inst = build_instance(load_config(c))
        self.assertTrue(all(inst.budget_key(p)[1] == 0 for p in range(0, inst.n, 17)))


if __name__ == "__main__":
    unittest.main()
