"""Test di integrita' dell'esportazione verso il formato di traccia v2 del viewer."""
import json
import unittest

import _path  # noqa: F401
import numpy as np

from aeossp.config import load_config
from aeossp.env import Env
from aeossp.instance import build_instance
from aeossp.policies import GreedyPriority, run_episode
from aeossp.trace import to_trace


class TestTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(_path.ROOT / "configs" / "aeossp.example.json")
        cls.inst = build_instance(cfg)
        cls.env = Env(cls.inst)
        run_episode(cls.env, GreedyPriority())
        cls.tr = to_trace(cls.env, policy="test", track_dt=10.0)

    def test_top_level_keys(self):
        tr = self.tr
        self.assertEqual(tr["version"], 2)
        for k in ("meta", "stations", "targets", "satellites", "targets_removed"):
            self.assertIn(k, tr)
        self.assertEqual(tr["stations"], [])                         # niente stazioni: downlink escluso
        self.assertEqual(len(tr["targets"]), len(self.inst.targets))
        json.dumps(tr)                                               # serializzabile

    def test_track_and_windows(self):
        H = self.inst.horizon_s
        for s in self.tr["satellites"]:
            n = len(s["track"]["lat"])
            self.assertEqual(n, int(H / 10.0) + 1)
            self.assertEqual(len(s["track"]["lon"]), n)
            self.assertTrue(all(-90 <= x <= 90 for x in s["track"]["lat"]))
            self.assertTrue(all(w["t0"] < w["t1"] for w in s["windows"]))
            self.assertGreater(len(s["windows"]), 10)

    def test_rounds_are_contiguous(self):
        H = self.inst.horizon_s
        for s in self.tr["satellites"]:
            R = s["rounds"]
            self.assertAlmostEqual(R[0]["t0"], 0.0)
            self.assertAlmostEqual(R[-1]["t1"], H, places=2)
            for a, b in zip(R, R[1:]):
                self.assertAlmostEqual(a["t1"], b["t0"], places=2)            # nessun buco e nessuna sovrapposizione
            self.assertTrue(all(r["t1"] > r["t0"] for r in R))

    def test_images_match_execution(self):
        inst, env, oc = self.inst, self.env, self.env.outcomes
        for k, s in enumerate(self.tr["satellites"]):
            mine = [p for p in sum((h["schedule"][k] for h in env.history), [])]
            img_rounds = [r for r in s["rounds"] if r["action"] == "image"]
            self.assertEqual(len(img_rounds), len(mine))                         # un segmento image per osservazione eseguita
            ok = [p for p in mine if oc.success[p]]
            self.assertEqual(len(s["imaged"]), len(ok))
            self.assertEqual(s["rounds"][-1]["imaged"], len(ok))
            self.assertEqual([e["target"] for e in s["imaged"]], [int(inst.target[p]) for p in sorted(ok, key=lambda p: inst.t0[p])])
            for r in img_rounds:
                self.assertEqual(("why" in r), (r["reward"] == 0.0))              # fallita <=> motivo presente e reward nullo

    def test_cumulative_reward_matches_env(self):
        total = sum(s["rounds"][-1]["cum_reward"] for s in self.tr["satellites"])
        self.assertAlmostEqual(total, self.env.total_reward, places=2)

    def test_state_bars_in_range(self):
        for s in self.tr["satellites"]:
            cap = s["meta"]["storage_capacity"]
            for r in s["rounds"]:
                self.assertLessEqual(r["storage"], cap + 1e-6)
                self.assertGreaterEqual(r["battery"], 0.0)
                self.assertLessEqual(r["battery"], 1.0)
            # all'inizio di ogni ciclo i budget ripartono: i segmenti idle a cavallo del confine sono divisi
            c = self.inst.cycle_s
            for r in s["rounds"]:
                if r["action"] == "idle":                                           # (un'osservazione puo' sconfinare di pochi secondi)
                    self.assertEqual(int((r["t0"] + 1e-2) // c), int((r["t1"] - 1e-2) // c))


if __name__ == "__main__":
    unittest.main()
