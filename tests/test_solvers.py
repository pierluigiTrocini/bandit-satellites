"""Test di integrita' di verifica dei piani e solver: vincoli a mano, confronto con l'enumerazione esaustiva."""
import itertools
import unittest

import _path  # noqa: F401
import numpy as np

from aeossp.config import load_config
from aeossp.instance import build_instance
from aeossp.schedule import evaluate, profit_of
from aeossp.solvers import default_weights, dp_bound, greedy, profit_cap, solve_milp
from synthetic import make


def brute(inst, weights=None, sat=0):
    """Ottimo per enumerazione di tutti i sottoinsiemi (un solo satellite), con `evaluate` come giudice."""
    lo, hi = inst.sat_slice[sat]
    ids = list(range(lo, hi))
    best, arg = 0.0, []
    for r in range(len(ids) + 1):
        for sub in itertools.combinations(ids, r):
            e = evaluate(inst, [list(sub)])
            if not e.valid:
                continue
            v = e.profit if weights is None else float(sum(weights[p] for p in sub))
            if v > best + 1e-12:
                best, arg = v, list(sub)
    return best, arg


def random_tasks(rng, n, n_targets, span=200.0, dur=2.0, sats=1):
    return [(int(rng.integers(0, sats)), int(rng.integers(0, n_targets)), t, t + dur, float(rng.uniform(0, 40)))
            for t in rng.uniform(0, span, n)]


class TestEvaluate(unittest.TestCase):
    def test_settle_only(self):
        i = make([(0, 0, 0, 2, 0.0), (0, 1, 7, 9, 0.0), (0, 2, 10.9, 13, 0.0)])
        self.assertTrue(evaluate(i, [[0, 1]]).valid)                  # 2 + 5 <= 7: ammesso al limite
        e = evaluate(i, [[1, 2]])                                      # 9 + 5 > 10.9
        self.assertFalse(e.valid)
        self.assertIn("(5)", e.errors[0])

    def test_angle_dependent_transition(self):
        # 20 gradi a 1 deg/s e 0.5 deg/s^2: theta > w^2/a = 2 -> 20/1 + 1/0.5 = 22 s di manovra + 5 di stabilizzazione = 27 s
        ok = make([(0, 0, 0, 2, 0.0), (0, 1, 29.0, 31, 20.0)])
        bad = make([(0, 0, 0, 2, 0.0), (0, 1, 28.9, 31, 20.0)])
        self.assertAlmostEqual(float(ok.transition(0, 1)), 27.0, places=9)
        self.assertTrue(evaluate(ok, [[0, 1]]).valid)
        self.assertFalse(evaluate(bad, [[0, 1]]).valid)

    def test_overlap_and_order(self):
        i = make([(0, 0, 0, 5, 0.0), (0, 1, 3, 8, 0.0)])
        self.assertFalse(evaluate(i, [[0, 1]]).valid)
        self.assertFalse(evaluate(i, [[1, 0]]).valid)                  # ordine non crescente

    def test_target_count_limit(self):
        i = make([(0, 0, 0, 2, 0.0), (0, 0, 50, 52, 0.0)])
        e = evaluate(i, [[0, 1]])
        self.assertFalse(e.valid)
        self.assertIn("(2)", e.errors[0])
        self.assertTrue(evaluate(make([(0, 0, 0, 2, 0.0), (0, 0, 50, 52, 0.0)], n_max=2), [[0, 1]]).valid)

    def test_count_across_satellites(self):
        i = make([(0, 0, 0, 2, 0.0), (1, 0, 0, 2, 0.0)], sats=2)
        self.assertFalse(evaluate(i, [[0], [1]]).valid)                # N_i vale su tutti i satelliti insieme

    def test_memory_budget(self):
        # 800 Mbps * 2 s = 1.6 Gbit a task; capacita' 3.3 Gbit: due task passano, tre no
        tasks = [(0, k, 20.0 * k, 20.0 * k + 2, 0.0) for k in range(3)]
        i = make(tasks, mem_cap=3.3)
        self.assertTrue(evaluate(i, [[0, 1]]).valid)
        e = evaluate(i, [[0, 1, 2]])
        self.assertFalse(e.valid)
        self.assertIn("(12)", e.errors[0])
        self.assertAlmostEqual(e.usage[(0, 0)]["memory_gbit"], 4.8, places=9)

    def test_memory_budget_is_per_cycle(self):
        tasks = [(0, 0, 10, 12, 0.0), (0, 1, 30, 32, 0.0), (0, 2, 110, 112, 0.0), (0, 3, 130, 132, 0.0)]
        per_cycle = make(tasks, mem_cap=3.3, cycle_s=100.0)
        self.assertTrue(evaluate(per_cycle, [[0, 1, 2, 3]]).valid)      # 2 task per ciclo = 3.2 Gbit <= 3.3
        horizon = make(tasks, mem_cap=3.3, cycle_s=100.0, scope="horizon")
        self.assertFalse(evaluate(horizon, [[0, 1, 2, 3]]).valid)       # sull'orizzonte: 6.4 > 3.3

    def test_energy_budget_counts_maneuvers(self):
        # imaging: 100 W * 2 s = 0.0556 Wh a task; manovra di 20 gradi: 22 s * 200 W = 1.222 Wh
        i = make([(0, 0, 0, 2, 0.0), (0, 1, 40, 42, 20.0)], en_cap=1.2)
        e = evaluate(i, [[0, 1]])
        self.assertFalse(e.valid)
        self.assertIn("(13)", e.errors[0])
        self.assertAlmostEqual(e.usage[(0, 0)]["energy_wh"], 2 * 100 * 2 / 3600 + 22 * 200 / 3600, places=9)
        self.assertTrue(evaluate(make([(0, 0, 0, 2, 0.0), (0, 1, 40, 42, 20.0)], en_cap=1.5), [[0, 1]]).valid)

    def test_profit_with_gain_curve(self):
        # osservazioni ripetute con guadagno marginale decrescente [1, 0.5, 0.25]: rho * (1 + 0.5) per due osservazioni
        i = make([(0, 0, 0, 2, 0.0), (0, 0, 50, 52, 0.0), (0, 0, 100, 102, 0.0)], n_max=3, gain=[1.0, 0.5, 0.25], priority=[4.0])
        self.assertAlmostEqual(profit_of(i, [[0]]), 4.0)
        self.assertAlmostEqual(profit_of(i, [[0, 1]]), 6.0)
        self.assertAlmostEqual(profit_of(i, [[0, 1, 2]]), 7.0)


class TestSolversAgainstBruteForce(unittest.TestCase):
    def test_milp_equals_brute_force(self):
        rng = np.random.default_rng(7)
        for trial in range(25):
            n = int(rng.integers(5, 13))
            tasks = random_tasks(rng, n, n_targets=int(rng.integers(3, 8)), span=float(rng.uniform(40, 120)))
            pri = rng.uniform(0.5, 3.0, 8)
            kw = dict(priority=pri, mem_cap=float(rng.choice([3.3, 6.5, 1e9])), en_cap=float(rng.choice([1.0, 1.4, 1e9])),
                      n_max=int(rng.choice([1, 2])))
            inst = make(tasks, **kw)
            opt, _ = brute(inst)
            res = solve_milp(inst)
            self.assertTrue(res["optimal"], res["message"])
            e = evaluate(inst, res["plan"])
            self.assertTrue(e.valid, (trial, e.errors))                  # il piano del MILP e' ammissibile secondo `evaluate`
            self.assertAlmostEqual(e.profit, opt, places=7, msg=f"trial {trial}")
            self.assertAlmostEqual(res["value"], opt, places=6)
            g = greedy(inst)
            eg = evaluate(inst, g)
            self.assertTrue(eg.valid, (trial, eg.errors))                # il greedy e' sempre ammissibile...
            self.assertLessEqual(eg.profit, opt + 1e-9)                  # ...e non supera l'ottimo

    def test_milp_with_weights(self):
        rng = np.random.default_rng(3)
        for trial in range(10):
            tasks = random_tasks(rng, 10, 4, span=80.0)
            inst = make(tasks, priority=rng.uniform(1, 2, 4), n_max=2)
            w = rng.uniform(0, 5, inst.n)
            opt, _ = brute(inst, weights=w)
            res = solve_milp(inst, weights=w)
            self.assertAlmostEqual(res["value"], opt, places=6, msg=f"trial {trial}")
            self.assertTrue(evaluate(inst, res["plan"]).valid)

    def test_milp_concave_gain(self):
        rng = np.random.default_rng(11)
        for trial in range(8):
            tasks = random_tasks(rng, 9, 3, span=90.0)
            inst = make(tasks, priority=rng.uniform(1, 3, 3), n_max=3, gain=[1.0, 0.6, 0.2])
            opt, _ = brute(inst)
            res = solve_milp(inst)
            self.assertAlmostEqual(evaluate(inst, res["plan"]).profit, opt, places=6, msg=f"trial {trial}")

    def test_dp_is_exact_without_interaction(self):
        # un task per target, N_i e budget non attivi: la DP coincide con l'ottimo
        rng = np.random.default_rng(5)
        for trial in range(20):
            n = int(rng.integers(5, 12))
            tasks = [(0, k, t, t + 2, float(rng.uniform(0, 30))) for k, t in enumerate(rng.uniform(0, 70, n))]
            inst = make(tasks, priority=rng.uniform(0.5, 2, n))
            opt, _ = brute(inst)
            val, plan = dp_bound(inst)
            self.assertAlmostEqual(val, opt, places=7, msg=f"trial {trial}")
            self.assertTrue(evaluate(inst, plan).valid)

    def test_dp_is_an_upper_bound(self):
        rng = np.random.default_rng(9)
        for trial in range(20):
            tasks = random_tasks(rng, 11, 3, span=100.0)
            inst = make(tasks, priority=rng.uniform(0.5, 2, 3), mem_cap=float(rng.choice([3.3, 1e9])))
            opt, _ = brute(inst)
            self.assertGreaterEqual(dp_bound(inst)[0], opt - 1e-9)

    def test_profit_cap_is_an_upper_bound(self):
        rng = np.random.default_rng(13)
        for trial in range(15):
            inst = make(random_tasks(rng, 10, 4, span=90.0), priority=rng.uniform(0.5, 2, 4), n_max=int(rng.choice([1, 2])))
            opt, _ = brute(inst)
            self.assertGreaterEqual(profit_cap(inst), opt - 1e-9)
            w = rng.uniform(0, 3, inst.n)
            self.assertGreaterEqual(profit_cap(inst, w), brute(inst, weights=w)[0] - 1e-9)

    def test_multi_satellite_coupling(self):
        # due satelliti vedono lo stesso target: va osservato una sola volta (N_i = 1), ma ciascuno ha anche un altro target
        inst = make([(0, 0, 0, 2, 0.0), (1, 0, 0, 2, 0.0), (0, 1, 20, 22, 0.0), (1, 2, 20, 22, 0.0)], sats=2, priority=[5, 2, 2])
        res = solve_milp(inst)
        e = evaluate(inst, res["plan"])
        self.assertTrue(e.valid)
        self.assertAlmostEqual(e.profit, 9.0)                          # 5 una volta + 2 + 2

    def test_cycle_boundary_transition_respected(self):
        # due task in cicli diversi ma a 3 s di distanza: la transizione (5) vale anche attraverso il confine di ciclo
        inst = make([(0, 0, 90, 92, 0.0), (0, 1, 96, 98, 0.0)], cycle_s=95.0, priority=[1, 1])
        self.assertEqual(int(inst.cycle[0]), 0)
        self.assertEqual(int(inst.cycle[1]), 1)
        res = solve_milp(inst)
        self.assertTrue(evaluate(inst, res["plan"]).valid)
        self.assertEqual(sum(map(len, res["plan"])), 1)               # 92 + 5 > 96: non si possono prendere entrambi


class TestOnGeometricInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config({
            "schema": "model/v1", "seed": 4, "horizon_hours": 6,
            "satellite_defaults": {"min_elevation_deg": 45, "observation_s": 2},
            "satellites": [{"name": "s1", "orbit": {"type": "keplerian", "altitude_km": 500, "inclination_deg": 55}}],
            "targets": {"mode": "random", "count": 25}, "discretization": {"step_s": 60}})
        cls.inst = build_instance(cfg)

    def test_greedy_valid_and_milp_between_greedy_and_bound(self):
        i = self.inst
        self.assertGreater(i.n, 20)
        g = greedy(i)
        eg = evaluate(i, g)
        self.assertTrue(eg.valid, eg.errors)
        res = solve_milp(i, time_limit=60)
        self.assertTrue(res["optimal"], res["message"])
        em = evaluate(i, res["plan"])
        self.assertTrue(em.valid, em.errors)
        self.assertGreaterEqual(em.profit, eg.profit - 1e-9)
        cap = float(i.targets.priority[np.unique(i.target)].sum() * i.targets.gain[0])     # tutti i target visibili una volta
        self.assertLessEqual(em.profit, cap + 1e-9)
        self.assertLessEqual(em.profit, dp_bound(i)[0] + 1e-9)

    def test_allowed_mask_is_respected(self):
        i = self.inst
        mask = (i.t0 < 3 * 3600.0)
        for plan in (greedy(i, allowed=mask), solve_milp(i, allowed=mask)["plan"]):
            self.assertTrue(all(mask[p] for s in plan for p in s))


if __name__ == "__main__":
    unittest.main()
