"""Test di integrita' di ambiente, feedback, esiti e regret."""
import copy
import unittest

import _path  # noqa: F401
import numpy as np

from aeossp.config import load_config
from aeossp.env import Env, Outcomes
from aeossp.instance import build_instance
from aeossp.policies import ClassUCB, GreedyPriority, KnownExpectation, RandomFeasible, plan_by_scores, run_episode
from aeossp.regret import benchmark_plan, regret_curve
from aeossp.schedule import evaluate, profit_of

BASE = {
    "schema": "model/v1", "seed": 4, "horizon_hours": 12,
    "satellite_defaults": {"min_elevation_deg": 50, "observation_s": 2,
                           "resources": {"memory_capacity_gbit": 4.0}},
    "walker": {"sats": 2, "planes": 2, "phasing": 1, "altitude_km": 500, "inclination_deg": 55},
    "targets": {"mode": "random", "count": 120},
    "classes": {"cell_deg": 60}, "discretization": {"step_s": 30},
    "outcomes": {"model": "cell_bernoulli", "p_min": 0.1, "p_max": 0.95, "quality_min": 0.6, "quality_max": 1.0, "offnadir_penalty": 0.3},
}


def cfg(**over):
    c = copy.deepcopy(BASE)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return load_config(c)


class TestEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = build_instance(cfg())

    def test_reproducible_episode(self):
        a = run_episode(Env(self.inst), RandomFeasible(1))
        b = run_episode(Env(self.inst), RandomFeasible(1))
        self.assertEqual(a, b)
        self.assertNotEqual(a, run_episode(Env(self.inst), RandomFeasible(2)))

    def test_common_random_numbers(self):
        # la realizzazione degli esiti dipende solo da seme e task, non dalle scelte dell'algoritmo
        e1, e2 = Env(self.inst), Env(self.inst)
        np.testing.assert_array_equal(e1.outcomes.success, e2.outcomes.success)
        run_episode(e1, RandomFeasible(1))
        run_episode(e2, GreedyPriority())
        fb1 = {f.task: f for h in e1.history for f in h["feedback"]}
        fb2 = {f.task: f for h in e2.history for f in h["feedback"]}
        common = set(fb1) & set(fb2)
        self.assertGreaterEqual(len(common), 1)
        for env, fb in ((e1, fb1), (e2, fb2)):                         # ogni feedback coincide con la realizzazione dell'ambiente
            for p, f in fb.items():
                self.assertEqual(f.success, bool(env.outcomes.success[p]))
        for p in common:
            self.assertEqual(fb1[p].success, fb2[p].success)
            self.assertEqual(fb1[p].quality, fb2[p].quality)

    def test_executed_schedule_is_globally_valid(self):
        for pol in (RandomFeasible(3), GreedyPriority()):
            env = Env(self.inst)
            run_episode(env, pol)
            e = evaluate(self.inst, env.executed_schedule())
            self.assertTrue(e.valid, e.errors[:3])                       # l'unione dei piani dei round e' un piano ammissibile
            self.assertGreater(e.n_tasks, 10)

    def test_budgets_are_binding_per_round(self):
        env = Env(self.inst)
        run_episode(env, GreedyPriority())
        cap = self.inst.memory_capacity(0)
        for h in env.history:
            for k, s in enumerate(h["schedule"]):
                self.assertLessEqual(float(self.inst.memory_use(np.array(s, dtype=int)).sum()) if s else 0.0, cap + 1e-9)

    def test_step_rejects_invalid_plans(self):
        env = Env(self.inst)
        ctx = env.context()
        outside = np.flatnonzero(~ctx.mask)[0]
        with self.assertRaises(ValueError):
            env.step([[int(outside)], []])                                # non e' un braccio di questo round
        arms0 = [int(p) for p in ctx.arms if self.inst.sat[p] == 0]
        same_target = {}
        for p in arms0:
            same_target.setdefault(int(self.inst.target[p]), []).append(p)
        dup = next(v for v in same_target.values() if len(v) >= 2)
        with self.assertRaises(ValueError):
            env.step([sorted(dup[:2], key=lambda p: self.inst.t0[p]), []])     # N_i = 1 violato
        self.assertEqual(env.round, 0)                                    # un piano rifiutato non fa avanzare l'episodio

    def test_arms_respect_state(self):
        env = Env(self.inst)
        ctx = env.reset()
        seen = set()
        for _ in range(env.inst.n_cycles):
            for k, prev in enumerate(ctx.prev_last):
                if prev is not None:
                    mine = ctx.arms[self.inst.sat[ctx.arms] == k]
                    self.assertTrue(np.all(self.inst.compatible(np.full(len(mine), prev), mine)))
            self.assertTrue(np.all(ctx.remaining[self.inst.target[ctx.arms]] > 0))
            self.assertFalse(seen & set(self.inst.target[ctx.arms].tolist()))      # un target gia' tentato (N_i = 1) non torna
            res = env.step(plan_by_scores(ctx, self.inst.targets.priority[self.inst.target[ctx.arms]]))
            seen |= {h for s in env.history[-1]["schedule"] for p in s for h in [int(self.inst.target[p])]}
            ctx = env.context()
            if ctx is None:
                break

    def test_deterministic_outcomes_reward_equals_profit(self):
        inst = build_instance(cfg(outcomes={"model": "deterministic"}))
        env = Env(inst)
        run_episode(env, GreedyPriority())
        self.assertAlmostEqual(env.total_reward, profit_of(inst, env.executed_schedule()), places=9)
        self.assertTrue(all(f.success for h in env.history for f in h["feedback"]))

    def test_feedback_latency(self):
        env = Env(self.inst, latency_rounds=2)
        ctx = env.reset()
        got, executed = [], []
        r = 0
        while ctx is not None:
            res = env.step(plan_by_scores(ctx, self.inst.targets.priority[self.inst.target[ctx.arms]]))
            executed.append([p for s in env.history[-1]["schedule"] for p in s])
            got.append([f.task for f in res.feedback])
            ctx = env.context()
        late = [f.task for f in env.flush()]
        n = len(executed)
        self.assertEqual(got[0], [])
        self.assertEqual(got[1], [])
        for r in range(2, n):
            self.assertEqual(sorted(got[r]), sorted(executed[r - 2]))      # il feedback del round r-2 arriva a fine round r
        self.assertEqual(sorted(late), sorted(executed[n - 2] + executed[n - 1]))
        self.assertEqual(sum(map(len, got)) + len(late), sum(map(len, executed)))

    def test_outcome_frequencies_match_theta(self):
        inst = build_instance(cfg(horizon_hours=48, targets={"count": 400}, classes={"cell_deg": 90}, outcomes={"offnadir_penalty": 0.0}))
        oc = Outcomes(inst)
        self.assertGreater(inst.n, 3000)
        for c in range(oc.ncell):
            m = inst.cls == c
            if m.sum() > 400:
                self.assertAlmostEqual(float(oc.success[m].mean()), float(oc.theta[0, c]), delta=0.06)

    def test_change_points_switch_regime(self):
        inst = build_instance(cfg(horizon_hours=24, outcomes={"change_points": [8 * 3600.0, 16 * 3600.0], "offnadir_penalty": 0.0}))
        oc = Outcomes(inst)
        self.assertEqual(oc.theta.shape[0], 3)
        self.assertFalse(np.allclose(oc.theta[0], oc.theta[1]))
        for r, (a, b) in enumerate([(0, 8 * 3600), (8 * 3600, 16 * 3600), (16 * 3600, 24 * 3600)]):
            m = (inst.t0 >= a) & (inst.t0 < b)
            np.testing.assert_allclose(oc.p_success[m], oc.theta[r][inst.cls[m]])

    def test_uncontrolled_rounds_are_excluded_from_regret(self):
        env = Env(self.inst)
        ctx = env.reset()
        flags = []
        while ctx is not None:
            controlled = ctx.round % 2 == 0
            env.step(plan_by_scores(ctx, self.inst.targets.priority[self.inst.target[ctx.arms]]), controlled=controlled)
            flags.append(controlled)
            ctx = env.context()
        rc = regret_curve(env, "expected", method="greedy")
        inc = np.diff(np.concatenate([[0.0], rc["regret"]]))
        self.assertTrue(np.all(inc[~np.array(flags)] == 0.0))
        self.assertTrue(np.all(np.diff(np.concatenate([[0.0], rc["benchmark"]]))[~np.array(flags)] == 0.0))


class TestRegret(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = copy.deepcopy(BASE)
        c.update(horizon_hours=6, targets={"mode": "random", "count": 25}, walker={"sats": 1, "planes": 1, "altitude_km": 500, "inclination_deg": 55})
        c["satellite_defaults"]["resources"]["memory_capacity_gbit"] = 3.3
        c["discretization"] = {"step_s": 60}
        cls.inst = build_instance(load_config(c))

    def test_pseudo_regret_nonnegative_against_exact_benchmark(self):
        inst = self.inst
        env = Env(inst)
        bench = benchmark_plan(inst, env.expected_weights, method="milp")
        self.assertEqual(bench["method"], "milp")
        self.assertTrue(bench["optimal"])
        self.assertTrue(evaluate(inst, bench["plan"]).valid)
        for pol in (RandomFeasible(1), RandomFeasible(2), GreedyPriority(), KnownExpectation(env)):
            e = Env(inst)
            run_episode(e, pol)
            rc = regret_curve(e, "expected", bench=bench)
            # ogni piano eseguito e' ammissibile sull'orizzonte, quindi il suo valore atteso non supera l'ottimo globale
            self.assertGreaterEqual(rc["regret"][-1], -1e-9, type(pol).__name__)
            self.assertAlmostEqual(rc["benchmark"][-1], bench["value"], places=9)

    def test_benchmark_milp_not_worse_than_greedy(self):
        env = Env(self.inst)
        w = env.expected_weights
        g = benchmark_plan(self.inst, w, method="greedy")
        m = benchmark_plan(self.inst, w, method="milp")
        self.assertGreaterEqual(m["value"], g["value"] - 1e-9)


class TestLearning(unittest.TestCase):
    def test_class_ucb_beats_random_with_binding_capacity(self):
        # capacita' molto stretta e molti round: conviene imparare quali classi hanno successo
        inst = build_instance(cfg(horizon_hours=96, targets={"count": 600}, classes={"cell_deg": 90},
                                  satellite_defaults={"resources": {"memory_capacity_gbit": 3.3}}))
        res = {}
        for name, mk in (("random", lambda e: RandomFeasible(1)), ("greedy", lambda e: GreedyPriority()),
                         ("ucb", lambda e: ClassUCB(8, c=0.25)), ("known", lambda e: KnownExpectation(e))):
            env = Env(inst)
            run_episode(env, mk(env))
            res[name] = regret_curve(env, "expected", method="greedy")["regret"][-1]
        self.assertLess(res["known"], res["random"])
        self.assertLess(res["ucb"], res["random"])                    # imparare le classi batte il caso
        self.assertLess(res["known"], res["ucb"] + 1e-9)               # e non puo' superare chi conosce le probabilita'


if __name__ == "__main__":
    unittest.main()
