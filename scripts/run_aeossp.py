"""Esegue un episodio del modello astratto AEOSSP (src/aeossp) e scrive la traccia per il viewer.

Uso: PYTHONPATH=src uv run python scripts/run_aeossp.py [--config configs/aeossp.example.json] [--policy ucb]
     [--regret] [--out outputs/traces]

Politiche (solo di riferimento, NON gli algoritmi della tesi): random, greedy, known, ucb.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeossp.config import ConfigError, load_config          # noqa: E402
from aeossp.env import Env                                  # noqa: E402
from aeossp.instance import build_instance                  # noqa: E402
from aeossp.policies import ClassUCB, GreedyPriority, KnownExpectation, RandomFeasible, run_episode   # noqa: E402
from aeossp.regret import regret_curve                      # noqa: E402
from aeossp.schedule import evaluate                        # noqa: E402
from aeossp.trace import to_trace                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "aeossp.example.json"))
    ap.add_argument("--policy", choices=["random", "greedy", "known", "ucb"], default="greedy")
    ap.add_argument("--seed", type=int, default=None, help="default: seed del file")
    ap.add_argument("--regret", action="store_true", help="calcola anche il pseudo-regret (puo' richiedere un MILP)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "traces"))
    a = ap.parse_args()
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        sys.exit("configurazione non valida:\n  " + "\n  ".join(e.errors))
    if a.seed is not None:
        cfg["seed"] = a.seed
    t0 = time.perf_counter()
    inst = build_instance(cfg)
    print(f"istanza: {inst.summary()} ({time.perf_counter() - t0:.1f} s)")
    env = Env(inst)
    pol = {"random": lambda: RandomFeasible(cfg["seed"]), "greedy": GreedyPriority, "known": lambda: KnownExpectation(env),
           "ucb": lambda: ClassUCB(env.outcomes.ncell, c=0.25)}[a.policy]()
    t0 = time.perf_counter()
    total = run_episode(env, pol)
    ev = evaluate(inst, env.executed_schedule())
    print(f"politica {a.policy}: reward {total:.3f}, osservazioni {ev.n_tasks}, piano ammissibile: {ev.valid} ({time.perf_counter() - t0:.1f} s)")
    if not ev.valid:
        print("  ", ev.errors[:3])
    if a.regret:
        rc = regret_curve(env, "expected")
        print(f"pseudo-regret {rc['regret'][-1]:.3f} su benchmark {rc['benchmark'][-1]:.3f} ({rc['bench_info']['method']}, ottimo: {rc['bench_info']['optimal']})")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"trace_aeossp_{Path(a.config).stem}_{a.policy}_seed{cfg['seed']}.json"
    path.write_text(json.dumps(to_trace(env, policy=f"{a.policy} (riferimento, non un bandit della tesi)"), separators=(",", ":")))
    print(f"traccia: {path} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
