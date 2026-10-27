"""Regret: confronto fra il reward di un algoritmo e quello di un benchmark.

Il benchmark e' un piano *ottimale (o best-known) su tutto l'orizzonte* calcolato con i pesi veri:
  - pesi attesi (`env.expected_weights`) -> pseudo-regret: R_T = sum_t (W*_t - W(a_t)) con W i reward attesi;
  - pesi realizzati (`env.realized_weights`) -> regret a posteriori (oracolo che conosce anche le realizzazioni).
Il benchmark e' *chiaroveggente*: vede in anticipo l'intero orizzonte (anche le richieste che arriveranno) e non
subisce il vincolo di causalita' fra round. Il regret misura quindi la distanza da un limite superiore, non da
«la migliore politica online». Il benchmark usa il MILP del paper se l'istanza e' piccola, altrimenti il greedy
(limite inferiore dell'ottimo: in quel caso il regret puo' risultare negativo per politiche molto buone).

Solo i round `controlled=True` (decisi dal bandit) contano: i round del canale di emergenza non sono attribuibili.
"""

import numpy as np

from .schedule import evaluate
from .solvers import greedy, solve_milp


def benchmark_plan(inst, weights, method="auto", time_limit=60.0, max_tasks_milp=800):
    """Piano benchmark per i pesi dati. method: 'greedy' | 'milp' | 'auto' (MILP se abbastanza piccolo e ottimo,
    altrimenti greedy). Restituisce dict(plan, value, method, optimal)."""
    g = greedy(inst, weights=weights)
    vg = float(sum(weights[p] for s in g for p in s))
    if method == "greedy" or (method == "auto" and int((weights > 0).sum()) > max_tasks_milp):
        return dict(plan=g, value=vg, method="greedy", optimal=False)
    try:
        res = solve_milp(inst, weights=weights, time_limit=time_limit)
    except ValueError:
        return dict(plan=g, value=vg, method="greedy", optimal=False)
    if res["plan"] and res["value"] >= vg - 1e-9 and evaluate(inst, res["plan"]).valid:
        return dict(plan=res["plan"], value=res["value"], method="milp", optimal=res["optimal"])
    return dict(plan=g, value=vg, method="greedy", optimal=False)


def per_round(inst, plan, weights):
    """Somma dei pesi del piano in ciascun round (ciclo)."""
    out = np.zeros(inst.n_cycles)
    for s in plan:
        for p in s:
            out[inst.cycle[p]] += weights[p]
    return out


def regret_curve(env, kind="expected", method="auto", bench=None):
    """Curve cumulate sull'episodio eseguito. kind: 'expected' (pseudo-regret) o 'realized'.
    Restituisce dict(regret, benchmark, achieved, bench_info), array cumulati sui soli round controllati."""
    inst = env.inst
    w = env.expected_weights if kind == "expected" else env.realized_weights
    bench = bench or benchmark_plan(inst, w, method=method)
    star = per_round(inst, bench["plan"], w)
    done = len(env.history)
    got = np.array([sum(w[p] for s in h["schedule"] for p in s) for h in env.history])
    ctl = np.array([h["controlled"] for h in env.history], dtype=bool)
    star, got = star[:done] * ctl, got * ctl
    return dict(regret=np.cumsum(star - got), benchmark=np.cumsum(star), achieved=np.cumsum(got), bench_info=bench)
