"""Politiche di riferimento e interfaccia per gli algoritmi di online learning.

Un algoritmo implementa due metodi:
    select(ctx) -> piano          sceglie il super-arm del round (lista, per satellite, di indici di task)
    update(feedback)              riceve i `Feedback` che arrivano (con il ritardo configurato)
Quasi tutti gli algoritmi combinatori si riducono a: assegnare un punteggio a ogni braccio (stima o indice di
confidenza del reward atteso) e chiamare il *combinatorial oracle* `plan_by_scores`, che trova il piano ammissibile di
punteggio massimo (euristica greedy veloce, oppure MILP esatto sui round piccoli).

Qui ci sono solo politiche di riferimento per collaudare l'infrastruttura: NON sono gli algoritmi della tesi
(OMD/EXP3 e Thompson Sampling), che andranno implementati su questa stessa interfaccia.
"""

import numpy as np

from .config import rng_stream
from .solvers import greedy, solve_milp


def plan_by_scores(ctx, scores, exact=False, time_limit=30.0):
    """Piano ammissibile di punteggio massimo fra i bracci del round. `scores`: array sugli n task (o sugli
    `arms`, allineato). exact=True usa il MILP (solo con budget per ciclo e round piccoli)."""
    inst = ctx.inst
    w = np.zeros(inst.n)
    s = np.asarray(scores, dtype=float)
    w[ctx.arms] = s if len(s) == len(ctx.arms) else s[ctx.arms]
    if exact:
        if inst.budget_scope != "cycle":
            raise ValueError("MILP per round: supportato solo con budget per ciclo")
        res = solve_milp(inst, weights=w, allowed=ctx.mask, remaining=ctx.remaining, time_limit=time_limit)
        return res["plan"]
    return greedy(inst, weights=w, allowed=ctx.mask, remaining=ctx.remaining, usage0=ctx.usage0, prev_last=ctx.prev_last)


class Policy:
    def select(self, ctx):
        raise NotImplementedError

    def update(self, feedback):
        pass


class RandomFeasible(Policy):
    """Punteggi casuali: un piano ammissibile a caso (greedy su pesi random)."""

    def __init__(self, seed=0):
        self.rng = rng_stream(seed, "policy")

    def select(self, ctx):
        return plan_by_scores(ctx, self.rng.uniform(size=len(ctx.arms)) + 1e-9)


class GreedyPriority(Policy):
    """Massimizza la priorita' nota rho_i, senza apprendere nulla sugli esiti."""

    def select(self, ctx):
        f = ctx.features()
        return plan_by_scores(ctx, f["priority"])


class KnownExpectation(Policy):
    """Privilegiata: conosce i reward attesi veri (limite per i round, non apprende)."""

    def __init__(self, env):
        self.env = env

    def select(self, ctx):
        return plan_by_scores(ctx, self.env.expected_weights[ctx.arms])


class ClassUCB(Policy):
    """UCB1 sui reward di classe (cella): collaudo del ciclo select/update, *non* l'algoritmo della tesi.
    Punteggio di un braccio = priorita' * min(1, media di classe + bonus di esplorazione)."""

    def __init__(self, n_classes, c=1.0):
        self.n, self.s, self.t, self.c = np.zeros(n_classes), np.zeros(n_classes), 0, c

    def select(self, ctx):
        f = ctx.features()
        cl = f["cls"]
        mean = np.where(self.n[cl] > 0, self.s[cl] / np.maximum(self.n[cl], 1), 1.0)
        bonus = np.where(self.n[cl] > 0, self.c * np.sqrt(2 * np.log(max(self.t, 2)) / np.maximum(self.n[cl], 1)), 1.0)
        return plan_by_scores(ctx, f["priority"] * np.minimum(1.0, mean + bonus))

    def update(self, feedback):
        for fb in feedback:
            self.n[fb.cls] += 1
            self.s[fb.cls] += fb.success * fb.quality          # reward normalizzato dalla priorita'
            self.t += 1


def run_episode(env, policy):
    """Esegue un episodio completo. Restituisce il reward totale."""
    ctx = env.reset()
    while ctx is not None:
        res = env.step(policy.select(ctx))
        policy.update(res.feedback)
        ctx = env.context()
    policy.update(env.flush())
    return env.total_reward
