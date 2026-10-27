"""Ambiente a round per gli algoritmi di online learning (bandit combinatorio con vincoli di risorsa).

Un *round* e' un ciclo orbitale (la stessa unita' dei budget del paper). A ogni round:
  1. `context()` espone i **bracci eseguibili**: i task del ciclo che (a) riguardano richieste gia' arrivate,
     (b) hanno ancora osservazioni disponibili (N_i), (c) rispettano la transizione (5) dall'ultimo task gia'
     impegnato. Ogni braccio ha un'identita' stabile (indice del task), una classe (cella) e un contesto (off-nadir, ...).
  2. L'algoritmo sceglie un **super-arm**: un piano, cioe' un insieme ordinato di task per satellite che rispetta
     (2), (5), (12), (13). Costruirlo da punteggi per task e' il «combinatorial oracle» (`solvers.greedy`/`solve_milp`).
  3. `step(piano)` esegue il piano: l'esito di ogni task (successo, qualita') e' campionato da un modello
     *nascosto all'algoritmo*, con numeri casuali comuni (dipendono solo da seme e task, non dalle scelte), cosi'
     due algoritmi sullo stesso seme vedono la stessa realizzazione ambientale.
  4. Il feedback (reward osservato) torna all'algoritmo con un ritardo di `latency_rounds` round (default 0):
     feedback ritardato senza modellare il downlink.

Il downlink e' escluso (come nel paper): la memoria e' un budget per ciclo. Il canale di emergenza non c'e' ancora
(`step(..., controlled=False)` marca un round come non attribuibile al bandit: non entra nel regret).
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .config import rng_stream
from .schedule import evaluate, normalize


# ---------------------------------------------------------------- esiti

class Outcomes:
    """Modello degli esiti (nascosto all'algoritmo). `deterministic`: ogni osservazione riesce con qualita' 1.
    `cell_bernoulli`: ogni cella ha una probabilita' di cielo sereno theta_c (con cambi bruschi programmati,
    `change_points`), attenuata da una penalita' sull'off-nadir; la qualita' e' uniforme in [q_min, q_max].
    Reward di un'osservazione riuscita = rho_i * gain[k] * qualita', con k = osservazioni riuscite del target."""

    def __init__(self, inst, cfg=None, seed=None):
        cfg = cfg or inst.cfg["outcomes"]
        seed = inst.cfg["seed"] if seed is None else seed
        self.inst, self.cfg = inst, cfg
        n = inst.n
        self.u = rng_stream(seed, "outcomes").uniform(size=n)         # un numero per task: realizzazione comune
        self.v = rng_stream(seed, "quality").uniform(size=n)
        self.cps = np.array(cfg["change_points"], dtype=float)
        ncell = int(np.ceil(180.0 / inst.cfg["cell_deg"]) * np.ceil(360.0 / inst.cfg["cell_deg"]))
        self.ncell = ncell
        if cfg["model"] == "cell_bernoulli":
            self.theta = np.stack([rng_stream(seed, "clouds", r).uniform(cfg["p_min"], cfg["p_max"], ncell)
                                   for r in range(len(self.cps) + 1)])
        else:
            self.theta = np.ones((len(self.cps) + 1, ncell))
        self.q_lo, self.q_hi = (cfg["quality_min"], cfg["quality_max"]) if cfg["model"] == "cell_bernoulli" else (1.0, 1.0)
        pen = cfg["offnadir_penalty"] if cfg["model"] == "cell_bernoulli" else 0.0
        regime = np.searchsorted(self.cps, inst.t0, side="right") if n else np.zeros(0, int)
        p = self.theta[regime, inst.cls] * (1.0 - pen * np.clip(inst.offnadir / (np.pi / 2), 0, 1)) if n else np.zeros(0)
        self.p_success = np.clip(p, 0.0, 1.0)                           # probabilita' vera di successo del task (nascosta)
        self.success = self.u < self.p_success if n else np.zeros(0, bool)
        self.quality = self.q_lo + (self.q_hi - self.q_lo) * self.v if n else np.zeros(0)
        base = inst.targets.priority[inst.target] * inst.targets.gain[0] if n else np.zeros(0)
        self.expected_weight = base * self.p_success * 0.5 * (self.q_lo + self.q_hi)     # E[reward] della prima osservazione
        self.realized_weight = base * self.success * self.quality                          # reward realizzato (prima osservazione)


# ---------------------------------------------------------------- strutture

@dataclass
class RoundContext:
    round: int
    t_start: float
    t_end: float
    arms: np.ndarray                  # indici dei task eseguibili
    mask: np.ndarray                  # maschera booleana sugli n task (arms = flatnonzero(mask))
    remaining: np.ndarray             # osservazioni ancora ammesse per target
    prev_last: list                   # ultimo task impegnato per satellite (o None)
    usage0: dict                      # risorse gia' consumate: {(sat, chiave): (memoria, energia)}
    inst: object = None

    @property
    def cls(self):
        return self.inst.cls[self.arms]

    def features(self):
        """Contesto di ogni braccio: priorita', off-nadir, elevazione, durata, classe (array allineati ad `arms`)."""
        i, a = self.inst, self.arms
        return dict(task=a, target=i.target[a], cls=i.cls[a], priority=i.targets.priority[i.target[a]], offnadir=i.offnadir[a],
                    elevation=i.elevation[a], duration=i.t1[a] - i.t0[a], sat=i.sat[a], t0=i.t0[a])


@dataclass
class Feedback:
    task: int
    target: int
    cls: int
    round: int                        # round in cui e' stata eseguita l'osservazione
    success: bool
    quality: float
    reward: float


@dataclass
class StepResult:
    reward: float
    feedback: list                    # feedback che *arriva* a fine di questo round (con il ritardo configurato)
    cost: dict                        # risorse consumate nel round, per (satellite, chiave)
    done: bool
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- ambiente

class Env:
    def __init__(self, inst, outcomes=None, latency_rounds=None):
        self.inst = inst
        self.outcomes = outcomes or Outcomes(inst)
        self.latency = inst.cfg["latency_rounds"] if latency_rounds is None else latency_rounds
        order = np.argsort(inst.cycle, kind="stable") if inst.n else np.zeros(0, int)
        bounds = np.searchsorted(inst.cycle[order], np.arange(inst.n_cycles + 1)) if inst.n else np.zeros(inst.n_cycles + 1, int)
        self._by_cycle = [order[bounds[c]:bounds[c + 1]] for c in range(inst.n_cycles)]
        self.reset()

    # privilegiati (solo per oracolo e analisi, mai per l'algoritmo)
    @property
    def expected_weights(self):
        return self.outcomes.expected_weight

    @property
    def realized_weights(self):
        return self.outcomes.realized_weight

    def reset(self):
        inst = self.inst
        self.round = 0
        self.attempts = defaultdict(int)                       # osservazioni tentate per target
        self.succ = defaultdict(int)                           # osservazioni riuscite per target
        self.usage = {}
        self.prev_last = [None] * len(inst.sats)
        self.pending = deque()                                 # (round di rilascio, Feedback)
        self.history = []
        self.total_reward = 0.0
        return self.context()

    @property
    def done(self):
        return self.round >= self.inst.n_cycles

    def context(self):
        if self.done:
            return None
        inst, r = self.inst, self.round
        t0, t1 = r * inst.cycle_s, min((r + 1) * inst.cycle_s, inst.horizon_s)
        ids = self._by_cycle[r]
        remaining = inst.targets.n_max.copy()
        for i, n in self.attempts.items():
            remaining[i] -= n
        mask = np.zeros(inst.n, bool)
        if len(ids):
            ok = (inst.targets.arrival[inst.target[ids]] <= t0 + 1e-9) & (remaining[inst.target[ids]] > 0)
            for k, prev in enumerate(self.prev_last):
                if prev is not None:
                    sel = inst.sat[ids] == k
                    ok[sel] &= inst.compatible(np.full(int(sel.sum()), prev), ids[sel])
            mask[ids[ok]] = True
        return RoundContext(round=r, t_start=t0, t_end=t1, arms=np.flatnonzero(mask), mask=mask, remaining=remaining,
                            prev_last=list(self.prev_last), usage0=dict(self.usage), inst=inst)

    def step(self, schedule, controlled=True):
        if self.done:
            raise RuntimeError("episodio concluso")
        ctx = self.context()
        sch = normalize(self.inst, schedule)
        bad = [p for s in sch for p in s if not ctx.mask[p]]
        if bad:
            raise ValueError(f"task non eseguibili in questo round (non sono bracci): {bad[:5]}")
        counts0 = {i: n for i, n in self.attempts.items()}
        ev = evaluate(self.inst, sch, usage0=self.usage, counts0=counts0, prev_last=self.prev_last)
        if not ev.valid:
            raise ValueError("piano non ammissibile: " + "; ".join(ev.errors[:3]))
        inst, oc = self.inst, self.outcomes
        executed = sorted((p for s in sch for p in s), key=lambda p: (inst.t0[p], p))
        fb, reward, expected = [], 0.0, 0.0
        for p in executed:
            i = int(inst.target[p])
            ok = bool(oc.success[p])
            k = self.succ[i]
            g = inst.targets.gain
            rew = float(inst.targets.priority[i] * g[min(k, len(g) - 1)] * oc.quality[p]) if ok else 0.0
            self.attempts[i] += 1
            if ok:
                self.succ[i] += 1
            reward += rew
            expected += float(oc.expected_weight[p])
            fb.append(Feedback(task=int(p), target=i, cls=int(inst.cls[p]), round=self.round, success=ok,
                               quality=float(oc.quality[p]) if ok else 0.0, reward=rew))
        for f in fb:
            self.pending.append((self.round + self.latency, f))
        cost = {kk: (u["memory_gbit"] - self.usage.get(kk, (0.0, 0.0))[0], u["energy_wh"] - self.usage.get(kk, (0.0, 0.0))[1])
                for kk, u in ev.usage.items() if kk[1] == (self.round if inst.budget_scope == "cycle" else 0)}
        self.usage = {kk: (u["memory_gbit"], u["energy_wh"]) for kk, u in ev.usage.items()}
        for k, s in enumerate(sch):
            if s:
                self.prev_last[k] = s[-1]
        arrived = []
        while self.pending and self.pending[0][0] <= self.round:
            arrived.append(self.pending.popleft()[1])
        self.history.append(dict(round=self.round, schedule=[list(s) for s in sch], reward=reward, expected=expected,
                                 controlled=controlled, feedback=fb, n_arms=int(len(ctx.arms))))
        self.total_reward += reward
        self.round += 1
        return StepResult(reward=reward, feedback=arrived, cost=cost, done=self.done, info=dict(n_arms=int(len(ctx.arms))))

    def flush(self):
        """Feedback ancora in volo a fine episodio (ritardo oltre l'ultimo round)."""
        out = [f for _, f in self.pending]
        self.pending.clear()
        return out

    def executed_schedule(self):
        """Piano complessivo eseguito (lista per satellite, in ordine di inizio)."""
        sch = [[] for _ in self.inst.sats]
        for h in self.history:
            for k, s in enumerate(h["schedule"]):
                sch[k] += s
        return sch
