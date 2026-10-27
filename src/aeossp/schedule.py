"""Verifica di un piano rispetto ai vincoli del paper e calcolo del profitto.

Un piano e' una lista con, per ogni satellite, gli indici dei task scelti in ordine di inizio.
Vincoli verificati (numeri del paper, Wang et al. 2020):
  - ogni task appartiene al satellite indicato e i task sono ordinati per inizio;
  - (2)/(9)  ogni target e' osservato al piu' N_i volte, contando tutti i satelliti;
  - (5)      transizione: fine(p) + DeltaT(p, q) <= inizio(q) per task consecutivi dello stesso satellite;
  - (12)     memoria: somma di M^I * d <= M^C per ciclo (o per orizzonte);
  - (13)     energia: somma di E^I * d + somma di E^M * DeltaV fra consecutivi <= E^C per ciclo (o per orizzonte).
Nel modello per ciclo la manovra fra l'ultimo task di un ciclo e il primo del successivo non e' contata
nell'energia (come nel paper, dove ogni orbita ha un task iniziale fittizio), ma il vincolo (5) vale comunque.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

TOL = 1e-7


@dataclass
class Evaluation:
    valid: bool
    errors: list
    profit: float
    n_tasks: int
    usage: dict = field(default_factory=dict)     # (satellite, ciclo) -> dict(memory_gbit, energy_wh)

    def __bool__(self):
        return self.valid


def normalize(inst, schedule):
    """Accetta lista per satellite o dizionario {satellite: [task]}; restituisce una lista di liste di int."""
    n = len(inst.sats)
    if isinstance(schedule, dict):
        return [[int(x) for x in schedule.get(k, [])] for k in range(n)]
    sch = [[int(x) for x in s] for s in schedule]
    return sch + [[] for _ in range(n - len(sch))]


def profit_of(inst, schedule):
    """Profitto totale: per ogni target rho * somma dei primi n guadagni marginali (n = osservazioni)."""
    cnt = defaultdict(int)
    for s in normalize(inst, schedule):
        for p in s:
            cnt[int(inst.target[p])] += 1
    return sum(inst.profit_of_count(i, n) for i, n in cnt.items())


def evaluate(inst, schedule, check_budgets=True, usage0=None, counts0=None, prev_last=None):
    """Verifica un piano. Per un round di un episodio si passano gli offset dello stato: `usage0`
    {(satellite, chiave): (memoria, energia)} gia' consumata, `counts0` {target: osservazioni gia' fatte},
    `prev_last` [ultimo task impegnato per satellite o None] per la transizione verso il primo task."""
    sch = normalize(inst, schedule)
    errors, usage, cnt = [], defaultdict(lambda: dict(memory_gbit=0.0, energy_wh=0.0)), defaultdict(int)
    for kk, (m, e) in (usage0 or {}).items():
        usage[kk]["memory_gbit"], usage[kk]["energy_wh"] = m, e
    for i, n in (counts0 or {}).items():
        cnt[int(i)] = int(n)
    for k, seq in enumerate(sch):
        for p in seq:
            if not 0 <= p < inst.n:
                errors.append(f"sat {k}: task {p} inesistente")
                return Evaluation(False, errors, 0.0, 0)
            if inst.sat[p] != k:
                errors.append(f"task {p} appartiene al satellite {int(inst.sat[p])}, non a {k}")
        if len(set(seq)) != len(seq):
            errors.append(f"sat {k}: task ripetuti")
        chain = ([prev_last[k]] if prev_last and prev_last[k] is not None and seq else []) + seq
        for a, b in zip(chain, chain[1:]):
            if inst.t0[b] < inst.t0[a]:
                errors.append(f"sat {k}: task non ordinati per inizio ({a} -> {b})")
            elif inst.t1[a] + float(inst.transition(a, b)) > inst.t0[b] + TOL:
                errors.append(f"sat {k}: transizione (5) violata {a} -> {b}: fine {inst.t1[a]:.1f} + DeltaT "
                              f"{float(inst.transition(a, b)):.1f} > inizio {inst.t0[b]:.1f}")
        for p in seq:
            cnt[int(inst.target[p])] += 1
            u = usage[inst.budget_key(p)]
            u["memory_gbit"] += float(inst.memory_use(p)[0])
            u["energy_wh"] += float(inst.energy_imaging(p)[0])
        for a, b in zip(chain, chain[1:]):
            if inst.budget_key(a) == inst.budget_key(b):                    # manovra dentro lo stesso budget
                usage[inst.budget_key(b)]["energy_wh"] += float(inst.energy_maneuver(a, b))
    for i, n in cnt.items():
        if n > inst.targets.n_max[i]:
            errors.append(f"target {i}: {n} osservazioni > N_i = {int(inst.targets.n_max[i])} (2)")
    if check_budgets:
        for (k, c), u in sorted(usage.items()):
            if u["memory_gbit"] > inst.memory_capacity(k) + TOL:
                errors.append(f"sat {k} ciclo {c}: memoria {u['memory_gbit']:.3f} > {inst.memory_capacity(k):.3f} Gbit (12)")
            if u["energy_wh"] > inst.energy_capacity(k) + TOL:
                errors.append(f"sat {k} ciclo {c}: energia {u['energy_wh']:.3f} > {inst.energy_capacity(k):.3f} Wh (13)")
    n_tasks = sum(len(s) for s in sch)
    return Evaluation(not errors, errors, profit_of(inst, sch), n_tasks, dict(usage))
