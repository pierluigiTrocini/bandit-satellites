"""Solver di riferimento sull'istanza: greedy, programmazione dinamica (limite superiore) e MILP del paper.

Tutti lavorano su *pesi per task* `w` (di default il profitto del target): per un bandit sono i reward stimati
(o realizzati, per l'oracolo a posteriori). Un piano e' una lista, per satellite, di indici di task in ordine di inizio.

  profit_cap    limite superiore banale (ogni target visibile al massimo N_i volte);
  greedy        inserimento per peso decrescente; rispetta (5), (2), (12), (13); euristica veloce, sempre ammissibile.
  dp_bound      cammino di peso massimo su ogni satellite, ignorando N_i e i budget: limite superiore (rilassamento).
  solve_milp    modello tempo-discreto del paper (8)-(13) con flusso su archi, per istanze piccole (HiGHS via scipy).
"""

import bisect
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .schedule import TOL


def default_weights(inst):
    """Peso di un task = profitto della prima osservazione del suo target."""
    return inst.targets.priority[inst.target] * inst.targets.gain[0] if inst.n else np.zeros(0)


def profit_cap(inst, weights=None):
    """Limite superiore banale: per ogni target visibile il valore massimo che puo' dare (N_i osservazioni con i
    guadagni marginali del profitto, oppure il peso massimo dei suoi task). Molto piu' stretto del rilassamento DP
    quando i target hanno molte OTW (il DP ignora N_i)."""
    if inst.n == 0:
        return 0.0
    if weights is None:
        g = inst.targets.gain
        return float(sum(inst.targets.priority[i] * g[:int(inst.targets.n_max[i])].sum() for i in np.unique(inst.target)))
    w = np.asarray(weights, dtype=float)
    tot = 0.0
    for i in np.unique(inst.target):
        v = np.sort(w[inst.target == i])[::-1]
        tot += float(v[:int(inst.targets.n_max[i])].clip(min=0).sum())
    return tot


# ---------------------------------------------------------------- greedy

def greedy(inst, weights=None, allowed=None, remaining=None, usage0=None, prev_last=None):
    """Inserimento per peso decrescente (a parita' il piu' presto). Un task si inserisce se, tra i vicini
    gia' scelti, rispetta la transizione (5) e se N_i, memoria (12) ed energia (13) restano ammissibili.

    allowed: maschera booleana dei task ammessi (default tutti); remaining: osservazioni ancora ammesse per
    target (default N_i); usage0 {(satellite, chiave): (memoria, energia)}: risorse gia' consumate; prev_last:
    ultimo task impegnato per satellite (vicino fisso a sinistra). Restituisce il piano (lista per satellite)."""
    w = default_weights(inst) if weights is None else np.asarray(weights, dtype=float)
    ok = np.ones(inst.n, bool) if allowed is None else np.asarray(allowed, bool)
    rem = (inst.targets.n_max.copy() if remaining is None else np.asarray(remaining).copy()).astype(int)
    cand = np.flatnonzero(ok & (w > 0))
    order = cand[np.lexsort((inst.t0[cand], -w[cand]))]
    ns = len(inst.sats)
    seq = [[] for _ in range(ns)]                  # task scelti per satellite, ordinati per inizio
    keys = [[] for _ in range(ns)]                 # inizi corrispondenti (per bisect)
    use = defaultdict(lambda: [0.0, 0.0])          # (satellite, chiave budget) -> [memoria, energia]
    for kk, (m, e) in (usage0 or {}).items():
        use[kk] = [m, e]
    fixed = [0] * ns
    for k in range(ns):
        if prev_last and prev_last[k] is not None:  # l'ultimo task impegnato e' un vicino fisso a sinistra
            seq[k].append(int(prev_last[k]))
            keys[k].append(inst.t0[prev_last[k]])
            fixed[k] = 1
    kb = lambda p: inst.budget_key(p)
    for p in order:
        i, k = int(inst.target[p]), int(inst.sat[p])
        if rem[i] <= 0:
            continue
        s, ks = seq[k], keys[k]
        pos = bisect.bisect_right(ks, inst.t0[p])
        a = s[pos - 1] if pos > 0 else None
        b = s[pos] if pos < len(s) else None
        if (a is not None and not inst.compatible(a, p)) or (b is not None and not inst.compatible(p, b)):
            continue
        # budget: memoria e variazione di energia sulle chiavi toccate
        key = kb(p)
        dm = float(inst.memory_use(p)[0])
        de = defaultdict(float)
        de[key] += float(inst.energy_imaging(p)[0])
        if a is not None and kb(a) == key:
            de[key] += float(inst.energy_maneuver(a, p))
        if b is not None and kb(b) == kb(p):
            de[kb(b)] += float(inst.energy_maneuver(p, b))
        if a is not None and b is not None and kb(a) == kb(b):
            de[kb(b)] -= float(inst.energy_maneuver(a, b))
        if use[key][0] + dm > inst.memory_capacity(k) + TOL:
            continue
        if any(use[kk][1] + v > inst.energy_capacity(k) + TOL for kk, v in de.items()):
            continue
        use[key][0] += dm
        for kk, v in de.items():
            use[kk][1] += v
        s.insert(pos, int(p))
        ks.insert(pos, inst.t0[p])
        rem[i] -= 1
    return [s[fixed[k]:] for k, s in enumerate(seq)]


# ---------------------------------------------------------------- limite superiore (DP)

def dp_bound(inst, weights=None, allowed=None):
    """Per ogni satellite il cammino di peso massimo nel grafo delle transizioni (5), ignorando N_i e i
    budget. E' un rilassamento: il valore e' un limite superiore dell'ottimo del problema completo, ed e'
    esatto quando non c'e' interazione fra task (target ripetuti, budget attivi). Restituisce (valore, piano)."""
    w = default_weights(inst) if weights is None else np.asarray(weights, dtype=float)
    ok = np.ones(inst.n, bool) if allowed is None else np.asarray(allowed, bool)
    total, plan = 0.0, []
    for k, (lo, hi) in enumerate(inst.sat_slice):
        ids = np.arange(lo, hi)[ok[lo:hi] & (w[lo:hi] > 0)]
        if len(ids) == 0:
            plan.append([])
            continue
        s = inst.sats[k]
        reach = s.slew.max_transition if s.agile else 0.0
        t0, t1 = inst.t0[ids], inst.t1[ids]
        n = len(ids)
        by_end = np.argsort(t1, kind="stable")            # per entrare nell'insieme dei predecessori «sempre compatibili»
        best, prev = np.zeros(n), np.full(n, -1)
        far_val, far_arg, ptr = 0.0, -1, 0                # massimo fra i task con fine + reach <= inizio corrente
        for i in range(n):
            while ptr < n and t1[by_end[ptr]] + reach <= t0[i] + 1e-9:
                j = by_end[ptr]
                if j < i and best[j] > far_val:
                    far_val, far_arg = best[j], j
                ptr += 1
            cand_v, cand_a = far_val, far_arg
            # predecessori vicini: iniziati prima, non ancora «sempre compatibili»
            lo_j = int(np.searchsorted(t0, t0[i] - (t1 - t0).max() - reach - 1e-9, "left"))
            js = np.arange(lo_j, i)
            js = js[t1[js] + reach > t0[i] + 1e-9]
            if len(js):
                m = inst.compatible(ids[js], np.full(len(js), ids[i]))
                js = js[m]
                if len(js):
                    jb = js[np.argmax(best[js])]
                    if best[jb] > cand_v:
                        cand_v, cand_a = best[jb], jb
            best[i] = w[ids[i]] + cand_v
            prev[i] = cand_a
        i = int(np.argmax(best))
        total += float(best[i])
        path = []
        while i >= 0:
            path.append(int(ids[i]))
            i = int(prev[i])
        plan.append(path[::-1])
    return total, plan


# ---------------------------------------------------------------- MILP del paper

def solve_milp(inst, weights=None, allowed=None, remaining=None, time_limit=60.0, max_arcs=300000, mip_rel_gap=1e-9):
    """Modello (8)-(13) del paper (tempo-discreto, flusso su archi) con i budget per ciclo (o per orizzonte).

    Variabili: x_p (task scelto), y_pq (arco fra task consecutivi nello stesso gruppo satellite-ciclo), y_sp / y_pe
    (inizio/fine di cammino). Se `weights` e' None l'obiettivo e' il profitto del paper (con il guadagno marginale
    decrescente tramite variabili z); altrimenti e' la somma dei pesi dei task scelti.
    Restituisce dict(plan, value, status, optimal, message). Solo per istanze piccole (centinaia di task per gruppo)."""
    ok = np.ones(inst.n, bool) if allowed is None else np.asarray(allowed, bool)
    rem = (inst.targets.n_max.copy() if remaining is None else np.asarray(remaining).copy()).astype(int)
    ids = np.flatnonzero(ok)
    m = len(ids)
    if m == 0:
        return dict(plan=[[] for _ in inst.sats], value=0.0, status=0, optimal=True, message="nessun task")
    loc = {int(p): a for a, p in enumerate(ids)}
    sat_of = inst.sat[ids]
    key = np.array([inst.budget_key(p) for p in ids])
    group = {}
    for a in range(m):
        group.setdefault((int(key[a, 0]), int(key[a, 1])), []).append(a)

    # profitto con guadagno decrescente: variabili z (solo se i guadagni non sono costanti e N_i > 1)
    g = inst.targets.gain
    zmode = weights is None and len(g) > 1 and not np.allclose(g, g[0])
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)[ids]
    elif not zmode:
        w = default_weights(inst)[ids]

    arcs = []                                                  # (a, b) indici locali
    for (k, c), mem in group.items():
        mem = sorted(mem, key=lambda a: (inst.t0[ids[a]], inst.t1[ids[a]]))
        arr = np.array(mem)
        for ii, a in enumerate(mem[:-1]):
            rest = arr[ii + 1:]
            okc = inst.compatible(np.full(len(rest), ids[a]), ids[rest])
            arcs += [(a, int(b)) for b in rest[okc]]
        if len(arcs) > max_arcs:
            raise ValueError(f"troppi archi (> {max_arcs}): istanza troppo grande per il MILP")
    na = len(arcs)
    ncol = m + na + 2 * m                                       # x | y | ys | ye
    iy, iys, iye = m, m + na, m + na + m
    tg = np.unique(inst.target[ids])
    zcol = {}
    if zmode:
        for i in tg:
            for kk in range(min(int(rem[i]), len(g))):
                zcol[(int(i), kk)] = ncol
                ncol += 1
    rows, cols, vals, lb, ub = [], [], [], [], []

    def row(entries, lo, hi):
        r = len(lb)
        for cidx, v in entries:
            rows.append(r)
            cols.append(cidx)
            vals.append(v)
        lb.append(lo)
        ub.append(hi)

    inn, out = defaultdict(list), defaultdict(list)
    for e, (a, b) in enumerate(arcs):
        out[a].append(iy + e)
        inn[b].append(iy + e)
    for a in range(m):
        row([(c, 1.0) for c in inn[a]] + [(iys + a, 1.0), (a, -1.0)], 0, 0)
        row([(c, 1.0) for c in out[a]] + [(iye + a, 1.0), (a, -1.0)], 0, 0)
    for (k, c), mem in group.items():
        row([(iys + a, 1.0) for a in mem], 0, 1)                # al piu' un cammino per gruppo
        p = ids[np.array(mem)]
        row([(a, float(x)) for a, x in zip(mem, inst.memory_use(p))], -np.inf, inst.memory_capacity(k))
        ent = [(a, float(x)) for a, x in zip(mem, inst.energy_imaging(p))]
        mset = set(mem)
        ent += [(iy + e, float(inst.energy_maneuver(ids[a], ids[b]))) for e, (a, b) in enumerate(arcs) if a in mset]
        row(ent, -np.inf, inst.energy_capacity(k))
    for i in tg:                                               # (2)/(9): al piu' N_i osservazioni (restanti)
        mem = np.flatnonzero(inst.target[ids] == i)
        row([(int(a), 1.0) for a in mem], 0, int(rem[i]))
        if zmode:
            row([(zcol[(int(i), kk)], 1.0) for kk in range(min(int(rem[i]), len(g)))] + [(int(a), -1.0) for a in mem], -np.inf, 0)
    P, Q = inst.conflicts()                                    # conflitti fra task di gruppi diversi (confine di ciclo)
    if len(P):
        sel = np.isin(P, ids) & np.isin(Q, ids)
        for p, q in zip(P[sel], Q[sel]):
            if key[loc[int(p)]].tolist() != key[loc[int(q)]].tolist():
                row([(loc[int(p)], 1.0), (loc[int(q)], 1.0)], -np.inf, 1)
    c = np.zeros(ncol)
    if zmode:
        for (i, kk), col in zcol.items():
            c[col] = -inst.targets.priority[i] * g[kk]
    else:
        c[:m] = -w
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), ncol))
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(ncol), bounds=Bounds(0, 1),
               options=dict(time_limit=time_limit, mip_rel_gap=mip_rel_gap))
    plan = [[] for _ in inst.sats]
    if res.x is not None:
        for a in np.flatnonzero(res.x[:m] > 0.5):
            plan[int(sat_of[a])].append(int(ids[a]))
        plan = [sorted(s, key=lambda p: (inst.t0[p], inst.t1[p])) for s in plan]
    return dict(plan=plan, value=float(-res.fun) if res.fun is not None else 0.0, status=int(res.status),
                optimal=bool(res.status == 0), message=str(res.message))
