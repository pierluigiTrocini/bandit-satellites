"""Segmento di terra e piano caricato al contatto (modello B).

Con `comms.uplink: at_ground_contact` le decisioni si prendono *a terra, a lotti*: quando un satellite entra in
contatto con una stazione che ha il ruolo `uplink`, il pianificatore sceglie un insieme ordinato di attivita'
(immagini e downlink) per l'intervallo fino al contatto di uplink successivo, e il piano sale al satellite, che lo
esegue da solo, senza scelte, fino al contatto dopo. Cio' che il satellite sa di una richiesta e' quello ricevuto
a un contatto precedente.

Il pianificatore ha interfaccia `plan(ctx) -> lista di attivita'`; `DemoPlanner` e' dimostrativo (NON e' un
bandit: massimizza la priorita' con una programmazione a intervalli pesata) e sara' sostituibile dal bandit.
Gli archi (arms) hanno identita' stabile: l'id della richiesta.

Ipotesi dichiarate (limiti del modello):
  * lo stato del satellite visto a terra e' quello reale all'istante della pianificazione (telemetria ideale);
  * a terra si sa che un'immagine e' stata fatta solo quando e' stata scaricata (registro FIFO dei dati);
  * la protezione di bordo (ricarica con batteria bassa) agisce anche contro il piano; non e' una scelta del piano.
"""

import bisect
from collections import deque

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from bsk_rl.gym import NO_ACTION

MAX_WINDOW = 3600.0   # [s] durata massima ipotizzata di una finestra di accesso (passaggio LEO)
SLEW_LEAD = 90.0      # [s] anticipo con cui il satellite inizia a puntare un'immagine pianificata
MIN_ARM_S = 20.0      # [s] finestre piu' brevi non sono proposte come bracci


# ------------------------------------------------------------------ finestre di accesso

class WindowEngine:
    """Calcolo delle finestre di accesso a campionamento denso, al posto di quello di BSK-RL negli episodi da file.

    Motivo: `AccessSatellite.calculate_additional_windows` cerca i passaggi dentro «finestre candidate» (campioni a
    distanza inferiore a una soglia) con una ricerca del massimo che fallisce quando la finestra candidata e'
    molto piu' lunga del passaggio, cioe' con elevazione minima bassa e quota alta. Misurato su Sentinel-2
    (786 km, stazioni a 5 gradi): trova 2 passaggi su 24 per un satellite e 0 su 24 per l'altro, mentre con le
    stazioni di default (10 gradi, 500 km) li trova tutti.
    Qui l'elevazione si campiona ogni STEP secondi e gli estremi si raffinano con una ricerca di zeri.
    Le finestre vanno in `sat.opportunities` con `_add_window` (stesso formato e stessa fusione di BSK-RL), a blocchi
    di CHUNK secondi: una finestra che attraversa il confine fra blocchi si ricompone perche' i due pezzi coincidono."""

    STEP, CHUNK, GROUP = 5.0, 6000.0, 400

    def __init__(self, sat, horizon):
        self.sat, self.horizon, self.locs, self.done_to = sat, horizon, [], 0.0

    def take_over(self):
        """Prende in carico i luoghi registrati da BSK-RL (stazioni e target dello scenario) e gli toglie il calcolo."""
        self.locs = list(self.sat.locations_for_access_checking)
        self.sat.locations_for_access_checking = []
        self.sat.opportunities = []
        self.done_to = 0.0

    def add(self, targets, t, minel):
        new = [dict(object=tg, r_LP_P=tg.r_LP_P, min_elev=minel, type="target", start_time=t) for tg in targets]
        self.locs += new
        if self.done_to > t:
            self._compute(new, t, self.done_to)

    def remove(self, targets):
        gone = {id(t) for t in targets}
        self.locs = [l for l in self.locs if id(l["object"]) not in gone]

    def extend_to(self, t):
        t = min(t, self.horizon)
        while self.done_to < t:
            a = self.done_to
            b = min(a + self.CHUNK, self.horizon)
            self._compute(self.locs, a, b)
            self.done_to = b

    def _compute(self, locs, a, b):
        if b <= a or not locs:
            return
        sat = self.sat
        interp = self._interp(a, b)
        ts = np.append(np.arange(a, b, self.STEP), b)
        pos = interp(ts)
        for g in range(0, len(locs), self.GROUP):
            group = locs[g:g + self.GROUP]
            rt = np.array([l["r_LP_P"] for l in group])                      # (m, 3)
            d = pos[None, :, :] - rt[:, None, :]
            sin_el = np.einsum("mnk,mk->mn", d, rt) / (np.linalg.norm(d, axis=2) * np.linalg.norm(rt, axis=1)[:, None])
            el = np.arcsin(np.clip(sin_el, -1.0, 1.0))
            for m, loc in enumerate(group):
                self._windows(sat, loc, interp, ts, el[m] >= loc["min_elev"], a)

    def _interp(self, a, b):
        tr = self.sat.trajectory
        tr.extend_to(b + 3 * tr.dt)
        ts = np.asarray(tr.times)
        lo, hi = max(0, np.searchsorted(ts, a) - 4), np.searchsorted(ts, b) + 5
        pos = [np.matmul(d, r) for d, r in zip(tr._J20002Pfix_log[lo:hi], tr._r_BN_N_log[lo:hi])]
        return interp1d(ts[lo:hi], pos, kind="cubic", axis=0, fill_value="extrapolate")

    def _windows(self, sat, loc, interp, ts, ok, a):
        i0 = int(np.searchsorted(ts, loc["start_time"], side="left"))        # niente prima di start_time
        if i0 >= len(ts) - 1:
            return
        ok, ts = ok[i0:], ts[i0:]
        rt, minel = loc["r_LP_P"], loc["min_elev"]

        def f(t):
            r = interp(t) - rt
            return np.arcsin(np.clip(np.dot(r, rt) / (np.linalg.norm(r) * np.linalg.norm(rt)), -1.0, 1.0)) - minel

        edges = np.flatnonzero(ok[1:] != ok[:-1])
        opens = [ts[0]] if ok[0] else []
        closes = []
        for j in edges:
            root = brentq(f, ts[j], ts[j + 1], xtol=1e-3)
            (opens if ok[j + 1] else closes).append(root)
        if ok[-1]:
            closes.append(ts[-1])
        for o, c in zip(opens, closes):
            if c - o > 0.1:
                sat._add_window(loc["object"], (float(o), float(c)), type=loc["type"], r_LP_P=rt, merge_time=ts[0])


def engine(sat):
    return sat._engine


def register_targets(sat, targets, t):
    """Aggiunge target a episodio in corso (a partire da t); le finestre si calcolano subito fino al punto gia' coperto."""
    engine(sat).add(targets, t, sat.sat_args["imageTargetMinimumElevation"])


def forget_targets(sat, targets):
    """Toglie i target dal calcolo delle finestre (il costo e' proporzionale ai luoghi registrati)."""
    engine(sat).remove(targets)


def ensure_windows(sat, t):
    engine(sat).extend_to(t)


def windows_between(sat, t_from, t_to, types=None):
    """Opportunita' (ordinate per chiusura) con chiusura >= t_from e apertura < t_to."""
    ops = sat.opportunities
    close = lambda x: x["window"][1]
    lo = bisect.bisect_left(ops, t_from, key=close)
    hi = bisect.bisect_right(ops, t_to + MAX_WINDOW, key=close)
    return [o for o in ops[lo:hi] if o["window"][0] < t_to and (types is None or o["type"] == types)]


# ------------------------------------------------------------------ pianificatore

class DemoPlanner:
    """Pianificatore dimostrativo (non e' un bandit): ogni richiesta una sola volta (la finestra piu' vicina),
    nessuna immagine sovrapposta a un downlink, e fra le immagini restanti l'insieme compatibile di priorita'
    massima (programmazione a intervalli pesata, ogni immagine occupa `service_s` dall'apertura della finestra)."""

    service_s = 90.0

    def plan(self, ctx):
        tasks = [dict(type="downlink", station=d["station"], t0=d["t0"], t1=d["t1"]) for d in ctx["downlinks"]]
        first = {}
        for a in sorted(ctx["arms"], key=lambda a: a["t0"]):
            first.setdefault(a["rid"], a)
        cand = [a for a in first.values()
                if not any(a["t0"] < d["t1"] and d["t0"] < a["t0"] + self.service_s for d in ctx["downlinks"])]
        cand.sort(key=lambda a: a["t0"] + self.service_s)
        ends = [a["t0"] + self.service_s for a in cand]
        prev = [bisect.bisect_right(ends, a["t0"]) - 1 for a in cand]
        best = [0.0] * (len(cand) + 1)
        for i, a in enumerate(cand):
            best[i + 1] = max(best[i], a["priority"] + best[prev[i] + 1])
        i, chosen = len(cand), []
        while i > 0:
            if best[i] != best[i - 1]:
                chosen.append(cand[i - 1])
                i = prev[i - 1] + 1
            else:
                i -= 1
        tasks += [dict(type="image", rid=a["rid"], idx=a["idx"], t0=a["t0"], t1=a["t1"], priority=a["priority"])
                  for a in chosen]
        return sorted(tasks, key=lambda x: x["t0"])


# ------------------------------------------------------------------ segmento di terra

class Ground:
    def __init__(self, sim, planner=None):
        self.sim, self.planner = sim, planner or DemoPlanner()
        n = sim.n
        self.processed = [set() for _ in range(n)]
        self.plans = [None] * n
        self.known = [dict() for _ in range(n)]        # idx richiesta -> istante in cui il satellite l'ha ricevuta
        self.reserved = {}                             # idx -> (satellite, chiusura): assegnata a un piano in corso
        self.ledger = [deque() for _ in range(n)]      # FIFO (idx, bit cumulati) delle immagini non ancora scaricate
        self.imaged_bits = [0.0] * n
        self.onboard_imaged = [set() for _ in range(n)]
        self.uplink_log = [[] for _ in range(n)]

    # ---- contatti
    def _uplink_windows(self, k, t_from, t_to):
        sim = self.sim
        out = []
        for o in windows_between(sim.sats[k], t_from, t_to, "ground_station"):
            i = sim.sidx.get(o["object"])
            if i is not None and "uplink" in sim.station_roles[i]:
                out.append((o["window"][0], o["window"][1], i))
        return out

    def downlink_windows(self, k, t_from, t_to):
        sim = self.sim
        out = []
        for o in windows_between(sim.sats[k], t_from, t_to, "ground_station"):
            i = sim.sidx.get(o["object"])
            if i is not None and "downlink" in sim.station_roles[i]:
                out.append(dict(station=i, t0=max(o["window"][0], t_from), t1=min(o["window"][1], t_to)))
        return [d for d in out if d["t1"] - d["t0"] >= MIN_ARM_S]

    def contacts(self, k, t0):
        """Contatti di uplink del satellite k che iniziano o sono in corso nel round [t0, t0 + round_s) e non
        sono ancora stati trattati (i contatti sovrapposti a uno gia' trattato contano come uno)."""
        sim = self.sim
        found = []
        ensure_windows(sim.sats[k], min(t0 + sim.round_s + 1.0, sim.seconds))
        for o, c, i in self._uplink_windows(k, t0, t0 + sim.round_s):
            key = (i, round(o, 1))
            if key in self.processed[k]:
                continue
            self.processed[k].add(key)
            found.append((o, c, i))
        return sorted(found)

    # ---- pianificazione
    def _horizon_end(self, k, t_up, close):
        sim, cap = self.sim, self.sim.planning["horizon_max_s"]
        end = min(t_up + cap, sim.seconds)
        ensure_windows(sim.sats[k], min(end + MAX_WINDOW, sim.seconds))
        nxt = [o for o, c, i in self._uplink_windows(k, close, end + MAX_WINDOW) if o > close + 1e-6]
        if nxt and min(nxt) <= end:
            return min(nxt), True                       # il piano arriva fino al contatto successivo
        return end, False

    def _arms(self, k, t_plan, t_up, t_h):
        sim, arms = self.sim, []
        for o in windows_between(sim.sats[k], t_up, t_h, "target"):
            idx = sim.rindex.get(o["object"])
            if idx is None:
                continue
            r = sim.reqs[idx]
            if r["status"] != "active" or r["arrival_s"] > t_plan or r["ground_imaged"] is not None:
                continue
            hold = self.reserved.get(idx)
            if hold and hold[0] != k and hold[1] > t_up:
                continue
            t0, t1 = max(o["window"][0], t_up), o["window"][1]
            if r["expires_s"] is not None:
                t1 = min(t1, r["expires_s"])
            if t1 - t0 >= MIN_ARM_S:
                arms.append(dict(k=k, rid=r["id"], idx=idx, t0=t0, t1=t1, priority=r["priority"]))
        return arms

    def initial_plan(self, k):
        """Piano iniziale: se all'avvio non c'e' un contatto di uplink in corso, il satellite parte con un piano come se
        fosse stato caricato al contatto precedente (t = 0)."""
        sim = self.sim
        ensure_windows(sim.sats[k], min(sim.round_s + 1.0, sim.seconds))
        if not self._uplink_windows(k, 0.0, sim.round_s):
            self.uplink(k, 0.0, (0.0, 0.0, None), initial=True)

    def uplink(self, k, t0, window, initial=False):
        """Trattamento di un contatto di uplink. Restituisce l'evento da registrare nella traccia."""
        sim = self.sim
        o, c, i = window
        t_up = max(o, t0)
        ev = dict(t=round(t_up, 1), t_end=round(c, 1), station=i, ok=True)
        if initial:
            ev["initial"] = True
        if not initial and c - o < sim.comms["min_uplink_contact_s"]:
            ev.update(ok=False, reason="contatto piu' breve del minimo per l'invio del piano")
            return ev
        t_plan = t_up - sim.comms["uplink_latency_s"]
        t_h, at_contact = self._horizon_end(k, t_up, c)
        for idx, hold in list(self.reserved.items()):        # il nuovo piano sostituisce il precedente di k
            if hold[0] == k:
                del self.reserved[idx]
        if self.plans[k]:
            for task in self.plans[k]["tasks"]:
                if task["state"] == "pending":
                    task["state"] = "replaced"
        arms = self._arms(k, t_plan, t_up, t_h)
        ctx = dict(k=k, t_up=t_up, t_plan=t_plan, horizon_end=t_h, arms=arms,
                   downlinks=self.downlink_windows(k, t_up, t_h))
        tasks = [dict(t, state="pending") for t in self.planner.plan(ctx)]
        for t in tasks:
            if t["type"] == "image":
                self.reserved[t["idx"]] = (k, t["t1"])
        self.plans[k] = dict(t_up=t_up, horizon_end=t_h, tasks=tasks)
        new = sorted(i2 for i2, r in enumerate(sim.reqs)
                     if r["status"] == "active" and r["arrival_s"] <= t_plan and i2 not in self.known[k])
        for i2 in new:
            self.known[k][i2] = t_up
        ev.update(horizon_end=round(t_h, 1), new_known=new, n_arms=len(arms),
                  **(dict(next_uplink=round(t_h, 1)) if at_contact else {}),
                  tasks=[dict(type=t["type"], t0=round(t["t0"], 1), t1=round(t["t1"], 1),
                              **(dict(target=t["idx"]) if t["type"] == "image" else dict(station=t["station"])))
                         for t in tasks])
        self.uplink_log[k].append(ev)
        return ev

    # ---- esecuzione a bordo
    def execute(self, k, t0, CH, DL):
        """Azione del satellite k nel round che inizia a t0: la prima attivita' del piano ancora da fare."""
        sim, sat = self.sim, self.sim.sats[k]
        if not sat.is_alive():
            return NO_ACTION, "charge", None
        batt = sat.dynamics.battery_charge_fraction
        sim.charging[k] = batt < 0.2 or (sim.charging[k] and batt < 0.5)     # protezione di bordo
        if sim.charging[k]:
            return CH, "charge", None
        for task in (self.plans[k] or {}).get("tasks", []):
            if task["state"] != "pending":
                continue
            if task["type"] == "image":
                r = sim.reqs[task["idx"]]
                if r["status"] == "expired" or task["idx"] in self.onboard_imaged[k]:
                    task["state"] = "skipped"
                    continue
                if t0 >= task["t1"] - 1e-6:
                    task["state"] = "missed"
                    self.reserved.pop(task["idx"], None)
                    continue
                if t0 + sim.round_s <= task["t0"] - SLEW_LEAD:
                    break
                return r["target"], "image", task["idx"]
            if t0 >= task["t1"] - 1e-6:
                task["state"] = "done"
                continue
            if t0 + sim.round_s <= task["t0"]:
                break
            if sat.dynamics.storage_level > 0:
                return DL, "downlink", None
            task["state"] = "skipped"
        return CH, "charge", None

    def after_step(self, k, t1, imaged_idx, bits):
        """Registra l'esito di un round: immagine fatta (con i suoi bit) e dati scaricati.
        Restituisce gli indici delle richieste che la terra viene a sapere ripresi (scaricati) in questo round."""
        sim, sat = self.sim, self.sim.sats[k]
        if imaged_idx is not None:
            self.onboard_imaged[k].add(imaged_idx)
            self.imaged_bits[k] += bits
            self.ledger[k].append((imaged_idx, self.imaged_bits[k]))
            for task in (self.plans[k] or {}).get("tasks", []):
                if task["type"] == "image" and task["idx"] == imaged_idx and task["state"] == "pending":
                    task["state"] = "done"
        downlinked = self.imaged_bits[k] - sat.dynamics.storage_level
        learned = []
        while self.ledger[k] and self.ledger[k][0][1] <= downlinked + 1.0:
            learned.append(self.ledger[k].popleft()[0])
        return learned
