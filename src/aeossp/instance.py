"""L'istanza del problema AEOSSP (livello «Instance»).

Un'istanza contiene, per un orizzonte di pianificazione fissato:
  - i satelliti (`SatSpec`) e i target (richieste con profitto rho_i, numero massimo di osservazioni N_i,
    arrivo e scadenza, classe del braccio);
  - i *task candidati* Theta: per ogni finestra di visibilita' (VTW) di un target da parte di un satellite,
    le finestre di osservazione (OTW) possibili. Modello agile: una griglia di inizi ogni `step_s` secondi con
    durata `d` fissa (modello tempo-discreto del paper); modello «none»: una sola OTW = l'intera VTW;
  - per ogni task la direzione di puntamento (linea di vista inerziale) all'inizio e alla fine dell'osservazione,
    da cui si calcola il tempo di transizione fra due task.

I task sono memorizzati come array (ordinati per satellite e istante di inizio): l'identita' di un task e' il
suo indice, stabile per una data configurazione e un dato seed.
"""

from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .config import rng_stream


@dataclass
class Targets:
    names: list
    lat: np.ndarray
    lon: np.ndarray
    priority: np.ndarray
    n_max: np.ndarray
    arrival: np.ndarray
    expires: np.ndarray            # inf = non scade
    cls: np.ndarray                # classe del braccio (cella lat/lon)
    gain: np.ndarray               # (n_max_globale,) guadagno marginale della k-esima osservazione (profitto = rho * somma)

    def __len__(self):
        return len(self.names)


def cell_index(lat, lon, cell_deg):
    ncols = int(np.ceil(360.0 / cell_deg))
    nrows = int(np.ceil(180.0 / cell_deg))
    r = np.clip(((np.asarray(lat) + 90.0) // cell_deg).astype(int), 0, nrows - 1)
    c = np.clip(((np.asarray(lon) + 180.0) // cell_deg).astype(int), 0, ncols - 1)
    return r * ncols + c


def make_targets(cfg):
    """Genera le richieste secondo `cfg['targets']`, con flussi casuali indipendenti derivati dal seed."""
    tg, H, seed, cell = cfg["targets"], cfg["horizon_s"], cfg["seed"], cfg["cell_deg"]
    pmin, pmax = tg["priority"]
    if tg["mode"] == "file":
        rows = tg["requests"]
        names = [r["name"] for r in rows]
        lat = np.array([r["lat"] for r in rows], dtype=float)
        lon = np.array([r["lon"] for r in rows], dtype=float)
        pri = np.array([r["priority"] for r in rows], dtype=float)
        arr = np.array([r["arrival_s"] for r in rows], dtype=float)
        exp = np.array([np.inf if r["expires_s"] is None else r["expires_s"] for r in rows], dtype=float)
    else:
        if tg["mode"] == "random":
            n, arr, exp = tg["count"], np.zeros(tg["count"]), np.full(tg["count"], np.inf)
        else:
            ra = rng_stream(seed, "arrivals")
            n0 = tg["initial_count"]
            k = ra.poisson(tg["rate_per_hour"] * H / 3600.0)
            arr = np.concatenate([np.zeros(n0), np.sort(ra.uniform(0.0, H, k))])
            n = len(arr)
            exp = arr + tg["validity_s"] if tg["validity_s"] is not None else np.full(n, np.inf)
        sp, pr = rng_stream(seed, "space"), rng_stream(seed, "priority")
        lat = np.degrees(np.arcsin(sp.uniform(-1.0, 1.0, n)))
        lon = sp.uniform(-180.0, 180.0, n)
        pri = pr.uniform(pmin, pmax, n)
        names = [f"tgt-{i}" for i in range(n)]
    n = len(names)
    return Targets(names=names, lat=lat, lon=lon, priority=pri, n_max=np.full(n, tg["n_max"], dtype=int), arrival=arr,
                   expires=exp, cls=cell_index(lat, lon, cell), gain=np.array(tg["gain"], dtype=float))


class Instance:
    """Istanza AEOSSP. Si costruisce con `build_instance(cfg)`."""

    def __init__(self, cfg, targets, providers, tasks):
        self.cfg, self.targets, self.providers = cfg, targets, providers
        self.sats = cfg["sats"]
        self.horizon_s = cfg["horizon_s"]
        self.budget_scope = cfg["budget_scope"]
        self.__dict__.update(tasks)                              # array dei task
        self.n = len(self.t0)
        self.cycle_s = cfg["cycle_s"] or float(np.mean([s.orbit.period_s for s in self.sats]))
        self.n_cycles = int(np.ceil(self.horizon_s / self.cycle_s - 1e-9))
        self.cycle = np.minimum((self.t0 // self.cycle_s).astype(int), self.n_cycles - 1)
        # per ogni satellite l'intervallo [lo, hi) di indici dei suoi task (ordinati per inizio)
        self.sat_slice = [(int(np.searchsorted(self.sat, k, "left")), int(np.searchsorted(self.sat, k, "right")))
                          for k in range(len(self.sats))]
        self._conflicts = None

    # ---- transizioni
    def angle(self, p, q):
        """Angolo [rad] fra il puntamento a fine del task p e all'inizio del task q (array)."""
        return np.arccos(np.clip(np.sum(self.los1[p] * self.los0[q], axis=-1), -1.0, 1.0))

    def delta_v(self, p, q):
        """Tempo di manovra (solo DeltaV) da p a q, [s]. Zero per i satelliti non agili."""
        p, q = np.broadcast_arrays(np.asarray(p), np.asarray(q))
        out = np.zeros(p.shape)
        sp = self.sat[p]
        for k, s in enumerate(self.sats):
            m = (sp == k)
            if s.agile and m.any():
                out[m] = s.slew.delta_v(self.angle(p[m], q[m]))
        return out

    def transition(self, p, q):
        """DeltaT = DeltaV + DeltaS da p a q (array); i satelliti non agili hanno DeltaT = 0."""
        p, q = np.broadcast_arrays(np.asarray(p), np.asarray(q))
        out = np.zeros(p.shape)
        sp = self.sat[p]
        for k, s in enumerate(self.sats):
            m = (sp == k)
            if s.agile and m.any():
                out[m] = s.slew.transition(self.angle(p[m], q[m]))
        return out

    def compatible(self, p, q):
        """True se q puo' seguire p sullo stesso satellite: fine(p) + DeltaT(p, q) <= inizio(q). Vincolo (5)."""
        p, q = np.broadcast_arrays(np.asarray(p), np.asarray(q))
        same = self.sat[p] == self.sat[q]
        return same & (self.t1[p] + self.transition(p, q) <= self.t0[q] + 1e-9)

    def conflicts(self):
        """Coppie (p, q), p < q (stesso satellite, p inizia prima), non compatibili: il grafo dei conflitti.

        Un insieme senza conflitti a coppie e' sempre ammissibile (le coppie consecutive ne sono un sottoinsieme),
        ma non e' detto il viceversa: il puntamento si sposta anche *durante* un'osservazione, quindi la
        verifica fra consecutivi (quella del paper, usata da `schedule.evaluate`) puo' ammettere piani che hanno
        una coppia non consecutiva in conflitto. I solver esatti usano perciò gli archi, non questo grafo;
        i conflitti servono solo come condizione sufficiente (confini di ciclo nel MILP)."""
        if self._conflicts is None:
            P, Q = [], []
            for k, (lo, hi) in enumerate(self.sat_slice):
                if hi <= lo:
                    continue
                s = self.sats[k]
                reach = s.slew.max_transition if s.agile else 0.0
                ts = self.t0[lo:hi]
                his = np.searchsorted(ts, self.t1[lo:hi] + reach + 1e-9, "left") + lo
                for i in range(lo, hi):
                    js = np.arange(i + 1, his[i - lo])
                    if len(js):
                        bad = ~self.compatible(np.full(len(js), i), js)
                        P.append(np.full(int(bad.sum()), i))
                        Q.append(js[bad])
            self._conflicts = (np.concatenate(P) if P else np.zeros(0, int), np.concatenate(Q) if Q else np.zeros(0, int))
        return self._conflicts

    # ---- risorse
    def memory_use(self, p):
        """Memoria occupata dal task p [Gbit] (M^I * d)."""
        p = np.atleast_1d(p)
        out = np.zeros(len(p))
        for k, s in enumerate(self.sats):
            m = self.sat[p] == k
            out[m] = s.resources.memory(self.t1[p[m]] - self.t0[p[m]])
        return out

    def energy_imaging(self, p):
        p = np.atleast_1d(p)
        out = np.zeros(len(p))
        for k, s in enumerate(self.sats):
            m = self.sat[p] == k
            out[m] = s.resources.energy_imaging(self.t1[p[m]] - self.t0[p[m]])
        return out

    def energy_maneuver(self, p, q):
        """Energia di manovra da p a q [Wh]."""
        dv = self.delta_v(p, q)
        out = np.zeros(dv.shape)
        sp = self.sat[np.asarray(p)]
        for k, s in enumerate(self.sats):
            m = sp == k
            out[m] = s.resources.energy_maneuver(dv[m])
        return out

    def budget_key(self, p):
        """Chiave (satellite, ciclo) del budget a cui appartiene il task (ciclo 0 se il budget e' sull'orizzonte)."""
        return int(self.sat[p]), (int(self.cycle[p]) if self.budget_scope == "cycle" else 0)

    def memory_capacity(self, k):
        return self.sats[k].resources.memory_capacity_gbit

    def energy_capacity(self, k):
        return self.sats[k].resources.energy_capacity_wh

    # ---- profitto
    def profit_of_count(self, target, n):
        """Profitto del target dopo n osservazioni: rho * somma dei primi n guadagni marginali."""
        g = self.targets.gain
        return float(self.targets.priority[target] * g[:min(int(n), len(g))].sum())

    def task_value(self, p):
        """Profitto della *prima* osservazione del target del task p (rho_i * gain[0])."""
        return self.targets.priority[self.target[p]] * self.targets.gain[0]

    def summary(self):
        per = [int(hi - lo) for lo, hi in self.sat_slice]
        return dict(tasks=self.n, per_sat=per, targets=len(self.targets), cycles=self.n_cycles, cycle_s=round(self.cycle_s, 1),
                    vtws=int(len(np.unique(self.vtw))) if self.n else 0)


def build_instance(cfg, providers=None, targets=None):
    """Costruisce l'istanza. `providers`: un fornitore di posizioni per satellite (default: `cfg['orbit_model']`)."""
    sats, H, epoch = cfg["sats"], cfg["horizon_s"], cfg["epoch"]
    if providers is None:
        mk = G.NumericalJ2 if cfg.get("orbit_model", "numerical_j2") == "numerical_j2" else None
        providers = [mk(s.orbit, epoch, H) if mk else G.KeplerJ2(s.orbit, epoch) for s in sats]
    T = targets or make_targets(cfg)
    r_tgt = G.lla_to_ecef(T.lat, T.lon)
    rows = []                                  # per ogni satellite un dizionario di array
    vtw_id = 0
    step = cfg["step_s"]
    for k, (s, prov) in enumerate(zip(sats, providers)):
        wins = G.find_windows(prov, r_tgt, s.min_elevation_deg * G.D2R, 0.0, H) if len(T) else []
        tgt, t0s, t1s, vid, vs, ve = [], [], [], [], [], []
        for i, ws in enumerate(wins):
            for a, c in ws:
                a, c = max(a, T.arrival[i]), min(c, T.expires[i])         # solo mentre la richiesta e' attiva
                if c - a <= 0:
                    continue
                if s.agile:
                    d = s.observation_s
                    if c - a < d:
                        continue
                    starts = np.arange(a, c - d + 1e-9, step)
                    if len(starts) == 0 or starts[-1] < c - d - 1e-9:
                        starts = np.append(starts, c - d)                 # sempre anche l'ultima OTW possibile
                    t0v, t1v = starts, starts + d
                else:
                    t0v, t1v = np.array([a]), np.array([c])              # CEOS: osservazione sull'intera VTW
                n = len(t0v)
                tgt += [i] * n
                t0s += list(t0v)
                t1s += list(t1v)
                vid += [vtw_id] * n
                vs += [a] * n
                ve += [c] * n
                vtw_id += 1
        rows.append((k, np.array(tgt, dtype=int), np.array(t0s), np.array(t1s), np.array(vid, dtype=int), np.array(vs), np.array(ve)))
    sat = np.concatenate([np.full(len(r[1]), r[0], dtype=int) for r in rows]) if rows else np.zeros(0, int)
    cat = lambda j, dt=float: np.concatenate([r[j] for r in rows]) if rows else np.zeros(0, dt)
    target, t0, t1, vtw, vs, ve = cat(1, int), cat(2), cat(3), cat(4, int), cat(5), cat(6)
    order = np.lexsort((target, t1, t0, sat))                     # per satellite, poi inizio
    sat, target, t0, t1, vtw, vs, ve = (x[order] for x in (sat, target, t0, t1, vtw, vs, ve))
    n = len(t0)
    los0, los1, offnadir, elev = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for k, prov in enumerate(providers):
        m = np.flatnonzero(sat == k)
        if not len(m):
            continue
        tm = 0.5 * (t0[m] + t1[m])
        for arr, tt in ((los0, t0[m]), (los1, t1[m]), (None, tm)):
            rs = prov.eci(tt)
            rt = prov.ecef_to_eci(r_tgt[target[m]], tt)
            v = rt - rs
            los = v / np.linalg.norm(v, axis=1, keepdims=True)
            if arr is not None:
                arr[m] = los
            else:
                nadir = -rs / np.linalg.norm(rs, axis=1, keepdims=True)
                offnadir[m] = np.arccos(np.clip(np.sum(los * nadir, axis=1), -1, 1))
                elev[m] = np.arcsin(np.clip(np.sum(-los * (rt / np.linalg.norm(rt, axis=1, keepdims=True)), axis=1), -1, 1))
    tasks = dict(sat=sat, target=target, t0=t0, t1=t1, vtw=vtw, vtw_start=vs, vtw_end=ve, los0=los0, los1=los1,
                 offnadir=offnadir, elevation=elev, cls=T.cls[target] if n else np.zeros(0, int))
    return Instance(cfg, T, providers, tasks)
