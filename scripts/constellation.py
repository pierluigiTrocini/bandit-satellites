"""Episodio BSK-RL (uno o più satelliti) che produce una traccia v2 incrementale.

Usato da export_trace.py (episodio completo su file) e da sim_server.py (stream live).
`reset(seed)` costruisce la traccia iniziale, `step()` avanza di un round e restituisce il
"delta" (solo i dati nuovi); `self.trace` contiene sempre la traccia accumulata finora.

Con più satelliti: costellazione Walker-delta (GeneralSatelliteTasking, un'azione per
satellite, round sincroni) e comunicazione libera tra satelliti. La politica e' dimostrativa
(non e' un bandit): per ogni satellite Charge se la batteria e' bassa, Downlink se c'e' una
stazione in vista e il buffer non e' vuoto, altrimenti Image sul target la cui finestra si
chiude per prima, evitando quelli scelti o gia' ripresi da altri.

Tre modalita' (`self.mode`):
  legacy  nessun file di configurazione: la politica sopra, a ogni round, su target casuali fissi;
  round   configurazione con `comms.uplink: continuous`: la stessa politica a ogni round, ma le richieste
          possono arrivare nel tempo (flusso Poisson o file) e scadere;
  plan    configurazione con `comms.uplink: at_ground_contact` (default): decisioni a terra a lotti, il piano
          sale al satellite al contatto con una stazione di uplink (vedi ground.py, modello B).
"""

import bisect

import gymnasium as gym
import numpy as np
from scipy.interpolate import interp1d
from Basilisk.utilities import orbitalMotion

from bsk_rl import comm, data, scene
from bsk_rl.gym import NO_ACTION
from bsk_rl.utils.orbital import walker_delta_args

from config_io import make_oe
from ground import Ground, WindowEngine, forget_targets, register_targets, windows_between, ensure_windows
from pilot_bskrl import K_AHEAD, ORBIT, make_sat
from request_stream import RequestTarget, make_source, requests_file

REQ = orbitalMotion.REQ_EARTH  # [km]
IMG_BITS = 8e6  # un'immagine (instrumentBaudRate di default)
DL, CH = K_AHEAD, K_AHEAD + 1  # indici azione Downlink / Charge
MAX_WINDOW = 3600.0  # [s] durata massima ipotizzata di una finestra di accesso (passaggio LEO)


def latlon(r):
    """Latitudine/longitudine geocentriche [deg] di un vettore planet-fixed."""
    r = np.asarray(r, dtype=float)
    n = np.linalg.norm(r, axis=-1)
    return np.degrees(np.arcsin(r[..., 2] / n)), np.degrees(np.arctan2(r[..., 1], r[..., 0]))


class Sim:
    def __init__(self, sats=1, planes=None, altitude=500.0, inc=55.0, targets=600, round_s=60.0,
                 track_dt=10.0, seconds=3 * ORBIT, plan=None, requests=None):
        """seconds: durata dell'episodio [s]; 0/None = senza limite di tempo.
        plan: piano da file di configurazione (config_io.build_plan); se presente sostituisce
        satelliti, orbite, stazioni, target e durata, e il seed di default e' quello dello scenario.
        requests: elenco normalizzato di richieste (config_io.load_requests) per `targets.mode: file`."""
        self.plan, self.requests_in = plan, requests
        self.mode, self.dynamic, self.source, self.ground = "legacy", False, None, None
        self.comms, self.planning = dict(uplink="continuous"), dict(horizon_max_s=21600.0)
        if plan:
            sc = plan["scenario"]
            sats, targets, round_s, track_dt = len(plan["sats"]), sc["n_targets"], sc["round_s"], sc["track_dt"]
            seconds = sc["duration_hours"] * 3600
            self.comms, self.planning = plan["comms"], sc["planning"]
            self.mode = "plan" if self.comms["uplink"] == "at_ground_contact" else "round"
            self.dynamic = sc["targets"]["mode"] != "random"
        if round_s % track_dt:
            raise ValueError("round_s deve essere multiplo di track_dt")
        self.n, self.planes = sats, planes or min(sats, 2)
        self.altitude, self.inc, self.n_targets = altitude, inc, targets
        self.round_s, self.track_dt, self.seconds = round_s, track_dt, seconds or float("inf")
        self.env = None

    @property
    def multi(self):
        return self.n > 1

    @property
    def t(self):
        return self.env.simulator.sim_time

    def _env_args(self):
        """Argomenti di gym.make che dipendono da come e' definita la costellazione."""
        if self.plan:
            specs = self.plan["sats"]
            return dict(
                satellites=[make_sat(False, s["name"], s["sat_args"], observe_opportunities=False) for s in specs],
                # orbite dal file (nessuna casualita'); il seed governa solo i target
                sat_arg_randomizer=lambda sats: {sat: {"oe": make_oe(s["orbit"])} for sat, s in zip(sats, specs)},
                world_args=self.plan["world_args"])
        return dict(
            satellites=[make_sat(variable_interval=False, name=f"sat_{i + 1}") for i in range(self.n)],
            # un satellite: orbita casuale (come nel pilota); piu' satelliti: Walker-delta con
            # fase e nodo casuali dal seed (np.random globale di BSK-RL)
            sat_arg_randomizer=(walker_delta_args(n_planes=self.planes, altitude=self.altitude, inc=self.inc)
                                if self.multi else None))

    def reset(self, seed=None):
        if seed is None:
            seed = self.plan["scenario"]["seed"] if self.plan else 4
        if self.env is None:
            self.env = gym.make(
                "GeneralSatelliteTasking-v1",
                scenario=scene.UniformTargets(self.n_targets),
                rewarder=data.UniqueImageReward(),
                communicator=comm.FreeCommunication() if self.multi and self.comms.get("crosslinks", "free") == "free" else None,
                **self._env_args(),
                time_limit=self.seconds,
                max_step_duration=self.round_s,
                log_level="ERROR",
                disable_env_checker=True,
            ).unwrapped
        env = self.env
        env.reset(seed=seed)
        self.seed, self.sats, self.targets = seed, env.satellites, env.scenario.targets
        if self.plan:                              # finestre di accesso: calcolo nostro (vedi ground.WindowEngine)
            for sat in self.sats:
                sat._engine = WindowEngine(sat, self.seconds)
                sat._engine.take_over()
        self.tindex = {t: i for i, t in enumerate(self.targets)}
        self.rindex = self.tindex                  # richieste dinamiche e target statici: stesso indice di traccia
        n = self.n
        self.current, self.charging, self.cum = [None] * n, [False] * n, [0.0] * n
        self.dev_km, self.done = [0.0] * n, False
        self.img_bits = [s.sat_args["instrumentBaudRate"] for s in self.sats]   # un'immagine = 1 s di strumento
        self._sent = [{} for _ in range(n)]   # (kind, obj, t0) -> indice in trace["satellites"][k]["windows"]
        self._last_sample = 0.0

        stations = [dict(name=g["name"], lat=g["lat"], lon=g["long"],
                         min_elev_deg=round(float(np.degrees(g.get("gsMinimumElevation", env.world_args["gsMinimumElevation"]))), 1))
                    for g in env.world_args["groundStationsData"]]
        self.sidx = {"GroundStation" + s["name"]: i for i, s in enumerate(stations)}
        plan_st = {g["name"]: g for g in self.plan["stations"]} if self.plan else {}
        self.station_roles = [plan_st[s["name"]]["roles"] if s["name"] in plan_st else ["uplink", "downlink"] for s in stations]
        for s, roles in zip(stations, self.station_roles):
            s["roles"] = list(roles)
        # richieste: i target statici di BSK-RL hanno arrivo 0 e nessuna scadenza
        self.reqs, self.expired_log = [], []
        for i, tg in enumerate(self.targets):
            if self.plan:
                tg._id = tg.name                    # identita' stabile (quella di BSK-RL contiene id(self))
            self.reqs.append(dict(idx=i, id=tg.name, target=tg, priority=float(tg.priority), arrival_s=0.0, expires_s=None,
                                  status="active", ground_imaged=None, imaged_by={}))
        tlat, tlon = latlon(np.array([x.r_LP_P for x in self.targets]).reshape(-1, 3))
        sat_out = []
        for k, sat in enumerate(self.sats):
            lat, lon, alt = self._sample(sat, [0.0])
            oe = sat.sat_args["oe"]
            sat_out.append(dict(
                name=sat.name,
                meta=dict(
                    inclination_deg=round(float(np.degrees(oe.i)), 2), altitude_km=round(float(alt[0]), 1),
                    raan_deg=round(float(np.degrees(oe.Omega)) % 360, 1),
                    image_min_elev_deg=round(float(np.degrees(sat.sat_args["imageTargetMinimumElevation"])), 1),
                    storage_capacity=round(sat.dynamics.storageUnit.storageCapacity / self.img_bits[k], 1),
                    track_vs_sim_max_km=0.0,
                ),
                track=dict(lat=lat, lon=lon, alt_km=alt), windows=[], rounds=[], imaged=[], uplinks=[],
            ))
        self.trace = dict(
            version=2,
            meta=dict(
                seed=seed, round_s=self.round_s, t_end=0.0, n_targets=len(self.targets), n_satellites=n,
                infinite=self.seconds == float("inf"),
                gs_min_elev_deg=round(float(np.degrees(env.world_args.get("gsMinimumElevation", np.radians(10)))), 1),
                earth_radius_km=REQ, track_dt=self.track_dt,
                constellation=(f"{self.plan['name']}: {n} satelliti da file di configurazione, epoca "
                               f"{self.plan['world_args']['utc_init']}, richieste: {self._targets_text()}") if self.plan else
                              (f"Walker-delta: {n} satelliti, {self.planes} piani, {self.altitude:.0f} km, "
                               f"{self.inc:.0f} deg, fase e nodo casuali dal seed; comunicazione libera tra satelliti")
                if self.multi else "satellite singolo, orbita casuale dal seed",
                note="Posizioni dalla traiettoria usata da BSK-RL per le finestre di accesso "
                     "(TrajectorySimulator), non loggate dalla simulazione completa. "
                     "Campo visivo = zona con elevazione >= image_min_elev_deg; assetto e slew non sono mostrati.",
                policy=self._policy_text(),
                mode=self.mode, comms=dict(self.comms), dynamic=self.dynamic,
                notices=[dict(n) for n in (self.plan["notices"] if self.plan else [])],
            ),
            stations=stations,
            targets=[dict(id=x.name if self.plan else x.name, name=x.name, lat=round(float(la), 3), lon=round(float(lo), 3),
                          priority=round(float(x.priority), 3)) for x, la, lo in zip(self.targets, tlat, tlon)],
            satellites=sat_out,
            targets_removed=[],
        )
        if self.dynamic:
            self.source = make_source(seed, self.plan["scenario"]["targets"], self.requests_in)
            self._arrivals(0.0)
        if self.mode == "plan":
            self.ground = Ground(self)
            for k in range(n):                  # i target statici sono noti ai satelliti fin dall'inizio
                for i in range(len(self.targets)):
                    self.ground.known[k][i] = 0.0
            for k in range(n):
                self.ground.initial_plan(k)
            self.trace["meta"]["notices"].append(dict(
                level="info", code="initial_plan", message=(
                    "All'avvio ogni satellite senza contatto di uplink in corso ha gia' un piano, come se fosse stato "
                    "caricato al contatto precedente (t = 0).")))
            for k in range(n):
                self.trace["satellites"][k]["uplinks"] = list(self.ground.uplink_log[k])
        for k in range(n):                      # finestre gia' aperte al tempo 0
            self._upsert_windows(k, -1.0, 0.0)

    def _targets_text(self):
        tg = self.plan["scenario"]["targets"]
        if tg["mode"] == "stream":
            return (f"flusso Poisson {tg['rate_per_hour']:g}/h"
                    + (f", validita' {tg['validity_s'] / 3600:g} h" if tg["validity_s"] else ", senza scadenza"))
        return "da file" if tg["mode"] == "file" else f"{tg['count']} target casuali fissi"

    def _policy_text(self):
        if self.mode == "plan":
            return ("piano a terra al contatto di uplink (pianificatore dimostrativo: programmazione a intervalli pesata), "
                    "esecuzione a bordo senza scelte; non e' un bandit")
        return "demo deterministica: charge/downlink/image (chiusura piu vicina), non un bandit"

    def _arrivals(self, t):
        """Richieste arrivate fino a t: nuovi target registrati nei satelliti. Restituisce le voci di traccia aggiunte."""
        new = self.source.pop_until(t) if self.source else []
        if not new:
            return []
        targets, added = [], []
        for r in new:
            tg = RequestTarget(r)
            idx = len(self.reqs)
            self.reqs.append(dict(idx=idx, id=r["id"], target=tg, priority=r["priority"], arrival_s=r["arrival_s"],
                                  expires_s=r["expires_s"], status="active", ground_imaged=None, imaged_by={}))
            self.rindex[tg] = idx
            entry = dict(id=r["id"], name=r["name"], lat=r["lat"], lon=r["lon"], priority=r["priority"],
                         arrival_s=r["arrival_s"])
            if r["expires_s"] is not None:
                entry["expires_s"] = r["expires_s"]
            self.trace["targets"].append(entry)
            added.append(entry)
            targets.append(tg)
        for sat in self.sats:
            register_targets(sat, targets, t)
        self.trace["meta"]["n_targets"] = len(self.reqs)
        return added

    def _finish(self, idx, t, reason):
        """Una richiesta esce dal calcolo delle finestre (scaduta, oppure ripresa e nota a terra)."""
        r = self.reqs[idx]
        r["status"] = "expired" if reason == "expired" else "imaged"
        if self.dynamic:
            for sat in self.sats:
                forget_targets(sat, [r["target"]])
            ev = dict(idx=idx, t=round(t, 1), reason=reason)
            self.trace["targets_removed"].append(ev)
            return ev
        return None

    def _expire(self, t):
        removed = []
        for r in self.reqs:
            if r["status"] == "active" and r["expires_s"] is not None and r["expires_s"] <= t and not r["imaged_by"]:
                removed.append(self._finish(r["idx"], t, "expired"))
        return [e for e in removed if e]

    def _positions(self, sat, times):
        """Posizioni planet-fixed [m] ai tempi dati. `trajectory.r_BP_P` ricostruisce a ogni accesso
        l'interpolazione su tutta la traiettoria (costo crescente nel tempo): qui lo si fa su una finestra
        locale di campioni, con lo stesso spline cubico."""
        tr, times = sat.trajectory, np.atleast_1d(times)
        tr.extend_to(times.max() + 3 * tr.dt)
        ts = np.asarray(tr.times)
        lo, hi = max(0, np.searchsorted(ts, times.min()) - 4), np.searchsorted(ts, times.max()) + 5
        pos = [np.matmul(d, r) for d, r in zip(tr._J20002Pfix_log[lo:hi], tr._r_BN_N_log[lo:hi])]
        return interp1d(ts[lo:hi], pos, kind="cubic", axis=0, fill_value="extrapolate")(times)

    def _sample(self, sat, times):
        pos = self._positions(sat, times)
        lat, lon = latlon(pos)
        alt = np.linalg.norm(pos, axis=1) / 1e3 - REQ
        return np.round(lat, 3).tolist(), np.round(lon, 3).tolist(), np.round(alt, 1).tolist()

    def _upsert_windows(self, k, t_from, t_to):
        """Aggiunge/aggiorna in trace le finestre con chiusura dopo t_from e apertura <= t_to."""
        sat, store, out = self.sats[k], self.trace["satellites"][k]["windows"], []
        ops = sat.opportunities       # ordinate per istante di chiusura
        close = lambda x: x["window"][1]
        # una finestra aperta prima di t_to non puo' chiudersi oltre t_to + MAX_WINDOW (passaggi LEO < 1 h):
        # senza questo limite ogni round riscorrerebbe tutte le finestre future dell'episodio
        lo, hi = bisect.bisect_left(ops, t_from - 1e-6, key=close), bisect.bisect_right(ops, t_to + MAX_WINDOW, key=close)
        for o in ops[lo:hi]:
            kind = o["type"]
            obj = self.tindex.get(o["object"]) if kind == "target" else self.sidx.get(o["object"])
            w0, w1 = round(o["window"][0], 1), round(o["window"][1], 1)
            if obj is None or w0 > t_to:
                continue
            key = ("target" if kind == "target" else "station", obj, w0)
            if key not in self._sent[k]:
                self._sent[k][key] = len(store)
                store.append(dict(kind=key[0], obj=obj, t0=w0, t1=w1)); out.append(dict(store[-1]))
            elif store[self._sent[k][key]]["t1"] != w1:     # finestra estesa da un nuovo blocco di calcolo
                store[self._sent[k][key]]["t1"] = w1; out.append(dict(store[self._sent[k][key]]))
        return out

    def _cands(self, sat, t):
        """Richieste attive con una finestra aperta o in arrivo entro un'ora, per chiusura crescente (modalita' round
        con richieste dinamiche: BSK-RL non va cercate in avanti se non ce ne sono)."""
        ensure_windows(sat, min(t + 3600.0, self.seconds))
        out = []
        for o in windows_between(sat, t + 1.0, t + 3600.0, "target"):
            idx = self.rindex.get(o["object"])
            if idx is not None and self.reqs[idx]["status"] == "active" and o["object"] not in sat.data_store.data.imaged \
                    and o["object"] not in out:
                out.append(o["object"])
        return out

    def _choose(self, t):
        """Un'azione per satellite; restituisce (azioni, etichette, indici target)."""
        sats, n = self.sats, self.n
        actions, labels, tgts = [], [], []
        for k, sat in enumerate(sats):
            if not sat.is_alive():
                actions.append(NO_ACTION); labels.append("charge"); tgts.append(None); continue
            batt, stor = sat.dynamics.battery_charge_fraction, sat.dynamics.storage_level_fraction
            self.charging[k] = batt < 0.35 or (self.charging[k] and batt < 0.7)
            gs_open = any(w[0] <= t <= w[1] for w in sat.next_opportunities_dict(types="ground_station").values())
            if self.charging[k]:
                actions.append(CH); labels.append("charge"); tgts.append(None)
            elif gs_open and stor > 0:
                actions.append(DL); labels.append("downlink"); tgts.append(None)
            else:
                cur = self.current
                # nuovo target se il satellite lo chiede o se qualcuno lo ha gia' ripreso
                if cur[k] is None or sat.requires_retasking or cur[k] in sat.data_store.data.imaged:
                    taken = {cur[j] for j in range(n) if j != k and cur[j] is not None and (j > k or labels[j] == "image")}
                    cands = (self._cands(sat, t)[: n + 1] if self.dynamic
                             else [o["object"] for o in sat.find_next_opportunities(n=n + 1, types="target")])
                    if not cands:
                        cur[k] = None
                        actions.append(CH); labels.append("charge"); tgts.append(None); continue
                    cur[k] = next((c for c in cands if c not in taken), cands[0])
                actions.append(cur[k]); labels.append("image"); tgts.append(self.tindex[cur[k]])
        return actions, labels, tgts

    def step(self):
        """Avanza di un round. Restituisce il delta {t0, t1, done, sats: [...], targets_added, targets_removed}."""
        env, t0, n = self.env, self.t, self.n
        added, removed, uplinks = [], [], [[] for _ in range(n)]
        if self.dynamic:
            added = self._arrivals(t0) if t0 > 0 else []
            removed += self._expire(t0)
        if self.mode == "plan":
            for k in range(n):
                for w in self.ground.contacts(k, t0):
                    ev = self.ground.uplink(k, t0, w)
                    uplinks[k].append(ev)
                    self.trace["satellites"][k]["uplinks"].append(ev)
            acts = [self.ground.execute(k, t0, CH, DL) for k in range(n)]
            actions, labels, tgts = [a[0] for a in acts], [a[1] for a in acts], [a[2] for a in acts]
        else:
            actions, labels, tgts = self._choose(t0)
        before = [s.imaged for s in self.sats]
        bits0 = [s.dynamics.storage_level for s in self.sats]
        _, _, term, trunc, _ = env.step(actions)
        t1 = self.t
        self.done = bool(term or trunc)
        sats_delta = []
        for k, sat in enumerate(self.sats):
            tr = self.trace["satellites"][k]
            r = env.reward_dict.get(sat.name, 0.0)
            self.cum[k] += r
            rnd = dict(t0=t0, t1=t1, action=labels[k], target=tgts[k], reward=round(r, 4), cum_reward=round(self.cum[k], 4),
                       storage=round(sat.dynamics.storage_level / self.img_bits[k], 2),
                       battery=round(sat.dynamics.battery_charge_fraction, 3), imaged=sat.imaged, alive=bool(sat.is_alive()))
            tr["rounds"].append(rnd)
            imaged, imaged_idx, bits = [], None, 0.0
            if sat.imaged > before[k]:
                imaged_idx = self.tindex[sat.latest_target]
                imaged.append(dict(t=t1, target=imaged_idx)); tr["imaged"] += imaged
                bits = max(sat.dynamics.storage_level - bits0[k], 0.0)
                self.reqs[imaged_idx]["imaged_by"][k] = t1
            if self.mode == "plan":
                for i in self.ground.after_step(k, t1, imaged_idx, bits):      # la terra sa che e' stata ripresa
                    if self.reqs[i]["status"] == "active":
                        removed.append(self._finish(i, t1, "imaged"))
            elif self.dynamic and imaged_idx is not None and self.reqs[imaged_idx]["status"] == "active":
                removed.append(self._finish(imaged_idx, t1, "imaged"))
            # scarto tra la traiettoria usata per le finestre e la posizione realmente simulata
            self.dev_km[k] = max(self.dev_km[k], float(np.linalg.norm(self._positions(sat, t1)[0] - sat.dynamics.r_BN_P)) / 1e3)
            tr["meta"]["track_vs_sim_max_km"] = round(self.dev_km[k], 2)
            times = list(np.arange(self._last_sample + self.track_dt, t1 + 1e-9, self.track_dt))
            lat, lon, alt = self._sample(sat, times) if times else ([], [], [])
            for key, vals in (("lat", lat), ("lon", lon), ("alt_km", alt)):
                tr["track"][key] += vals
            sats_delta.append(dict(round=rnd, imaged=imaged, windows=self._upsert_windows(k, t0, t1), uplinks=uplinks[k],
                                   track=dict(lat=lat, lon=lon, alt_km=alt), dev=tr["meta"]["track_vs_sim_max_km"]))
        if times:
            self._last_sample = times[-1]
        self.trace["meta"]["t_end"] = t1
        return dict(t0=t0, t1=t1, done=self.done, sats=sats_delta, targets_added=added,
                    targets_removed=[e for e in removed if e])

    def requests_export(self):
        """Richieste arrivate finora in formato requests/v1 (per salvare un flusso generato e rileggerlo)."""
        dyn = [dict(id=r["id"], lat=self.trace["targets"][r["idx"]]["lat"], lon=self.trace["targets"][r["idx"]]["lon"],
                    name=self.trace["targets"][r["idx"]]["name"], elevation_m=0.0, priority=r["priority"],
                    arrival_s=r["arrival_s"], expires_s=r["expires_s"]) for r in self.reqs]
        return requests_file(dyn, name=f"richieste generate (seed {self.seed})")
