"""Esporta in JSON la traccia di un episodio BSK-RL per il viewer (viewer/).

Con --sats N > 1 simula una costellazione Walker-delta (GeneralSatelliteTasking, un'azione
per satellite, round sincroni) con comunicazione libera tra satelliti: ognuno sa quali target
sono gia' stati ripresi dagli altri.

Politica dimostrativa (non e' un bandit), per ogni satellite: Charge se la batteria e' bassa,
Downlink quando c'e' una stazione in vista e il buffer non e' vuoto, altrimenti Image sul
target la cui finestra si chiude per prima, evitando quelli gia' scelti dagli altri satelliti.

Uso: uv run python scripts/export_trace.py [--sats 3 --planes 2] [--seed 4] [--orbits 3]
Output: outputs/traces/trace_seed<seed>.json (1 satellite) o trace_seed<seed>_<N>sats.json
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl import comm, data, scene
from bsk_rl.gym import NO_ACTION
from bsk_rl.utils.orbital import walker_delta_args

from pilot_bskrl import K_AHEAD, ORBIT, make_sat

REQ = orbitalMotion.REQ_EARTH  # [km]
IMG_BITS = 8e6  # un'immagine (instrumentBaudRate di default)


def latlon(r):
    """Latitudine/longitudine geocentriche [deg] di un vettore planet-fixed."""
    r = np.asarray(r, dtype=float)
    n = np.linalg.norm(r, axis=-1)
    return np.degrees(np.arcsin(r[..., 2] / n)), np.degrees(np.arctan2(r[..., 1], r[..., 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)  # seed con finestre di downlink
    ap.add_argument("--orbits", type=float, default=3.0)
    ap.add_argument("--targets", type=int, default=600)
    ap.add_argument("--round", type=float, default=60.0, help="durata round [s]")
    ap.add_argument("--track-dt", type=float, default=10.0)
    ap.add_argument("--sats", type=int, default=1, help="numero di satelliti")
    ap.add_argument("--planes", type=int, default=None, help="piani orbitali Walker (default min(sats, 2))")
    ap.add_argument("--altitude", type=float, default=500.0, help="quota Walker [km]")
    ap.add_argument("--inc", type=float, default=55.0, help="inclinazione Walker [deg]")
    ap.add_argument("--out", default="outputs/traces")
    a = ap.parse_args()
    multi = a.sats > 1
    planes = a.planes or min(a.sats, 2)

    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=[make_sat(variable_interval=False, name=f"sat_{i + 1}") for i in range(a.sats)],
        scenario=scene.UniformTargets(a.targets),
        rewarder=data.UniqueImageReward(),
        communicator=comm.FreeCommunication() if multi else None,
        # una sola satellite: orbita casuale come nel pilota; piu' satelliti: costellazione Walker
        # con fase e nodo casuali (dal seed, via np.random globale di BSK-RL)
        sat_arg_randomizer=walker_delta_args(n_planes=planes, altitude=a.altitude, inc=a.inc) if multi else None,
        time_limit=a.orbits * ORBIT,
        max_step_duration=a.round,
        log_level="ERROR",
        disable_env_checker=True,
    ).unwrapped
    env.reset(seed=a.seed)
    sats = env.satellites
    targets = env.scenario.targets
    tindex = {t: i for i, t in enumerate(targets)}
    DL, CH = K_AHEAD, K_AHEAD + 1

    n = len(sats)
    rounds = [[] for _ in sats]
    imaged_events = [[] for _ in sats]
    current = [None] * n
    charging = [False] * n
    cum = [0.0] * n
    dev_km = [0.0] * n
    done = False
    while not done:
        t = env.simulator.sim_time
        actions, labels, tgts = [], [], []
        for k, sat in enumerate(sats):
            if not sat.is_alive():
                actions.append(NO_ACTION); labels.append("charge"); tgts.append(None); continue
            batt, stor = sat.dynamics.battery_charge_fraction, sat.dynamics.storage_level_fraction
            charging[k] = batt < 0.35 or (charging[k] and batt < 0.7)
            gs_open = any(w[0] <= t <= w[1] for w in sat.next_opportunities_dict(types="ground_station").values())
            if charging[k]:
                actions.append(CH); labels.append("charge"); tgts.append(None)
            elif gs_open and stor > 0:
                actions.append(DL); labels.append("downlink"); tgts.append(None)
            else:
                # nuovo target se il satellite lo chiede o se qualcuno lo ha gia' ripreso
                if current[k] is None or sat.requires_retasking or current[k] in sat.data_store.data.imaged:
                    taken = {current[j] for j in range(n) if j != k and current[j] is not None and (j > k or labels[j] == "image")}
                    cands = [o["object"] for o in sat.find_next_opportunities(n=n + 1, types="target")]
                    current[k] = next((c for c in cands if c not in taken), cands[0])
                actions.append(current[k]); labels.append("image"); tgts.append(tindex[current[k]])
        before = [s.imaged for s in sats]
        _, _, term, trunc, _ = env.step(actions)
        t1 = env.simulator.sim_time
        for k, sat in enumerate(sats):
            r = env.reward_dict.get(sat.name, 0.0)
            cum[k] += r
            if sat.imaged > before[k]:
                imaged_events[k].append(dict(t=t1, target=tindex[sat.latest_target]))
            # scarto tra la traiettoria usata per le finestre e la posizione realmente simulata
            dev_km[k] = max(dev_km[k], float(np.linalg.norm(sat.trajectory.r_BP_P(t1) - sat.dynamics.r_BN_P)) / 1e3)
            rounds[k].append(dict(
                t0=t, t1=t1, action=labels[k], target=tgts[k], reward=round(r, 4), cum_reward=round(cum[k], 4),
                storage=round(sat.dynamics.storage_level / IMG_BITS, 2),
                battery=round(sat.dynamics.battery_charge_fraction, 3),
                imaged=sat.imaged, alive=bool(sat.is_alive()),
            ))
        done = term or trunc

    t_end = env.simulator.sim_time
    stations = [dict(name=g["name"], lat=g["lat"], lon=g["long"]) for g in env.world_args["groundStationsData"]]
    sidx = {"GroundStation" + s["name"]: i for i, s in enumerate(stations)}
    tlat, tlon = latlon(np.array([x.r_LP_P for x in targets]))
    ts = np.arange(0.0, t_end + 1e-9, a.track_dt)

    sat_out = []
    for k, sat in enumerate(sats):
        sat.trajectory.extend_to(t_end + a.track_dt)
        pos = np.array([sat.trajectory.r_BP_P(x) for x in ts])
        lat, lon = latlon(pos)
        alt = np.linalg.norm(pos, axis=1) / 1e3 - REQ
        windows = []
        for o in sat.opportunities:
            kind = o["type"]
            obj = tindex.get(o["object"]) if kind == "target" else sidx.get(o["object"])
            if obj is not None and o["window"][0] < t_end:
                windows.append(dict(kind="target" if kind == "target" else "station", obj=obj,
                                    t0=round(o["window"][0], 1), t1=round(o["window"][1], 1)))
        oe = sat.sat_args["oe"]
        sat_out.append(dict(
            name=sat.name,
            meta=dict(
                inclination_deg=round(float(np.degrees(oe.i)), 2), altitude_km=round(float(alt.mean()), 1),
                raan_deg=round(float(np.degrees(oe.Omega)) % 360, 1),
                image_min_elev_deg=round(float(np.degrees(sat.sat_args["imageTargetMinimumElevation"])), 1),
                storage_capacity=round(sat.dynamics.storageUnit.storageCapacity / IMG_BITS, 1),
                track_vs_sim_max_km=round(dev_km[k], 2),
            ),
            track=dict(lat=np.round(lat, 3).tolist(), lon=np.round(lon, 3).tolist(), alt_km=np.round(alt, 1).tolist()),
            windows=windows, rounds=rounds[k], imaged=imaged_events[k],
        ))

    # Formato v2: dati comuni (target, stazioni) + lista di satelliti.
    trace = dict(
        version=2,
        meta=dict(
            seed=a.seed, round_s=a.round, t_end=t_end, n_targets=len(targets), n_satellites=n,
            gs_min_elev_deg=round(float(np.degrees(env.world_args.get("gsMinimumElevation", np.radians(10)))), 1),
            earth_radius_km=REQ, track_dt=a.track_dt,
            constellation=(f"Walker-delta: {n} satelliti, {planes} piani, {a.altitude:.0f} km, {a.inc:.0f} deg, "
                           "fase e nodo casuali dal seed; comunicazione libera tra satelliti") if multi
            else "satellite singolo, orbita casuale dal seed",
            note="Posizioni dalla traiettoria usata da BSK-RL per le finestre di accesso "
                 "(TrajectorySimulator), non loggate dalla simulazione completa. "
                 "Campo visivo = zona con elevazione >= image_min_elev_deg; assetto e slew non sono mostrati.",
            policy="demo deterministica: charge/downlink/image (chiusura piu vicina), non un bandit",
        ),
        stations=stations,
        targets=[dict(name=x.name, lat=round(float(la), 3), lon=round(float(lo), 3), priority=round(float(x.priority), 3))
                 for x, la, lo in zip(targets, tlat, tlon)],
        satellites=sat_out,
    )
    env.close()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"trace_seed{a.seed}_{n}sats.json" if multi else f"trace_seed{a.seed}.json")
    path.write_text(json.dumps(trace, separators=(",", ":")))
    print(f"{path} ({path.stat().st_size / 1e3:.0f} kB): t_end={t_end:.0f}s, round={len(rounds[0])}")
    for k, s in enumerate(sat_out):
        last = rounds[k][-1]
        print(f"  {s['name']}: immagini={last['imaged']}, reward={last['cum_reward']}, vivo={last['alive']}, "
              f"incl={s['meta']['inclination_deg']}, RAAN={s['meta']['raan_deg']}, scarto traiettoria-sim max={dev_km[k]:.2f} km")
    imaged_total = len({e["target"] for evs in imaged_events for e in evs})
    dup = sum(len(e) for e in imaged_events) - imaged_total
    print(f"  target distinti ripresi={imaged_total}, immagini duplicate={dup}")


if __name__ == "__main__":
    main()
