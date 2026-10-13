"""Esporta in JSON la traccia di un episodio BSK-RL per il viewer (viewer/).

Politica dimostrativa (non e' un bandit): Charge se la batteria e' bassa, Downlink
quando c'e' una stazione in vista e il buffer non e' vuoto, altrimenti Image su un
target la cui finestra si chiude per prima. Round a durata fissa.

Uso: uv run python scripts/export_trace.py [--seed 4] [--orbits 3]
Output: outputs/traces/trace_seed<seed>.json
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl import data, scene

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
    ap.add_argument("--out", default="outputs/traces")
    a = ap.parse_args()

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=make_sat(variable_interval=False),
        scenario=scene.UniformTargets(a.targets),
        rewarder=data.UniqueImageReward(),
        time_limit=a.orbits * ORBIT,
        max_step_duration=a.round,
        log_level="ERROR",
        disable_env_checker=True,
    ).unwrapped
    env.reset(seed=a.seed)
    sat = env.satellites[0]
    targets = env.scenario.targets
    tindex = {t: i for i, t in enumerate(targets)}
    DL, CH = K_AHEAD, K_AHEAD + 1

    rounds, imaged_events = [], []
    current, charging, done, cum = None, False, False, 0.0
    sim_dev_km = 0.0
    while not done:
        t = env.simulator.sim_time
        batt = sat.dynamics.battery_charge_fraction
        stor = sat.dynamics.storage_level_fraction
        charging = batt < 0.35 or (charging and batt < 0.7)
        gs_open = any(w[0] <= t <= w[1] for w in sat.next_opportunities_dict(types="ground_station").values())
        if charging:
            action, label, tgt = CH, "charge", None
        elif gs_open and stor > 0:
            action, label, tgt = DL, "downlink", None
        else:
            if current is None or sat.requires_retasking:
                current = sat.find_next_opportunities(n=1, types="target")[0]["object"]
            action, label, tgt = current, "image", tindex[current]
        n_before = sat.imaged
        _, r, term, trunc, _ = env.step(action)
        cum += r
        if sat.imaged > n_before:
            imaged_events.append(dict(t=env.simulator.sim_time, target=tindex[sat.latest_target]))
        # scarto tra la traiettoria usata per le finestre e la posizione realmente simulata
        t1 = env.simulator.sim_time
        sim_dev_km = max(sim_dev_km, float(np.linalg.norm(sat.trajectory.r_BP_P(t1) - sat.dynamics.r_BN_P)) / 1e3)
        rounds.append(dict(
            t0=t, t1=t1, action=label, target=tgt, reward=round(r, 4), cum_reward=round(cum, 4),
            storage=round(sat.dynamics.storage_level / IMG_BITS, 2),
            battery=round(sat.dynamics.battery_charge_fraction, 3),
            imaged=sat.imaged, alive=bool(sat.is_alive()),
        ))
        done = term or trunc

    t_end = env.simulator.sim_time
    sat.trajectory.extend_to(t_end + a.track_dt)
    ts = np.arange(0.0, t_end + 1e-9, a.track_dt)
    pos = np.array([sat.trajectory.r_BP_P(x) for x in ts])
    lat, lon = latlon(pos)
    alt = np.linalg.norm(pos, axis=1) / 1e3 - REQ

    stations = [dict(name=g["name"], lat=g["lat"], lon=g["long"]) for g in env.world_args["groundStationsData"]]
    sidx = {"GroundStation" + s["name"]: i for i, s in enumerate(stations)}
    windows = []
    for o in sat.opportunities:
        kind = o["type"]
        obj = tindex.get(o["object"]) if kind == "target" else sidx.get(o["object"])
        if obj is not None and o["window"][0] < t_end:
            windows.append(dict(kind="target" if kind == "target" else "station", obj=obj,
                                t0=round(o["window"][0], 1), t1=round(o["window"][1], 1)))
    tlat, tlon = latlon(np.array([x.r_LP_P for x in targets]))
    oe = sat.sat_args["oe"]

    # Formato v2: dati comuni (target, stazioni) + lista di satelliti. Questo script ne esporta uno;
    # il viewer ne mostra quanti ne trova in "satellites".
    trace = dict(
        version=2,
        meta=dict(
            seed=a.seed, round_s=a.round, t_end=t_end, n_targets=len(targets),
            gs_min_elev_deg=round(float(np.degrees(env.world_args.get("gsMinimumElevation", np.radians(10)))), 1),
            earth_radius_km=REQ, track_dt=a.track_dt,
            note="Posizioni dalla traiettoria usata da BSK-RL per le finestre di accesso "
                 "(TrajectorySimulator), non loggate dalla simulazione completa. "
                 "Campo visivo = zona con elevazione >= image_min_elev_deg; assetto e slew non sono mostrati.",
            policy="demo deterministica: charge/downlink/image (chiusura piu vicina), non un bandit",
        ),
        stations=stations,
        targets=[dict(name=x.name, lat=round(float(la), 3), lon=round(float(lo), 3), priority=round(float(x.priority), 3))
                 for x, la, lo in zip(targets, tlat, tlon)],
        satellites=[dict(
            name=sat.name,
            meta=dict(
                inclination_deg=round(float(np.degrees(oe.i)), 2), altitude_km=round(float(alt.mean()), 1),
                image_min_elev_deg=round(float(np.degrees(sat.sat_args["imageTargetMinimumElevation"])), 1),
                storage_capacity=round(sat.dynamics.storageUnit.storageCapacity / IMG_BITS, 1),
                track_vs_sim_max_km=round(sim_dev_km, 2),
            ),
            track=dict(lat=np.round(lat, 3).tolist(), lon=np.round(lon, 3).tolist(), alt_km=np.round(alt, 1).tolist()),
            windows=windows,
            rounds=rounds,
            imaged=imaged_events,
        )],
    )
    env.close()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"trace_seed{a.seed}.json"
    path.write_text(json.dumps(trace, separators=(",", ":")))
    last = rounds[-1]
    print(f"{path} ({path.stat().st_size / 1e3:.0f} kB): t_end={t_end:.0f}s, round={len(rounds)}, "
          f"immagini={last['imaged']}, reward={last['cum_reward']}, vivo={last['alive']}, "
          f"scarto traiettoria-sim max={sim_dev_km:.2f} km")


if __name__ == "__main__":
    main()
