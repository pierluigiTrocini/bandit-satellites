"""Genera un file di replay Vizard (.bin) di un episodio del pilota BSK-RL.

Il file si apre con l'app Vizard di AVS Lab (da installare a parte); questo script
non la richiede. Stessa politica casuale di `pilot_bskrl.py`.

Uso: uv run python scripts/make_vizard_replay.py [--seed 4] [--orbits 3]
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from bsk_rl import data, scene

from pilot_bskrl import K_AHEAD, ORBIT, make_sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)  # seed con finestre di downlink
    ap.add_argument("--orbits", type=float, default=3.0)
    ap.add_argument("--targets", type=int, default=200)
    ap.add_argument("--round", type=float, default=60.0, help="durata round [s]")
    ap.add_argument("--out", default="outputs/vizard/pilot.bin")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=make_sat(variable_interval=False),
        scenario=scene.UniformTargets(a.targets),
        rewarder=data.UniqueImageReward(),
        time_limit=a.orbits * ORBIT,
        max_step_duration=a.round,
        log_level="ERROR",
        disable_env_checker=True,
        vizard_dir=str(out),
        vizard_settings=dict(vizard_rate=10.0, use_simple_earth=True),
    ).unwrapped

    rng = np.random.default_rng(a.seed)
    env.reset(seed=a.seed)
    sat = env.satellites[0]
    current, done, total = None, False, 0.0
    while not done:
        if rng.random() < 0.15 or sat.dynamics.storage_level_fraction >= 0.15:
            action = K_AHEAD  # Downlink
        else:
            if current is None or sat.requires_retasking:
                cands = sat.find_next_opportunities(n=K_AHEAD, types="target")
                current = cands[int(rng.integers(len(cands)))]["object"]
            action = current
        _, r, term, trunc, _ = env.step(action)
        total += r
        done = term or trunc
    t_end, n_img = env.simulator.sim_time, sat.imaged
    env.close()  # chiude il simulatore e completa il file di replay
    print(f"episodio finito a t={t_end:.0f} s, immagini={n_img}, reward={total:.3f}")
    print(f"replay: {out.resolve()} ({out.stat().st_size / 1e6:.1f} MB)" if out.exists() else "replay NON creato")


if __name__ == "__main__":
    main()
