"""Pilot BSK-RL: misura velocità e verifica alcune assunzioni del CLAUDE.md.

Verifica (senza costruire ancora il wrapper bandit):
  1. throughput (step/s e secondi simulati per secondo reale) con round a durata
     variabile (default BSK-RL) e a durata fissa (variable_interval=False);
  2. riproducibilità con lo stesso seed;
  3. identità dei target tra episodi (Target.id vs Target.name);
  4. effetto di Downlink sul buffer immagini (serve per la coda di reward pendenti).

Uso: uv run python scripts/pilot_bskrl.py [--episodes N] [--targets N]
"""

import argparse
import time

import gymnasium as gym
import numpy as np

from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw

K_AHEAD = 8  # numero di target "prossimi" esposti come azioni Image
ORBIT = 5700.0  # [s] circa un'orbita LEO


def make_sat(variable_interval: bool, name: str = "sat_1", sat_args: dict | None = None,
             observe_opportunities: bool = True) -> sats.ImagingSatellite:
    """observe_opportunities=False: modalita' dei file di configurazione. L'osservazione non include i prossimi target
    e BSK-RL non calcola alcuna finestra (le calcola `ground.WindowEngine`, vedi li' il motivo)."""
    class PilotSat(sats.ImagingSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
            ),
            *([obs.OpportunityProperties(
                dict(prop="priority"),
                dict(prop="opportunity_close", norm=ORBIT),
                n_ahead_observe=K_AHEAD,
            )] if observe_opportunities else []),
            obs.Time(),
        ]
        # indici: 0..K-1 Image, K Downlink, K+1 Charge
        action_spec = [
            act.Image(n_ahead_image=K_AHEAD),
            act.Downlink(duration=60.0),
            act.Charge(duration=60.0),
        ]
        dyn_type = dyn.FullFeaturedDynModel
        fsw_type = fsw.ImagingFSWModel

    extra = {} if observe_opportunities else dict(initial_generation_duration=0, max_generation_duration_beyond_initial=0)
    return PilotSat(name, sat_args=sat_args, variable_interval=variable_interval, **extra)


def make_env(variable_interval: bool, n_targets: int, time_limit: float, step: float):
    return gym.make(
        "SatelliteTasking-v1",
        satellite=make_sat(variable_interval),
        scenario=scene.UniformTargets(n_targets),
        rewarder=data.UniqueImageReward(),
        time_limit=time_limit,
        max_step_duration=step,
        log_level="ERROR",
        disable_env_checker=True,
    ).unwrapped


def rollout(env, seed: int, policy_seed: int, p_dl: float = 0.15):
    """Un episodio con politica casuale (Generator proprio, indipendente dal seed env).

    L'azione e' un oggetto Target (identita' stabile, via set_action_override), che la
    politica ripresenta a ogni round finche' il satellite non chiede un nuovo task.
    """
    rng = np.random.default_rng(policy_seed)
    t0 = time.perf_counter()
    env.reset(seed=seed)
    reset_s = time.perf_counter() - t0
    sat = env.satellites[0]
    current, steps, total, rewards, done = None, 0, 0.0, [], False
    while not done:
        if rng.random() < p_dl:
            action = K_AHEAD  # indice Downlink
        else:
            if current is None or sat.requires_retasking:
                cands = sat.find_next_opportunities(n=K_AHEAD, types="target")
                current = cands[int(rng.integers(len(cands)))]["object"]
            action = current
        _, r, term, trunc, _ = env.step(action)
        steps += 1
        total += r
        rewards.append(r)
        done = term or trunc
    return steps, total, rewards, env.simulator.sim_time, reset_s


def bench(label, variable_interval, episodes, n_targets, step):
    env = make_env(variable_interval, n_targets, 3 * ORBIT, step)
    rollout(env, seed=999, policy_seed=999)  # warmup (download dati, import lazy)
    steps = sim_s = n_img = 0
    reset_t = step_t = 0.0
    for ep in range(episodes):
        t0 = time.perf_counter()
        s, _, rw, t, rs = rollout(env, seed=ep, policy_seed=ep)
        step_t += time.perf_counter() - t0 - rs
        reset_t += rs
        steps += s
        sim_s += t
        n_img += sum(1 for x in rw if x > 0)
    print(f"{label:30s} step/s={steps / step_t:7.1f}  sim-s/s(step)={sim_s / step_t:8.0f}  "
          f"step/ep={steps / episodes:6.1f}  reset={reset_t / episodes:5.2f}s/ep  "
          f"round con reward>0/ep={n_img / episodes:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--targets", type=int, default=500)
    a = ap.parse_args()

    print(f"== 1. Throughput (1 satellite, {a.targets} target, 3 orbite per episodio)")
    bench("variable_interval=True (default)", True, a.episodes, a.targets, 1e9)
    bench("variable_interval=False, round 60 s", False, a.episodes, a.targets, 60.0)
    bench("variable_interval=False, round 300 s", False, a.episodes, a.targets, 300.0)

    print("\n== 2. Riproducibilita' (stesso seed env, stessa sequenza di azioni)")
    env = make_env(False, a.targets, 3 * ORBIT, 60.0)
    r1 = rollout(env, seed=8, policy_seed=1)
    r2 = rollout(env, seed=8, policy_seed=1)
    r3 = rollout(env, seed=9, policy_seed=1)
    print(f"round: {r1[0]} / {r2[0]} / {r3[0]}")
    print(f"stesso seed:      reward identici = {r1[2] == r2[2]}  (tot {r1[1]:.4f} vs {r2[1]:.4f})")
    print(f"seed diverso:     reward identici = {r1[2] == r3[2]}  (tot {r1[1]:.4f} vs {r3[1]:.4f})")

    print("\n== 3. Identita' dei target tra episodi")
    for seed in (8, 8, 9):
        env.reset(seed=seed)
        t = env.scenario.targets[:2]
        print(f"seed={seed}: " + ", ".join(f"{x.name} id={x.id} prio={x.priority:.3f}" for x in t))

    print("\n== 4. Immagini -> buffer -> Downlink (round 60 s)")
    env.reset(seed=4)  # seed con finestre di downlink nelle prime orbite
    sat = env.satellites[0]
    current, last_stor = None, 0.0
    for i in range(1000):
        stor = sat.dynamics.storage_level / 8e6
        dl = stor >= 3 or (last_stor > 0 and stor < last_stor and stor > 0)
        if dl:
            action = K_AHEAD
        else:
            if current is None or sat.requires_retasking:
                current = sat.find_next_opportunities(n=1, types="target")[0]["object"]
            action = current
        _, r, term, trunc, _ = env.step(action)
        new = sat.dynamics.storage_level / 8e6
        if new != stor or r > 0:
            print(f"t={env.simulator.sim_time:6.0f} {'DL ' if dl else 'IMG'} r={r:.3f} "
                  f"storage {stor:4.1f} -> {new:4.1f} img  (imaged={sat.imaged})")
        last_stor = new
        if term or trunc:
            break


if __name__ == "__main__":
    main()
