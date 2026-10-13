"""Esporta in JSON la traccia di un episodio BSK-RL per il viewer (viewer/).

La simulazione e la politica dimostrativa (non e' un bandit) sono in constellation.py,
condivise con lo stream live (sim_server.py). Con --sats N > 1: costellazione Walker-delta.

Uso: uv run python scripts/export_trace.py [--sats 3 --planes 3] [--seed 4] [--orbits 3]
Output: outputs/traces/trace_seed<seed>.json (1 satellite) o trace_seed<seed>_<N>sats.json
"""

import argparse
import json
from pathlib import Path

from constellation import ORBIT, Sim


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

    sim = Sim(sats=a.sats, planes=a.planes, altitude=a.altitude, inc=a.inc, targets=a.targets,
              round_s=a.round, track_dt=a.track_dt, seconds=a.orbits * ORBIT)
    sim.reset(a.seed)
    while not sim.done:
        sim.step()
    trace = sim.trace
    sim.env.close()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"trace_seed{a.seed}_{a.sats}sats.json" if a.sats > 1 else f"trace_seed{a.seed}.json")
    path.write_text(json.dumps(trace, separators=(",", ":")))
    print(f"{path} ({path.stat().st_size / 1e3:.0f} kB): t_end={trace['meta']['t_end']:.0f}s, "
          f"round={len(trace['satellites'][0]['rounds'])}")
    imaged = set()
    for s in trace["satellites"]:
        last = s["rounds"][-1]
        imaged |= {e["target"] for e in s["imaged"]}
        print(f"  {s['name']}: immagini={last['imaged']}, reward={last['cum_reward']}, vivo={last['alive']}, "
              f"incl={s['meta']['inclination_deg']}, RAAN={s['meta']['raan_deg']}, "
              f"scarto traiettoria-sim max={s['meta']['track_vs_sim_max_km']} km")
    dup = sum(len(s["imaged"]) for s in trace["satellites"]) - len(imaged)
    print(f"  target distinti ripresi={len(imaged)}, immagini duplicate={dup}")


if __name__ == "__main__":
    main()
