"""Validatore BSK-RL del modello astratto AEOSSP (src/aeossp).

Confronta il livello astratto con BSK-RL (gravita' completa + dinamica d'assetto) in tre modi:
  geometry   posizioni e finestre di accesso: propagatori `NumericalJ2` e `KeplerJ2` contro la traiettoria di BSK-RL
  slew       tempo reale fra immagini consecutive di BSK-RL in funzione dell'angolo di puntamento, e stima dei parametri
             del modello di transizione (velocita' e accelerazione massime, stabilizzazione) per minimi quadrati
  execute    esegue in BSK-RL un piano prodotto dal modello astratto e misura quanti task riescono davvero (con il
             modello di transizione neutro di default e con quello calibrato)
Uso: uv run python scripts/validate_aeossp.py {geometry|slew|execute|all} [--hours 8] [--out outputs/validation]
I risultati (JSON) sono in outputs/validation/. Il validatore usa BSK-RL come *riferimento*, non come verita'.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config_io                                              # noqa: E402
from constellation import CH, Sim                            # noqa: E402
from ground import ensure_windows, windows_between           # noqa: E402

from aeossp import geometry as G                             # noqa: E402
from aeossp.bsk_adapter import BskTrajectory                 # noqa: E402
from aeossp.model import SlewModel                           # noqa: E402

EPOCH = "2025-03-21T00:00:00Z"
ORBIT = dict(type="keplerian", altitude_km=500, inclination_deg=55, raan_deg=40, true_anomaly_deg=10)


def random_targets(n, seed):
    rng = np.random.default_rng(seed)
    return np.degrees(np.arcsin(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n)


def bsk_sim(orbit=ORBIT, hours=8.0, round_s=2.0, min_elev_deg=30.0, targets=None, n_random=1, seed=4):
    """Sim di BSK-RL con un satellite agile. `targets` = (lat, lon) arrays -> richieste da file (arrivo 0).
    Memoria e batteria molto grandi: la validazione riguarda geometria e manovre, non i budget."""
    cons = dict(schema="constellation/v1", name="validazione", satellites=[dict(name="s1", orbit=orbit)],
                instrument=dict(min_elevation_deg=min_elev_deg), resources=dict(storage_capacity_gbit=1000, battery_capacity_wh=100000),
                ground_stations=[dict(name="G", lat=0, lon=0, min_elevation_deg=10)], comms=dict(uplink="continuous"))
    tg = dict(mode="file") if targets is not None else dict(mode="random", count=n_random)
    scen = dict(schema="scenario/v1", seed=seed, epoch=EPOCH, duration_hours=hours, round_s=round_s, track_dt=round_s, targets=tg)
    plan = config_io.build_plan(config_io.load_constellation(cons), config_io.load_scenario(scen))
    reqs = None
    if targets is not None:
        reqs = config_io.load_requests(dict(schema="requests/v1", requests=[
            dict(id=f"R{i}", target=dict(lat=float(a), lon=float(b)), priority=0.5) for i, (a, b) in enumerate(zip(*targets))]))["requests"]
    sim = Sim(plan=plan, requests=reqs)
    sim.reset(seed=seed)
    return sim, plan


# ---------------------------------------------------------------- geometria

def compare_geometry(hours=24.0, n_targets=300, min_elev_deg=50.0, orbit=ORBIT):
    sim, plan = bsk_sim(orbit, hours)
    H = hours * 3600.0
    bsk = BskTrajectory(sim.sats[0], H)
    ep = G.parse_epoch(EPOCH)
    orb = G.Orbit.from_spec(plan["sats"][0]["orbit"], ep)
    lat, lon = random_targets(n_targets, 2)
    rt = G.lla_to_ecef(lat, lon)
    t = np.arange(0, H, 30.0)
    out = {}
    for name, prov in (("NumericalJ2", G.NumericalJ2(orb, ep, H)), ("KeplerJ2", G.KeplerJ2(orb, ep))):
        err = np.linalg.norm(prov.ecef(t) - bsk.ecef(t), axis=1) / 1e3
        wm = G.find_windows(prov, rt, min_elev_deg * G.D2R, 0, H)
        wb = G.find_windows(bsk, rt, min_elev_deg * G.D2R, 0, H)
        d_open, d_close, unmatched = [], [], 0
        for a, b in zip(wm, wb):
            for o1, c1 in a:
                best = min(b, key=lambda w: abs((w[0] + w[1]) / 2 - (o1 + c1) / 2), default=None)
                if best is None or abs((best[0] + best[1]) / 2 - (o1 + c1) / 2) > 300:
                    unmatched += 1
                    continue
                d_open.append(o1 - best[0])
                d_close.append(c1 - best[1])
        out[name] = dict(pos_err_km_max=float(err.max()), pos_err_km_at_end=float(err[-1]), pos_err_km_mean=float(err.mean()),
                         windows=sum(map(len, wm)), windows_bsk=sum(map(len, wb)), unmatched=unmatched,
                         d_open_mean_abs_s=float(np.mean(np.abs(d_open))) if d_open else None,
                         d_open_max_abs_s=float(np.max(np.abs(d_open))) if d_open else None,
                         d_close_mean_abs_s=float(np.mean(np.abs(d_close))) if d_close else None)
    return dict(hours=hours, orbit=orbit, min_elev_deg=min_elev_deg, providers=out)


# ---------------------------------------------------------------- slew

def measure_slew(hours=8.0, n_targets=500, round_s=2.0, min_elev_deg=30.0, seed=3, rs_seed=5):
    """Pilota BSK-RL: comanda target visibili a caso uno dopo l'altro e registra il tempo fra immagini consecutive
    (comando emesso subito dopo la precedente). Restituisce righe (angolo [deg], tempo [s]); l'angolo e' quello
    inerziale fra le linee di vista (propagatore NumericalJ2) a fine del primo e inizio del secondo task."""
    lat, lon = random_targets(n_targets, seed)
    sim, plan = bsk_sim(ORBIT, hours, round_s, min_elev_deg, targets=(lat, lon))
    sat, env, H = sim.sats[0], sim.env, hours * 3600.0
    ensure_windows(sat, H)
    rs = np.random.default_rng(rs_seed)
    events, t_prev, cur, cmd_t, win = [], None, None, 0.0, None
    while sim.t < H - 60:
        t = sim.t
        if cur is None:
            vis = [o for o in windows_between(sat, t, t + 400, "target")
                   if o["window"][0] <= t + 2 and o["window"][1] > t + 90 and o["object"] not in sat.data_store.data.imaged]
            if not vis:
                env.step([CH])
                continue
            o = vis[int(rs.integers(len(vis)))]
            cur, cmd_t, win = o["object"], t, o["window"]
        before = sat.imaged
        env.step([cur])
        if sat.imaged > before:
            events.append((sim.t, cmd_t, sim.tindex[cur], t_prev if t_prev is not None else -1.0))
            t_prev, cur = sim.t, None
        elif sim.t > win[1] - 3 or sim.t - cmd_t > 400:
            cur, t_prev = None, None
    ep = G.parse_epoch(EPOCH)
    prov = G.NumericalJ2(G.Orbit.from_spec(plan["sats"][0]["orbit"], ep), ep, H)
    rt = G.lla_to_ecef(lat, lon)

    def los(i, tt):
        v = prov.ecef_to_eci(rt[i:i + 1], tt)[0] - prov.eci(tt)[0]
        return v / np.linalg.norm(v)

    rows = []
    for k in range(1, len(events)):
        t_img, cmd, i, tp = events[k]
        if tp < 0 or abs(cmd - tp) > 1.5 * round_s or abs(events[k - 1][0] - tp) > 1e-6:
            continue                                                  # solo coppie con comando subito dopo la precedente
        a = math.degrees(math.acos(max(-1.0, min(1.0, float(los(int(events[k - 1][2]), tp) @ los(int(i), t_img))))))
        rows.append((a, t_img - tp))
    return np.array(rows), len(events)


def fit_slew(rows, x0=(1.0, 0.5, 30.0)):
    """Minimi quadrati robusti (soft-L1) di (velocita' max [deg/s], accelerazione max [deg/s^2], stabilizzazione [s])."""
    from scipy.optimize import least_squares
    ang, dt = np.radians(rows[:, 0]), rows[:, 1]

    def model(p):
        return SlewModel(math.exp(p[0]), math.exp(p[1]), p[2]).transition(ang)

    r = least_squares(lambda p: model(p) - dt, x0=[math.log(math.radians(x0[0])), math.log(math.radians(x0[1])), x0[2]],
                      loss="soft_l1", f_scale=10.0)
    w, a, s = math.exp(r.x[0]), math.exp(r.x[1]), float(r.x[2])
    res = model(r.x) - dt
    return dict(max_rate_deg_s=math.degrees(w), max_accel_deg_s2=math.degrees(a), settle_s=s, rms_s=float(np.sqrt((res ** 2).mean())),
                median_abs_res_s=float(np.median(np.abs(res))), bias_s=float(res.mean()), n=int(len(dt)))


def slew_report(hours=8.0, **kw):
    rows, n_img = measure_slew(hours, **kw)
    out = dict(hours=hours, images=int(n_img), pairs=int(len(rows)))
    if len(rows) >= 8:
        out["fit"] = fit_slew(rows)
        d = SlewModel().transition(np.radians(rows[:, 0])) - rows[:, 1]
        out["neutral_default"] = dict(bias_s=float(d.mean()), rms_s=float(np.sqrt((d ** 2).mean())))
        bins = []
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 70), (70, 180)):
            m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
            if m.any():
                bins.append(dict(angle_deg=[lo, hi], n=int(m.sum()), mean_s=float(rows[m, 1].mean()), min_s=float(rows[m, 1].min()), max_s=float(rows[m, 1].max())))
        out["by_angle"] = bins
    out["rows"] = rows.tolist()
    return out


# ---------------------------------------------------------------- esecuzione di un piano

def execute_plan(sim, inst, plan_tasks, tols=(0.0, 5.0, 30.0), round_s=2.0):
    """Esegue in BSK-RL i task di un satellite (indici dell'istanza, in ordine). Il comando del task j parte quando
    finisce il precedente *del piano* (o subito se il tempo e' gia' passato); l'immagine riesce se BSK la fa entro
    fine OTW + tolleranza. Restituisce la distribuzione della latenza (tempo dell'immagine - inizio OTW)."""
    sat, env = sim.sats[0], sim.env
    tmap = {r["id"]: r["target"] for r in sim.reqs}
    ensure_windows(sat, inst.horizon_s)
    lat_s, done_at = [], []
    prev_end = 0.0
    for p in plan_tasks:
        tg = tmap[f"R{int(inst.target[p])}"]
        t0, t1 = float(inst.t0[p]), float(inst.t1[p])
        while sim.t < max(prev_end, 0.0) - 1e-9 and sim.t < inst.horizon_s - 10:
            env.step([CH])
        if sim.t >= inst.horizon_s - 10:
            break
        deadline = t1 + max(tols)
        got = None
        while sim.t < deadline:
            before = sat.imaged
            if not any(o["object"] is tg for o in sat.upcoming_opportunities):
                break                                               # finestra chiusa: non comandabile
            env.step([tg])
            if sat.imaged > before:
                got = sim.t
                break
        done_at.append(got)
        lat_s.append(None if got is None else got - t0)
        prev_end = t1
    n = len(done_at)
    res = dict(tasks=n, imaged=sum(g is not None for g in done_at))
    for tol in tols:
        res[f"on_time_tol_{tol:g}s"] = sum(1 for p, g in zip(plan_tasks, done_at) if g is not None and g <= inst.t1[p] + tol) / max(n, 1)
    lats = [x for x in lat_s if x is not None]
    res["latency_median_s"] = float(np.median(lats)) if lats else None
    res["latency_max_s"] = float(np.max(lats)) if lats else None
    return res


def execute_report(hours=4.0, n_targets=400, fitted=None, min_elev_deg=30.0, step_s=15.0):
    """Costruisce un'istanza (un satellite, stessa orbita di BSK-RL), ne ricava un piano con il modello neutro e con quello
    calibrato, e lo fa eseguire a BSK-RL."""
    from aeossp.config import load_config
    from aeossp.instance import build_instance
    from aeossp.solvers import greedy
    lat, lon = random_targets(n_targets, 7)
    out = {}
    models = {"neutro": dict(max_rate_deg_s=1.0, max_accel_deg_s2=0.5, settle_s=5.0)}
    if fitted:
        models["calibrato"] = {k: fitted[k] for k in ("max_rate_deg_s", "max_accel_deg_s2", "settle_s")}
    for label, slew in models.items():
        cfg = load_config(dict(
            schema="model/v1", seed=4, epoch=EPOCH, horizon_hours=hours,
            satellite_defaults=dict(min_elevation_deg=min_elev_deg, observation_s=2.0, slew=slew,
                                    resources=dict(memory_capacity_gbit=1000, energy_capacity_wh=100000)),
            satellites=[dict(name="s1", orbit=dict(ORBIT))], targets=dict(mode="file", path=None) if False else dict(mode="random", count=1),
            discretization=dict(step_s=step_s)))
        # target esplicitamente quelli di BSK-RL
        from aeossp.instance import Targets, cell_index
        T = Targets(names=[f"R{i}" for i in range(len(lat))], lat=lat, lon=lon, priority=np.ones(len(lat)), n_max=np.ones(len(lat), int),
                    arrival=np.zeros(len(lat)), expires=np.full(len(lat), np.inf), cls=cell_index(lat, lon, 30.0), gain=np.ones(1))
        inst = build_instance(cfg, targets=T)
        sch = greedy(inst)[0]
        sim, _ = bsk_sim(ORBIT, hours, 2.0, min_elev_deg, targets=(lat, lon))
        res = execute_plan(sim, inst, sch)
        res["planned"] = len(sch)
        out[label] = res
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["geometry", "slew", "execute", "all"])
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "validation"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    if a.what in ("geometry", "all"):
        res["geometry"] = compare_geometry(a.hours or 24.0)
        print("== geometria ==")
        for k, v in res["geometry"]["providers"].items():
            print(f"  {k:12s} errore posizione max {v['pos_err_km_max']:.1f} km (a fine {v['pos_err_km_at_end']:.1f}); finestre {v['windows']} vs {v['windows_bsk']} BSK, "
                  f"senza corrispondenza {v['unmatched']}; |d apertura| media {v['d_open_mean_abs_s']} s")
    fit = None
    if a.what in ("slew", "all"):
        res["slew"] = slew_report(a.hours or 8.0)
        s = res["slew"]
        print(f"== slew == {s['images']} immagini, {s['pairs']} coppie consecutive")
        for b in s.get("by_angle", []):
            print(f"  angolo {b['angle_deg'][0]:3d}-{b['angle_deg'][1]:3d} gradi: n={b['n']:3d} media {b['mean_s']:.1f} s (min {b['min_s']:.0f}, max {b['max_s']:.0f})")
        if "fit" in s:
            fit = s["fit"]
            print(f"  stima: {fit['max_rate_deg_s']:.2f} deg/s, {fit['max_accel_deg_s2']:.2f} deg/s^2, stabilizzazione {fit['settle_s']:.1f} s | RMS {fit['rms_s']:.1f} s, "
                  f"mediana |res| {fit['median_abs_res_s']:.1f} s; modello neutro: bias {s['neutral_default']['bias_s']:.1f} s, RMS {s['neutral_default']['rms_s']:.1f} s")
    if a.what in ("execute", "all"):
        if fit is None and (out / "validation.json").exists():
            fit = json.loads((out / "validation.json").read_text()).get("slew", {}).get("fit")
        res["execute"] = execute_report(a.hours or 4.0, fitted=fit)
        print("== esecuzione in BSK-RL ==")
        for k, v in res["execute"].items():
            print(f"  modello {k:9s}: pianificati {v['planned']}, eseguiti {v['tasks']}, immagini {v['imaged']}; in tempo (tol 0 s) {v['on_time_tol_0s']:.0%}, "
                  f"(5 s) {v['on_time_tol_5s']:.0%}, (30 s) {v['on_time_tol_30s']:.0%}; latenza mediana {v['latency_median_s']} s")
    path = out / "validation.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1))
    print(f"risultati: {path}")


if __name__ == "__main__":
    main()
