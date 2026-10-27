"""Esporta un episodio dell'ambiente nel formato di traccia v2 del viewer (`viewer/web/index.html`).

Mappatura:
  satellite.track     posizione dal fornitore di geometria (ogni `track_dt` s)
  satellite.windows   una voce per VTW (kind=target)
  satellite.rounds    segmenti contigui: un'osservazione = un segmento `image` [inizio, fine] del task;
                      i vuoti sono segmenti `idle` (divisi ai confini di ciclo, dove i budget si azzerano).
                      Lo stato a fine segmento: `storage` = memoria usata nel ciclo in unita' di immagini,
                      `battery` = energia residua del ciclo (frazione), `imaged` = osservazioni riuscite.
  satellite.imaged    osservazioni riuscite (istante di fine, indice del target)
Un'osservazione con esito negativo (nuvole/qualita') ha `why` e non incrementa `imaged`: il viewer la mostra
come «non acquisita» con quel motivo.
"""

import numpy as np

from . import geometry as G


def _track(prov, horizon, dt):
    t = np.arange(0.0, horizon + 1e-9, dt)
    r = prov.ecef(t)
    lat, lon = G.ecef_to_latlon(r)
    alt = np.linalg.norm(r, axis=1) / 1e3 - G.RE / 1e3
    return [round(float(x), 3) for x in lat], [round(float(x), 3) for x in lon], [round(float(x), 1) for x in alt]


def to_trace(env, policy="", track_dt=10.0):
    inst, oc = env.inst, env.outcomes
    H = inst.horizon_s
    fb = {f.task: f for h in env.history for f in h["feedback"]}
    # le osservazioni eseguite possono avere feedback non ancora arrivato (ritardo): l'esito e' comunque noto all'ambiente
    done_tasks = [p for h in env.history for s in h["schedule"] for p in s]
    reward_of = {}
    for h in env.history:
        for f in h["feedback"]:
            reward_of[f.task] = f.reward
    sats = []
    for k, s in enumerate(inst.sats):
        prov = inst.providers[k]
        lat, lon, alt = _track(prov, H, track_dt)
        img_gbit = s.resources.memory(s.observation_s if s.agile else 1.0)
        cap = s.resources.memory_capacity_gbit
        mine = sorted((p for p in done_tasks if inst.sat[p] == k), key=lambda p: inst.t0[p])
        rounds, imaged = [], []
        cursor, n_img, cum = 0.0, 0, 0.0
        mem = en = 0.0
        key = None

        def state(t_end, m, e):
            return dict(storage=round(m / img_gbit, 2) if img_gbit else 0.0, battery=round(max(0.0, 1.0 - e / s.resources.energy_capacity_wh), 3))

        def idle_until(t_to, mem, en):
            nonlocal cursor
            while cursor < t_to - 1e-9:
                c_end = (np.floor(cursor / inst.cycle_s + 1e-9) + 1) * inst.cycle_s if inst.budget_scope == "cycle" else H + 1
                seg_end = min(t_to, c_end)
                rounds.append(dict(t0=round(cursor, 3), t1=round(seg_end, 3), action="idle", target=None, reward=0.0, cum_reward=round(cum, 4),
                                   imaged=n_img, alive=True, **state(seg_end, mem, en)))
                if inst.budget_scope == "cycle" and seg_end >= c_end - 1e-9 and seg_end < t_to + 1e-9:
                    mem = en = 0.0                                   # nuovo ciclo: i budget ripartono
                cursor = seg_end
            return mem, en

        prev = None
        for p in mine:
            kp = inst.budget_key(p)
            if key is not None and kp != key:
                mem = en = 0.0
            key = kp
            mem, en = idle_until(float(inst.t0[p]), mem, en)
            if inst.budget_scope == "cycle" and int(inst.cycle[p]) != int(np.floor(cursor / inst.cycle_s + 1e-9)):
                mem = en = 0.0
            mem += float(inst.memory_use(p)[0])
            en += float(inst.energy_imaging(p)[0])
            if prev is not None and inst.budget_key(prev) == kp:
                en += float(inst.energy_maneuver(prev, p))
            ok = bool(oc.success[p])
            rew = reward_of.get(int(p))
            if rew is None:                                          # feedback ancora in volo: ricalcolo dal modello
                rew = float(inst.targets.priority[inst.target[p]] * oc.quality[p]) if ok else 0.0
            cum += rew
            r = dict(t0=round(float(inst.t0[p]), 3), t1=round(float(inst.t1[p]), 3), action="image", target=int(inst.target[p]), reward=round(rew, 4),
                     cum_reward=round(cum, 4), imaged=n_img + (1 if ok else 0), alive=True, **state(inst.t1[p], mem, en))
            if ok:
                n_img += 1
                imaged.append(dict(t=round(float(inst.t1[p]), 3), target=int(inst.target[p])))
            else:
                r["why"] = "esito negativo (nuvole o qualita' insufficiente)"
            rounds.append(r)
            cursor, prev = float(inst.t1[p]), p
        idle_until(H, mem, en)
        vt = {}
        for p in np.flatnonzero(inst.sat == k):
            vt.setdefault(int(inst.vtw[p]), (int(inst.target[p]), float(inst.vtw_start[p]), float(inst.vtw_end[p])))
        windows = [dict(kind="target", obj=i, t0=round(a, 1), t1=round(c, 1)) for i, a, c in sorted(vt.values(), key=lambda x: x[2])]
        o = s.orbit
        sats.append(dict(name=s.name, meta=dict(inclination_deg=round(float(np.degrees(o.inc)), 2), altitude_km=round(float((o.a - G.RE) / 1e3), 1),
                                                raan_deg=round(float(np.degrees(o.raan)), 1), image_min_elev_deg=round(float(s.min_elevation_deg), 1),
                                                storage_capacity=round(cap / img_gbit, 2) if img_gbit else 20, track_vs_sim_max_km=0.0),
                         track=dict(lat=lat, lon=lon, alt_km=alt), windows=windows, rounds=rounds, imaged=imaged, uplinks=[]))
    T = inst.targets
    targets = [dict(id=T.names[i], name=T.names[i], lat=round(float(T.lat[i]), 3), lon=round(float(T.lon[i]), 3),
                    priority=round(float(T.priority[i]), 3), arrival_s=round(float(T.arrival[i]), 1)) for i in range(len(T))]
    notices = [dict(level="info", message="Modello astratto AEOSSP (src/aeossp): il downlink e' escluso e la memoria e' un budget per ciclo orbitale, come nel paper di riferimento. "
                                          "Un'osservazione puo' riuscire o fallire (esito nascosto all'algoritmo)."),
               dict(level="info", message="Barra «Buffer immagini» = memoria usata nel ciclo corrente; «Batteria» = energia residua del ciclo (imaging + manovre).")]
    meta = dict(seed=inst.cfg["seed"], round_s=track_dt, t_end=H, n_targets=len(T), n_satellites=len(inst.sats), infinite=False,
                gs_min_elev_deg=10.0, earth_radius_km=G.RE / 1e3, track_dt=track_dt,
                constellation=f"{inst.cfg['name']}: {len(inst.sats)} satelliti, {inst.n} task candidati, {inst.n_cycles} cicli da {inst.cycle_s:.0f} s",
                note="Posizioni dal propagatore kepleriano+J2 (o dal fornitore di geometria in uso); gli istanti e le durate delle osservazioni sono quelli del modello "
                     "tempo-discreto. Campo visivo = zona con elevazione >= image_min_elev_deg; assetto e slew non sono mostrati.",
                policy=policy or "n/d", mode="aeossp", comms=dict(uplink="continuous", crosslinks="none"), dynamic=False, notices=notices)
    return dict(version=2, meta=meta, stations=[], targets=targets, satellites=sats, targets_removed=[])
