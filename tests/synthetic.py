"""Istanze sintetiche a mano per i test: geometria banale (puntamenti su un cerchio massimo) e tempi scelti
a piacere, cosi' i vincoli e i solver si verificano contro enumerazioni esaustive."""
import numpy as np

import _path  # noqa: F401
from aeossp.config import load_config
from aeossp.instance import Instance, Targets


def make(tasks, n_targets=None, n_max=1, gain=None, priority=None, sats=1, mem_cap=1e9, en_cap=1e9, data_rate=800.0,
         maneuver_w=200.0, imaging_w=100.0, cycle_s=1e9, scope="cycle", agility="agile", horizon_h=24, rate_deg=1.0, accel_deg=0.5,
         settle=5.0):
    """tasks: lista di (sat, target, t0, t1, phi_deg) con phi la direzione di puntamento (gradi, su un cerchio)."""
    n_targets = n_targets or (max(t[1] for t in tasks) + 1)
    cfg = load_config({
        "schema": "model/v1", "horizon_hours": horizon_h, "cycle_s": cycle_s,
        "satellite_defaults": {"agility": agility, "slew": {"max_rate_deg_s": rate_deg, "max_accel_deg_s2": accel_deg, "settle_s": settle},
                               "resources": {"memory_capacity_gbit": mem_cap, "data_rate_mbps": data_rate, "energy_capacity_wh": en_cap,
                                             "maneuver_power_w": maneuver_w, "imaging_power_w": imaging_w}},
        "walker": {"sats": sats, "planes": 1, "altitude_km": 500, "inclination_deg": 55},
        "targets": {"mode": "random", "count": 1, "n_max": n_max, **({"gain": gain} if gain else {})},
        "budgets": {"scope": scope}})
    g = np.array(cfg["targets"]["gain"])
    pri = np.ones(n_targets) if priority is None else np.asarray(priority, dtype=float)
    T = Targets(names=[f"t{i}" for i in range(n_targets)], lat=np.zeros(n_targets), lon=np.zeros(n_targets), priority=pri,
                n_max=np.full(n_targets, n_max), arrival=np.zeros(n_targets), expires=np.full(n_targets, np.inf),
                cls=np.zeros(n_targets, dtype=int), gain=g)
    rows = sorted(tasks, key=lambda t: (t[0], t[2], t[3], t[1]))
    sat = np.array([r[0] for r in rows], dtype=int)
    los = np.array([[np.cos(np.radians(r[4])), np.sin(np.radians(r[4])), 0.0] for r in rows])
    t0 = np.array([r[2] for r in rows], dtype=float)
    t1 = np.array([r[3] for r in rows], dtype=float)
    n = len(rows)
    arr = dict(sat=sat, target=np.array([r[1] for r in rows], dtype=int), t0=t0, t1=t1, vtw=np.arange(n), vtw_start=t0, vtw_end=t1,
               los0=los, los1=los.copy(), offnadir=np.zeros(n), elevation=np.full(n, np.pi / 2), cls=np.zeros(n, dtype=int))
    return Instance(cfg, T, None, arr)
