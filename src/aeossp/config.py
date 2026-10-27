"""File di configurazione `model/v1`: validazione e normalizzazione.

Come in `scripts/config_io.py`: ogni chiave sconosciuta e' un errore, gli errori si raccolgono tutti insieme
(`ConfigError.errors`, con il percorso del campo) e le chiavi che iniziano con `_` sono commenti.
Le orbite hanno lo stesso formato dei file `constellation/v1`.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

from . import geometry as G
from .model import Resources, SatSpec, SlewModel

D2R = math.pi / 180.0
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
AGILITY = ("agile", "none")
DOWNLINK = ("none", "stations")
ORBIT_MODELS = ("numerical_j2", "kepler_j2")
SCOPES = ("cycle", "horizon")
OUTCOMES = ("deterministic", "cell_bernoulli")
# flussi casuali indipendenti derivati dal seed (chiave fissa per flusso: aggiungerne non sposta gli altri)
STREAMS = dict(arrivals=2, space=3, priority=4, outcomes=5, clouds=6, policy=7, quality=8)


def rng_stream(seed, name, sub=0):
    """Generator indipendente dal seed e dal nome del flusso (mai np.random globale)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(STREAMS[name], sub)))


class ConfigError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class _Check:
    def __init__(self):
        self.errors = []

    def err(self, path, msg):
        self.errors.append(f"{path or '<radice>'}: {msg}")

    def obj(self, v, path, allowed):
        if v is None:
            return {}
        if not isinstance(v, dict):
            self.err(path, "deve essere un oggetto")
            return {}
        for k in v:
            if k not in allowed and not str(k).startswith("_"):
                self.err(f"{path}.{k}" if path else k, f"chiave sconosciuta (ammesse: {', '.join(sorted(allowed))})")
        return v

    def num(self, d, key, path, default, lo=None, hi=None, integer=False, lo_open=False):
        v = d.get(key, default)
        p = f"{path}.{key}" if path else key
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            self.err(p, "deve essere un numero")
            return default
        if integer and v != int(v):
            self.err(p, "deve essere un intero")
        if lo is not None and (v <= lo if lo_open else v < lo):
            self.err(p, f"deve essere {'>' if lo_open else '>='} {lo}")
        if hi is not None and v > hi:
            self.err(p, f"deve essere <= {hi}")
        return int(v) if integer else float(v)

    def choice(self, d, key, path, default, options):
        v = d.get(key, default)
        if v not in options:
            self.err(f"{path}.{key}" if path else key, f"valori ammessi: {', '.join(options)}")
            return default
        return v


ORBIT_KEYS = {"type", "altitude_km", "inclination_deg", "raan_deg", "eccentricity", "arg_perigee_deg",
              "true_anomaly_deg", "ltdn_hours"}
SAT_KEYS = {"name", "orbit", "agility", "observation_s", "min_elevation_deg", "max_offnadir_deg", "slew", "resources"}
SLEW_KEYS = {"max_rate_deg_s", "max_accel_deg_s2", "settle_s"}
RES_KEYS = {"memory_capacity_gbit", "data_rate_mbps", "energy_capacity_wh", "imaging_power_w", "maneuver_power_w"}
WALKER_KEYS = {"sats", "planes", "phasing", "altitude_km", "inclination_deg"}
TOP_KEYS = {"schema", "name", "description", "seed", "epoch", "horizon_hours", "cycle_s", "downlink", "orbit_model", "satellite_defaults",
            "satellites", "walker", "targets", "classes", "discretization", "budgets", "feedback", "outcomes"}
TARGET_KEYS = {"mode", "count", "initial_count", "arrival", "priority", "n_max", "gain", "validity_hours", "path"}
OUTCOME_KEYS = {"model", "p_min", "p_max", "quality_min", "quality_max", "offnadir_penalty", "change_points"}


def _orbit(c, o, path, epoch):
    o = c.obj(o, path, ORBIT_KEYS)
    kind = o.get("type")
    if kind not in ("keplerian", "sun_synchronous"):
        c.err(f"{path}.type", "valori ammessi: keplerian, sun_synchronous")
        return None
    alt = c.num(o, "altitude_km", path, None, lo=160, hi=2000)
    if alt is None:
        c.err(f"{path}.altitude_km", "obbligatorio")
        return None
    spec = dict(type=kind, altitude_km=alt)
    if kind == "keplerian":
        spec["inclination_deg"] = c.num(o, "inclination_deg", path, None, lo=0, hi=180)
        spec["raan_deg"] = c.num(o, "raan_deg", path, 0.0, lo=0, hi=360)
        if spec["inclination_deg"] is None:
            c.err(f"{path}.inclination_deg", "obbligatorio per type=keplerian")
            return None
    else:
        spec["ltdn_hours"] = c.num(o, "ltdn_hours", path, None, lo=0, hi=24)
        if spec["ltdn_hours"] is None:
            c.err(f"{path}.ltdn_hours", "obbligatorio per type=sun_synchronous")
            return None
    spec["eccentricity"] = c.num(o, "eccentricity", path, 0.0, lo=0, hi=0.2)
    spec["arg_perigee_deg"] = c.num(o, "arg_perigee_deg", path, 0.0, lo=0, hi=360)
    spec["true_anomaly_deg"] = c.num(o, "true_anomaly_deg", path, 0.0, lo=0, hi=360)
    try:
        return G.Orbit.from_spec(spec, epoch)
    except ValueError as e:
        c.err(path, str(e))
        return None


def _sat(c, s, path, epoch, defaults):
    s = c.obj(s, path, SAT_KEYS)
    m = {**defaults, **{k: v for k, v in s.items() if k not in ("slew", "resources")}}
    m["slew"] = {**defaults.get("slew", {}), **(s.get("slew") or {})}
    m["resources"] = {**defaults.get("resources", {}), **(s.get("resources") or {})}
    name = m.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        c.err(f"{path}.name", "obbligatorio: identificatore (lettere, cifre, _)")
        name = "sat"
    orbit = _orbit(c, m.get("orbit"), f"{path}.orbit", epoch)
    agility = c.choice(m, "agility", path, "agile", AGILITY)
    obs = c.num(m, "observation_s", path, 2.0, lo=0, hi=600, lo_open=True)
    sl = c.obj(m["slew"], f"{path}.slew", SLEW_KEYS)
    slew = SlewModel(max_rate=c.num(sl, "max_rate_deg_s", f"{path}.slew", 1.0, lo=0, lo_open=True, hi=30) * D2R,
                     max_accel=c.num(sl, "max_accel_deg_s2", f"{path}.slew", 0.5, lo=0, lo_open=True, hi=30) * D2R,
                     settle_s=c.num(sl, "settle_s", f"{path}.slew", 5.0, lo=0, hi=600))
    rs = c.obj(m["resources"], f"{path}.resources", RES_KEYS)
    res = Resources(
        memory_capacity_gbit=c.num(rs, "memory_capacity_gbit", f"{path}.resources", 20.0, lo=0, lo_open=True),
        data_rate_mbps=c.num(rs, "data_rate_mbps", f"{path}.resources", 800.0, lo=0, lo_open=True),
        energy_capacity_wh=c.num(rs, "energy_capacity_wh", f"{path}.resources", 5.0, lo=0, lo_open=True),
        imaging_power_w=c.num(rs, "imaging_power_w", f"{path}.resources", 100.0, lo=0),
        maneuver_power_w=c.num(rs, "maneuver_power_w", f"{path}.resources", 200.0, lo=0))
    if "min_elevation_deg" in m and "max_offnadir_deg" in m:
        c.err(path, "specificare min_elevation_deg oppure max_offnadir_deg, non entrambi")
    if "max_offnadir_deg" in m and orbit is not None:
        off = c.num(m, "max_offnadir_deg", path, None, lo=0, hi=89)
        el = G.min_elevation_from_offnadir(off, orbit.a - G.RE)
    else:
        el = c.num(m, "min_elevation_deg", path, 40.0, lo=0, hi=90)
    if orbit is None:
        return None
    return SatSpec(name=name, orbit=orbit, agility=agility, observation_s=obs, min_elevation_deg=el, slew=slew, resources=res)


def _walker(c, w, epoch, defaults):
    w = c.obj(w, "walker", WALKER_KEYS)
    T = c.num(w, "sats", "walker", None, lo=1, integer=True)
    P = c.num(w, "planes", "walker", 1, lo=1, integer=True)
    F = c.num(w, "phasing", "walker", 0, lo=0, integer=True)
    alt = c.num(w, "altitude_km", "walker", 500.0, lo=160, hi=2000)
    inc = c.num(w, "inclination_deg", "walker", 55.0, lo=0, hi=180)
    if not T or T % P:
        c.err("walker.sats", "obbligatorio e multiplo di planes")
        return []
    per = T // P
    out = []
    for i in range(T):
        p, s = divmod(i, per)
        spec = dict(name=f"sat_{i + 1}", orbit=dict(type="keplerian", altitude_km=alt, inclination_deg=inc, raan_deg=360.0 * p / P,
                                                    true_anomaly_deg=(360.0 * s / per + 360.0 * F * p / T) % 360.0))
        out.append(_sat(c, spec, f"walker[{i}]", epoch, defaults))
    return out


def _requests(c, path, base):
    try:
        data = json.loads(Path(path if Path(path).is_absolute() else Path(base) / path).read_text())
    except (OSError, ValueError) as e:
        c.err("targets.path", f"file di richieste illeggibile: {e}")
        return []
    if not isinstance(data, dict) or data.get("schema") != "requests/v1":
        c.err("targets.path", "atteso un file requests/v1")
        return []
    out = []
    for i, r in enumerate(data.get("requests", [])):
        p, t = f"requests[{i}]", r.get("target", {})
        try:
            pr = float(r["priority"])
            rid = str(r.get("id", f"R{i + 1}"))
            lat, lon = float(t["lat"]), float(t["lon"])
        except (KeyError, TypeError, ValueError):
            c.err(p, "richiesta non valida (servono id, target.lat/lon, priority)")
            continue
        if not RID.match(rid) or pr <= 0 or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            c.err(p, "valori non validi")
            continue
        arr = float(r.get("arrival_s", 0.0))
        exp = r.get("expires_s", arr + r["validity_s"] if "validity_s" in r else None)
        out.append(dict(name=str(t.get("name", rid)), lat=lat, lon=lon, priority=pr, arrival_s=arr,
                        expires_s=None if exp is None else float(exp)))
    return out


def load_config(src, base_dir=None):
    """Legge e valida un file `model/v1` (percorso o dizionario). Restituisce un dizionario normalizzato."""
    if not isinstance(src, dict):
        path = Path(src)
        base_dir = base_dir or path.parent
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise ConfigError([f"{path}: {e}"])
    else:
        data = src
    c = _Check()
    top = c.obj(data, "", TOP_KEYS)
    if top.get("schema") != "model/v1":
        c.err("schema", "atteso 'model/v1'")
    seed = c.num(top, "seed", "", 4, lo=0, integer=True)
    try:
        epoch = G.parse_epoch(top.get("epoch", "2025-03-21T00:00:00Z"))
    except ValueError:
        c.err("epoch", "formato non valido, atteso ISO 8601")
        epoch = G.parse_epoch("2025-03-21T00:00:00Z")
    hours = c.num(top, "horizon_hours", "", 24.0, lo=0, lo_open=True, hi=24 * 30)
    downlink = c.choice(top, "downlink", "", "none", DOWNLINK)
    orbit_model = c.choice(top, "orbit_model", "", "numerical_j2", ORBIT_MODELS)
    if downlink == "stations":
        c.err("downlink", "«stations» e' riservato: il livello astratto non lo modella ancora (resta nel simulatore BSK-RL, scripts/ground.py)")
    defaults = c.obj(top.get("satellite_defaults"), "satellite_defaults", SAT_KEYS - {"name"})
    sats = []
    if "walker" in top and "satellites" in top:
        c.err("walker", "usare walker oppure satellites, non entrambi")
    elif "walker" in top:
        sats = _walker(c, top["walker"], epoch, defaults)
    elif isinstance(top.get("satellites"), list) and top["satellites"]:
        sats = [_sat(c, s, f"satellites[{i}]", epoch, defaults) for i, s in enumerate(top["satellites"])]
        names = [s.name for s in sats if s]
        if len(set(names)) != len(names):
            c.err("satellites", "nomi duplicati")
    else:
        c.err("satellites", "obbligatorio (lista non vuota) oppure walker")
    sats = [s for s in sats if s]

    t = c.obj(top.get("targets"), "targets", TARGET_KEYS)
    mode = c.choice(t, "mode", "targets", "random", ("random", "stream", "file"))
    pr = c.obj(t.get("priority"), "targets.priority", {"min", "max"})
    pmin, pmax = c.num(pr, "min", "targets.priority", 0.1, lo=0, lo_open=True), c.num(pr, "max", "targets.priority", 1.0, lo=0, lo_open=True)
    if pmin > pmax:
        c.err("targets.priority", "min > max")
    n_max = c.num(t, "n_max", "targets", 1, lo=1, integer=True, hi=20)
    gain = t.get("gain")
    if gain is None:
        gain = [1.0] * n_max
    elif not (isinstance(gain, list) and len(gain) == n_max and all(isinstance(g, (int, float)) and g >= 0 for g in gain)):
        c.err("targets.gain", f"lista di {n_max} guadagni marginali >= 0")
        gain = [1.0] * n_max
    tg = dict(mode=mode, priority=(pmin, pmax), n_max=n_max, gain=[float(g) for g in gain])
    if mode == "random":
        tg["count"] = c.num(t, "count", "targets", 200, lo=0, integer=True, hi=50000)
    elif mode == "stream":
        ar = c.obj(t.get("arrival"), "targets.arrival", {"type", "rate_per_hour"})
        tg["initial_count"] = c.num(t, "initial_count", "targets", 0, lo=0, integer=True)
        tg["rate_per_hour"] = c.num(ar, "rate_per_hour", "targets.arrival", None, lo=0, lo_open=True)
        if tg["rate_per_hour"] is None:
            c.err("targets.arrival.rate_per_hour", "obbligatorio per mode=stream")
        v = c.num(t, "validity_hours", "targets", None, lo=0, lo_open=True)
        tg["validity_s"] = None if v is None else v * 3600.0
    else:
        if "path" not in t:
            c.err("targets.path", "obbligatorio per mode=file")
        else:
            tg["requests"] = _requests(c, t["path"], base_dir or ".")
    cl = c.obj(top.get("classes"), "classes", {"cell_deg"})
    cell = c.num(cl, "cell_deg", "classes", 30.0, lo=1, hi=180, lo_open=True)
    ds = c.obj(top.get("discretization"), "discretization", {"step_s"})
    step = c.num(ds, "step_s", "discretization", 10.0, lo=0.5, hi=3600)
    bud = c.obj(top.get("budgets"), "budgets", {"scope"})
    scope = c.choice(bud, "scope", "budgets", "cycle", SCOPES)
    fb = c.obj(top.get("feedback"), "feedback", {"latency_rounds"})
    latency = c.num(fb, "latency_rounds", "feedback", 0, lo=0, integer=True, hi=1000)
    oc = c.obj(top.get("outcomes"), "outcomes", OUTCOME_KEYS)
    out = dict(model=c.choice(oc, "model", "outcomes", "deterministic", OUTCOMES),
               p_min=c.num(oc, "p_min", "outcomes", 0.2, lo=0, hi=1), p_max=c.num(oc, "p_max", "outcomes", 0.95, lo=0, hi=1),
               quality_min=c.num(oc, "quality_min", "outcomes", 0.6, lo=0, hi=1), quality_max=c.num(oc, "quality_max", "outcomes", 1.0, lo=0, hi=1),
               offnadir_penalty=c.num(oc, "offnadir_penalty", "outcomes", 0.0, lo=0, hi=1))
    cps = oc.get("change_points", [])
    if not (isinstance(cps, list) and all(isinstance(x, (int, float)) and x > 0 for x in cps)):
        c.err("outcomes.change_points", "lista di istanti [s] > 0")
        cps = []
    out["change_points"] = sorted(float(x) for x in cps)
    cycle = c.num(top, "cycle_s", "", None, lo=0, lo_open=True)
    if c.errors:
        raise ConfigError(c.errors)
    return dict(name=str(top.get("name", "model")), seed=seed, epoch=epoch, horizon_s=hours * 3600.0, cycle_s=cycle,
                downlink=downlink, orbit_model=orbit_model, sats=sats, targets=tg, cell_deg=cell, step_s=step, budget_scope=scope,
                latency_rounds=latency, outcomes=out)
