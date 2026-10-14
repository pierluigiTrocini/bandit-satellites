"""Caricamento, validazione e traduzione in argomenti BSK-RL dei file di configurazione.

File JSON (schemi in configs/README.md):
  constellation/v1  satelliti (orbita, strumento, risorse, downlink), stazioni a terra con ruolo
                    (uplink/downlink) e collegamenti (`comms`)
  scenario/v1       episodio: seed, epoca, durata, round, richieste di imaging (target casuali fissi,
                    flusso Poisson o file) e pianificazione a terra; la non stazionarieta' e' riservata
                    (chiave `non_stationarity`, per ora deve essere assente o null)
  requests/v1       elenco di richieste di imaging (id, target, priorita', arrivo, scadenza)

Ogni chiave sconosciuta e' un errore (cosi' un refuso non viene ignorato in silenzio). Gli errori
sono raccolti tutti insieme in `ConfigError.errors`, con il percorso del campo.

Nessun valore del file viene mai eseguito: sono solo numeri e stringhe controllati qui.
"""

import json
import math
import re
from datetime import datetime, timezone

import numpy as np
from Basilisk.utilities import orbitalMotion

RE_KM = orbitalMotion.REQ_EARTH            # [km] raggio equatoriale usato da BSK-RL
MU = 3.98600436e14                         # [m^3/s^2], come il default dei satelliti BSK-RL
J2 = 1.08262668e-3
SIDEREAL_YEAR_S = 365.2422 * 86400.0       # anno tropico: velocita' di precessione del Sole medio
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
RID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Valori usati quando il file non li specifica. Sono neutri (nessuna ipotesi su un'operazione reale): gli
# esempi in letteratura (invio del piano in ~25 s di contatto, ottimizzazione ~3 h prima del passaggio) sono
# storici e non standard. Si cambiano qui, in un solo punto.
UPLINK_DEFAULTS = dict(uplink="at_ground_contact", uplink_latency_s=0.0, min_uplink_contact_s=0.0)
UPLINKS = ("continuous", "at_ground_contact")
CROSSLINKS = ("none", "free")
ROLES = ("uplink", "downlink")
PLANNING_DEFAULTS = dict(horizon_max_s=21600.0)
# flussi casuali indipendenti derivati dal seed dello scenario (chiave fissa per flusso: aggiungerne non sposta gli altri)
STREAM_KEYS = dict(station_roles=1, arrivals=2, space=3, priority=4)


def rng_stream(seed, name, sub=0):
    """Generator indipendente dal seed e dal nome del flusso (mai np.random globale)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(STREAM_KEYS[name], sub)))


class ConfigError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class _Check:
    """Accumulatore di errori con lettura tipizzata dei campi di un dizionario."""

    def __init__(self):
        self.errors = []

    def err(self, path, msg):
        self.errors.append(f"{path}: {msg}")

    def obj(self, d, path, allowed, required=()):
        if not isinstance(d, dict):
            self.err(path, "deve essere un oggetto JSON")
            return {}
        for k in d:
            if k not in allowed and not k.startswith("_"):      # le chiavi con "_" sono commenti
                self.err(f"{path}.{k}" if path else k, f"chiave sconosciuta (ammesse: {', '.join(sorted(allowed))})")
        for k in required:
            if k not in d:
                self.err(f"{path}.{k}" if path else k, "campo obbligatorio")
        return d

    def num(self, d, key, path, lo=None, hi=None, lo_open=False, default=None, required=False):
        where = f"{path}.{key}" if path else key
        if key not in d or d[key] is None:
            if required:
                self.err(where, "campo obbligatorio")
            return default
        v = d[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            self.err(where, "deve essere un numero")
            return default
        if (lo is not None and (v < lo or (lo_open and v == lo))) or (hi is not None and v > hi):
            rng = f"{'(' if lo_open else '['}{'-inf' if lo is None else lo}, {'+inf' if hi is None else hi}]"
            self.err(where, f"{v} fuori dall'intervallo {rng}")
            return default
        return float(v)


def _epoch(text, c, path):
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        c.err(path, "formato non valido, atteso ISO 8601 (es. 2025-03-21T00:00:00Z)")
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def bsk_epoch(dt):
    """Epoca nel formato di BSK-RL: 'YYYY MMM DD HH:MM:SS.SSS (UTC)'."""
    return f"{dt.year:04d} {MONTHS[dt.month - 1]} {dt.day:02d} {dt:%H:%M:%S}.000 (UTC)"


# ---------------------------------------------------------------- orbite

def sso_inclination_deg(alt_km):
    """Inclinazione che rende l'orbita circolare eliocentrica-sincrona (precessione J2 = 360 deg/anno)."""
    a = (RE_KM + alt_km) * 1e3
    n = math.sqrt(MU / a**3)
    c = -(2 * math.pi / SIDEREAL_YEAR_S) / (1.5 * J2 * (RE_KM * 1e3 / a) ** 2 * n)
    if abs(c) > 1:
        raise ConfigError([f"orbita sun-synchronous impossibile a {alt_km} km di quota"])
    return math.degrees(math.acos(c))


def mean_sun_ra_deg(dt):
    """Ascensione retta del Sole *medio* all'epoca [deg] (formula astronomica a bassa precisione)."""
    d = (dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    return (280.460 + 0.9856474 * d) % 360.0


def sso_raan_deg(dt, ltdn_hours):
    """RAAN [deg] per cui il nodo discendente passa alle `ltdn_hours` ora solare media locale."""
    ltan = (ltdn_hours + 12.0) % 24.0
    return (mean_sun_ra_deg(dt) + (ltan - 12.0) * 15.0) % 360.0


def swath_to_min_elevation_deg(swath_km, alt_km):
    """Elevazione minima del satellite visto da un punto a meta' swath dalla traccia a terra (Terra sferica)."""
    lam = (swath_km / 2) / RE_KM
    r = RE_KM + alt_km
    return math.degrees(math.atan2(r * math.cos(lam) - RE_KM, r * math.sin(lam)))


def make_oe(p):
    """ClassicElements di Basilisk da un dizionario di parametri numerici (angoli in gradi)."""
    oe = orbitalMotion.ClassicElements()
    oe.a, oe.e = (RE_KM + p["altitude_km"]) * 1e3, p["eccentricity"]
    oe.i, oe.Omega = np.radians(p["inclination_deg"]), np.radians(p["raan_deg"])
    oe.omega, oe.f = np.radians(p["arg_perigee_deg"]), np.radians(p["true_anomaly_deg"])
    return oe


# ---------------------------------------------------------------- schemi

INSTRUMENT_KEYS = ("swath_km", "min_elevation_deg", "max_range_km", "data_rate_mbps", "power_draw_w")
RESOURCE_KEYS = ("battery_capacity_wh", "storage_capacity_gbit", "mass_kg", "panel_area_m2", "panel_efficiency",
                 "wheel_torque_max_nm")
DOWNLINK_KEYS = ("rate_mbps", "power_draw_w")
ORBIT_KEYS = ("type", "altitude_km", "eccentricity", "inclination_deg", "raan_deg", "arg_perigee_deg",
              "true_anomaly_deg", "ltdn_hours")


def _read_orbit(o, path, c):
    o = c.obj(o, path, ORBIT_KEYS, required=("type", "altitude_km"))
    t = o.get("type")
    if t not in ("keplerian", "sun_synchronous"):
        c.err(f"{path}.type", "deve essere 'keplerian' oppure 'sun_synchronous'")
        return None
    p = dict(type=t, altitude_km=c.num(o, "altitude_km", path, lo=160, hi=2000, required=True),
             eccentricity=c.num(o, "eccentricity", path, lo=0, hi=0.3, default=0.0),
             arg_perigee_deg=c.num(o, "arg_perigee_deg", path, lo=0, hi=360, default=0.0),
             true_anomaly_deg=c.num(o, "true_anomaly_deg", path, lo=0, hi=360, default=0.0))
    if t == "keplerian":
        p.update(inclination_deg=c.num(o, "inclination_deg", path, lo=0, hi=180, required=True),
                 raan_deg=c.num(o, "raan_deg", path, lo=0, hi=360, required=True))
        if "ltdn_hours" in o:
            c.err(f"{path}.ltdn_hours", "ammesso solo per type 'sun_synchronous'")
    else:
        p.update(ltdn_hours=c.num(o, "ltdn_hours", path, lo=0, hi=24, required=True))
        for k in ("inclination_deg", "raan_deg"):
            if k in o:
                c.err(f"{path}.{k}", "calcolato dal simulatore per type 'sun_synchronous'")
    return None if None in (p["altitude_km"], p["true_anomaly_deg"]) else p


def _read_block(d, key, keys, path, c):
    if key not in d:
        return {}
    b = c.obj(d[key], f"{path}.{key}", keys)
    return {k: b[k] for k in keys if k in b}


def load_constellation(src):
    """src: percorso, stringa JSON o dizionario. Restituisce il dizionario validato (con valori di
    default per satellite risolti), pronto per `build_plan`."""
    data = _load(src)
    c = _Check()
    top = c.obj(data, "", {"schema", "name", "description", "satellites", "instrument", "resources", "downlink",
                           "ground_stations", "comms"}, required=("satellites",))
    if top.get("schema") not in (None, "constellation/v1"):
        c.err("schema", "atteso 'constellation/v1'")
    shared = {b: _read_block(top, b, ks, "", c)
              for b, ks in (("instrument", INSTRUMENT_KEYS), ("resources", RESOURCE_KEYS), ("downlink", DOWNLINK_KEYS))}
    sats = top.get("satellites")
    out_sats = []
    if not isinstance(sats, list) or not sats:
        c.err("satellites", "deve essere una lista non vuota")
        sats = []
    if len(sats) > 64:
        c.err("satellites", "massimo 64 satelliti")
        sats = sats[:64]
    names = set()
    for i, s in enumerate(sats):
        path = f"satellites[{i}]"
        s = c.obj(s, path, {"name", "orbit", "instrument", "resources", "downlink"}, required=("name", "orbit"))
        name = s.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            c.err(f"{path}.name", "deve essere un identificatore (lettere, cifre, _; non inizia con una cifra)")
        elif name in names:
            c.err(f"{path}.name", f"'{name}' duplicato")
        names.add(name)
        merged = {}
        for b, ks in (("instrument", INSTRUMENT_KEYS), ("resources", RESOURCE_KEYS), ("downlink", DOWNLINK_KEYS)):
            merged[b] = {**shared[b], **_read_block(s, b, ks, path, c)}
        orbit = _read_orbit(s.get("orbit"), f"{path}.orbit", c) if "orbit" in s else None
        out_sats.append(dict(name=name, orbit=orbit, **merged))
    # controlli di valore sui blocchi (una sola volta per satellite, sul risultato dell'unione)
    for i, s in enumerate(out_sats):
        path = f"satellites[{i}]"
        ins, res, dl = s["instrument"], s["resources"], s["downlink"]
        if "swath_km" in ins and "min_elevation_deg" in ins:
            c.err(f"{path}.instrument", "indicare 'swath_km' oppure 'min_elevation_deg', non entrambi")
        for blk, spec in (("instrument", dict(swath_km=(0, 3000, True), min_elevation_deg=(0, 90, False),
                                              max_range_km=(0, 20000, True), data_rate_mbps=(0, 1e5, True),
                                              power_draw_w=(0, 1e4, False))),
                          ("resources", dict(battery_capacity_wh=(0, 1e5, True), storage_capacity_gbit=(0, 1e5, True),
                                             mass_kg=(1, 1e4, False), panel_area_m2=(0, 100, True),
                                             panel_efficiency=(0, 1, True), wheel_torque_max_nm=(0, 10, True))),
                          ("downlink", dict(rate_mbps=(0, 1e5, True), power_draw_w=(0, 1e4, False)))):
            for k, (lo, hi, op) in spec.items():
                if k in s[blk]:
                    s[blk][k] = c.num(s[blk], k, f"{path}.{blk}", lo=lo, hi=hi, lo_open=op)
    cm = c.obj(top.get("comms", {}), "comms", {"crosslinks", "uplink", "uplink_latency_s", "min_uplink_contact_s"})
    comms = {}
    for key, allowed in (("uplink", UPLINKS), ("crosslinks", CROSSLINKS)):
        if key in cm:
            if cm[key] == "line_of_sight" and key == "crosslinks":
                c.err("comms.crosslinks", "'line_of_sight' non ancora implementato (ammessi: none, free)")
            elif cm[key] not in allowed:
                c.err(f"comms.{key}", f"deve essere uno di: {', '.join(allowed)}")
            else:
                comms[key] = cm[key]
    for key, hi in (("uplink_latency_s", 86400), ("min_uplink_contact_s", 3600)):
        v = c.num(cm, key, "comms", lo=0, hi=hi)
        if v is not None:
            comms[key] = v
    stations, snames = [], set()
    gs = top.get("ground_stations", [])
    if not isinstance(gs, list):
        c.err("ground_stations", "deve essere una lista")
        gs = []
    for i, g in enumerate(gs):
        path = f"ground_stations[{i}]"
        g = c.obj(g, path, {"name", "lat", "lon", "elevation_m", "min_elevation_deg", "roles"}, required=("name", "lat", "lon"))
        roles = g.get("roles")
        if roles is not None and (not isinstance(roles, list) or not roles or any(r not in ROLES for r in roles)
                                  or len(set(roles)) != len(roles)):
            c.err(f"{path}.roles", "deve essere una lista non vuota, senza ripetizioni, di: uplink, downlink")
            roles = None
        name = g.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            c.err(f"{path}.name", "deve essere un identificatore (lettere, cifre, _)")
        elif name in snames:
            c.err(f"{path}.name", f"'{name}' duplicato")
        snames.add(name)
        stations.append(dict(name=name, lat=c.num(g, "lat", path, lo=-90, hi=90, required=True),
                             lon=c.num(g, "lon", path, lo=-180, hi=180, required=True),
                             elevation_m=c.num(g, "elevation_m", path, lo=-500, hi=9000, default=0.0),
                             min_elevation_deg=c.num(g, "min_elevation_deg", path, lo=0, hi=90, default=10.0),
                             roles=list(roles) if roles else None))
    if not stations and "ground_stations" not in c.errors:
        c.err("ground_stations", "serve almeno una stazione a terra (senza downlink il buffer si riempie)")
    if c.errors:
        raise ConfigError(c.errors)
    return dict(name=str(top.get("name", "costellazione")), satellites=out_sats, ground_stations=stations, comms=comms)


def load_scenario(src=None):
    """Scenario validato. Senza argomento: valori di default (episodio di 24 h, 600 target casuali)."""
    data = _load(src) if src is not None else {}
    c = _Check()
    top = c.obj(data, "", {"schema", "name", "description", "seed", "epoch", "duration_hours", "round_s", "track_dt",
                           "targets", "planning", "non_stationarity"})
    if top.get("schema") not in (None, "scenario/v1"):
        c.err("schema", "atteso 'scenario/v1'")
    seed = top.get("seed", 4)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        c.err("seed", "deve essere un intero tra 0 e 2^31-1")
    epoch = _epoch(top.get("epoch", "2025-03-21T00:00:00Z"), c, "epoch")
    hours = c.num(top, "duration_hours", "", lo=0, hi=24 * 365, default=24.0)
    rnd = c.num(top, "round_s", "", lo=1, hi=3600, default=60.0)
    dt = c.num(top, "track_dt", "", lo=1, hi=600, default=10.0)
    if rnd and dt and abs(rnd / dt - round(rnd / dt)) > 1e-9:
        c.err("round_s", "deve essere multiplo di track_dt")
    targets = _read_targets(top.get("targets", {}), c)
    pl = c.obj(top.get("planning", {}), "planning", {"horizon_max_s"})
    planning = dict(horizon_max_s=c.num(pl, "horizon_max_s", "planning", lo=60, hi=7 * 86400,
                                        default=PLANNING_DEFAULTS["horizon_max_s"]))
    if top.get("non_stationarity") not in (None, {}):
        c.err("non_stationarity", "non ancora implementata: lasciare assente o null")
    if c.errors:
        raise ConfigError(c.errors)
    return dict(name=str(top.get("name", "scenario")), seed=seed, epoch=epoch, duration_hours=hours, round_s=rnd,
                track_dt=dt, targets=targets, planning=planning,
                n_targets=targets.get("count", 0))        # n_targets: compatibilita' con la modalita' random


def _read_targets(tg, c):
    """Blocco `targets` dello scenario: mode random (lista fissa), stream (Poisson) o file."""
    keys = {"mode", "count", "initial_count", "arrival", "spatial", "priority", "validity_hours", "path"}
    tg = c.obj(tg, "targets", keys)
    mode = tg.get("mode", "random")
    per_mode = dict(random={"mode", "count"},
                    stream={"mode", "initial_count", "arrival", "spatial", "priority", "validity_hours"},
                    file={"mode", "path"})
    if mode not in per_mode:
        c.err("targets.mode", "deve essere uno di: random, stream, file")
        return dict(mode="random", count=600)
    for k in tg:
        if k in keys and k not in per_mode[mode]:
            c.err(f"targets.{k}", f"non ammesso con mode '{mode}'")
    if mode == "random":
        return dict(mode=mode, count=int(c.num(tg, "count", "targets", lo=1, hi=20000, default=600)))
    if mode == "file":
        path = tg.get("path")
        if path is not None and not isinstance(path, str):
            c.err("targets.path", "deve essere una stringa")
            path = None
        return dict(mode=mode, path=path)
    # stream
    ar = c.obj(tg.get("arrival"), "targets.arrival", {"type", "rate_per_hour"}, required=("rate_per_hour",)) \
        if tg.get("arrival") is not None else (c.err("targets.arrival", "campo obbligatorio per mode 'stream'") or {})
    if ar.get("type", "poisson") != "poisson":
        c.err("targets.arrival.type", "per ora e' ammesso solo 'poisson' (tasso costante per ora)")
    rate = c.num(ar, "rate_per_hour", "targets.arrival", lo=0, hi=3600, lo_open=True)
    sp = c.obj(tg.get("spatial", {}), "targets.spatial", {"type"})
    if sp.get("type", "uniform") != "uniform":
        c.err("targets.spatial.type", "per ora e' ammesso solo 'uniform' (uniforme sulla sfera)")
    pr = c.obj(tg.get("priority", {}), "targets.priority", {"type", "min", "max"})
    if pr.get("type", "uniform") != "uniform":
        c.err("targets.priority.type", "per ora e' ammesso solo 'uniform'")
    pmin = c.num(pr, "min", "targets.priority", lo=0, default=0.0)
    pmax = c.num(pr, "max", "targets.priority", lo=0, lo_open=True, default=1.0)
    if None not in (pmin, pmax) and pmin >= pmax:
        c.err("targets.priority", "min deve essere minore di max")
    val = c.num(tg, "validity_hours", "targets", lo=0, hi=24 * 365, lo_open=True)
    return dict(mode=mode, initial_count=int(c.num(tg, "initial_count", "targets", lo=0, hi=20000, default=0)),
                rate_per_hour=rate, priority=dict(min=pmin, max=pmax),
                validity_s=None if val is None else val * 3600.0)


def load_requests(src):
    """requests/v1: elenco di richieste di imaging. Restituisce {name, requests: [...]} con ogni richiesta normalizzata
    (id, lat, lon, name, elevation_m, priority, arrival_s, expires_s|None), ordinata per istante di arrivo."""
    data = _load(src)
    c = _Check()
    top = c.obj(data, "", {"schema", "name", "description", "requests"}, required=("requests",))
    if top.get("schema") not in (None, "requests/v1"):
        c.err("schema", "atteso 'requests/v1'")
    items = top.get("requests")
    if not isinstance(items, list):
        c.err("requests", "deve essere una lista")
        items = []
    if len(items) > 50000:
        c.err("requests", "massimo 50000 richieste")
        items = items[:50000]
    out, ids = [], set()
    for i, r in enumerate(items):
        path = f"requests[{i}]"
        r = c.obj(r, path, {"id", "target", "priority", "arrival_s", "expires_s", "validity_s"},
                  required=("target", "priority"))
        rid = r.get("id", f"R{i + 1:05d}")
        if not isinstance(rid, str) or not RID.match(rid):
            c.err(f"{path}.id", "deve essere una stringa di 1-64 caratteri tra lettere, cifre, _ . -")
        elif rid in ids:
            c.err(f"{path}.id", f"'{rid}' duplicato")
        ids.add(rid)
        t = c.obj(r.get("target"), f"{path}.target", {"lat", "lon", "name", "elevation_m"}, required=("lat", "lon"))
        lat = c.num(t, "lat", f"{path}.target", lo=-90, hi=90, required=True)
        lon = c.num(t, "lon", f"{path}.target", lo=-180, hi=180, required=True)
        pr = c.num(r, "priority", path, lo=0, lo_open=True, required=True)
        arr = c.num(r, "arrival_s", path, lo=0, default=0.0)
        exp, val = c.num(r, "expires_s", path, lo=0), c.num(r, "validity_s", path, lo=0, lo_open=True)
        if exp is not None and val is not None:
            c.err(path, "indicare 'expires_s' oppure 'validity_s', non entrambi")
        if val is not None and arr is not None:
            exp = arr + val
        if exp is not None and arr is not None and exp <= arr:
            c.err(f"{path}.expires_s", "deve essere successivo a arrival_s")
        out.append(dict(id=rid, lat=lat, lon=lon, name=str(t.get("name", rid)), elevation_m=c.num(t, "elevation_m", f"{path}.target", lo=-500, hi=9000, default=0.0),
                        priority=pr, arrival_s=arr, expires_s=exp))
    if c.errors:
        raise ConfigError(c.errors)
    out.sort(key=lambda x: x["arrival_s"])
    return dict(name=str(top.get("name", "richieste")), requests=out)


def fix_station_roles(stations, seed, notices):
    """Garantisce almeno una stazione con ruolo downlink e una con ruolo uplink (decisione dell'utente).

    Senza ruolo indicato una stazione fa entrambi. Se nessuna stazione ha un ruolo, ne sceglie una a caso (flusso
    casuale derivato dal seed: stessa scelta a parita' di seed) e le assegna il ruolo mancante. Se le altre stazioni
    coprono ancora l'altro ruolo la stazione *cambia* ruolo, altrimenti (una sola stazione) li ha entrambi.
    Ogni decisione e' registrata in `notices`."""
    st = [dict(s, roles=list(s["roles"]) if s["roles"] else list(ROLES)) for s in stations]
    for sub, (need, other) in enumerate((("downlink", "uplink"), ("uplink", "downlink"))):
        if any(need in s["roles"] for s in st):
            continue
        i = int(rng_stream(seed, "station_roles", sub).integers(len(st)))
        others_cover = any(other in s["roles"] for j, s in enumerate(st) if j != i)
        before = list(st[i]["roles"])
        st[i]["roles"] = [need] if others_cover else sorted(set(before) | {need}, key=ROLES.index)
        notices.append(dict(
            level="warning", code="station_role_assigned",
            message=(f"Nessuna stazione a terra con ruolo '{need}': la stazione '{st[i]['name']}' e' stata scelta a caso "
                     f"(seed {seed}) e ora ha ruolo {' + '.join(st[i]['roles'])} (prima: {' + '.join(before)})."
                     + ("" if others_cover else " E' l'unica stazione utilizzabile, quindi svolge entrambi i ruoli."))))
    return st


def resolve_comms(comms, n_sats, notices):
    """Applica i default ai parametri di comunicazione mancanti e registra un avviso informativo."""
    out = dict(comms or {})
    used = []
    for k, v in UPLINK_DEFAULTS.items():
        if k not in out:
            out[k] = v
            used.append(f"comms.{k} = {v}")
    if "crosslinks" not in out:
        out["crosslinks"] = "free" if n_sats > 1 else "none"
        used.append(f"comms.crosslinks = {out['crosslinks']}")
    if used:
        notices.append(dict(level="info", code="comms_defaults",
                            message="Parametri non specificati nel file, usati i default: " + "; ".join(used) + "."))
    return out


def _load(src):
    if isinstance(src, dict):
        return src
    text = str(src)
    try:
        if text.lstrip().startswith("{"):
            return json.loads(text)
        with open(text, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise ConfigError([f"file non leggibile: {e.strerror or e}"])
    except json.JSONDecodeError as e:
        raise ConfigError([f"JSON non valido: {e}"])


# ---------------------------------------------------------------- traduzione in argomenti BSK-RL

def build_plan(constellation, scenario):
    """Unisce costellazione e scenario negli argomenti da passare a BSK-RL.

    Restituisce un dizionario: `sats` (nome, parametri orbitali risolti, `sat_args` BSK-RL, descrizione),
    `world_args`, `stations` e i parametri di episodio."""
    sats = []
    for s in constellation["satellites"]:
        orb = dict(s["orbit"])
        if orb["type"] == "sun_synchronous":
            orb["inclination_deg"] = sso_inclination_deg(orb["altitude_km"])
            orb["raan_deg"] = sso_raan_deg(scenario["epoch"], orb["ltdn_hours"])
        ins, res, dl = s["instrument"], s["resources"], s["downlink"]
        a = {}
        if "swath_km" in ins:
            a["imageTargetMinimumElevation"] = np.radians(swath_to_min_elevation_deg(ins["swath_km"], orb["altitude_km"]))
        if "min_elevation_deg" in ins:
            a["imageTargetMinimumElevation"] = np.radians(ins["min_elevation_deg"])
        for k, key, f in (("max_range_km", "imageTargetMaximumRange", 1e3), ("data_rate_mbps", "instrumentBaudRate", 1e6),
                          ("power_draw_w", "instrumentPowerDraw", -1.0)):
            if k in ins:
                a[key] = ins[k] * f
        for k, key, f in (("battery_capacity_wh", "batteryStorageCapacity", 3600.0),
                          ("storage_capacity_gbit", "dataStorageCapacity", 1e9), ("mass_kg", "mass", 1.0),
                          ("panel_area_m2", "panelArea", 1.0), ("panel_efficiency", "panelEfficiency", 1.0),
                          ("wheel_torque_max_nm", "u_max", 1.0)):
            if k in res:
                a[key] = res[k] * f
        for k, key, f in (("rate_mbps", "transmitterBaudRate", -1e6), ("power_draw_w", "transmitterPowerDraw", -1.0)):
            if k in dl:
                a[key] = dl[k] * f
        sats.append(dict(name=s["name"], orbit=orb, sat_args=a))
    notices = []
    comms = resolve_comms(constellation["comms"], len(sats), notices)
    if any(g["roles"] is None for g in constellation["ground_stations"]):
        notices.append(dict(level="info", code="roles_default", message=(
            "Ruoli non specificati per alcune stazioni: svolgono sia uplink sia downlink.")))
    stations = fix_station_roles(constellation["ground_stations"], scenario["seed"], notices)
    world = dict(utc_init=bsk_epoch(scenario["epoch"]))
    if stations:
        world["groundStationsData"] = [dict(name=g["name"], lat=g["lat"], long=g["lon"], elev=g["elevation_m"],
                                            gsMinimumElevation=np.radians(g["min_elevation_deg"])) for g in stations]
    return dict(name=constellation["name"], scenario=scenario, sats=sats, world_args=world, stations=stations,
                comms=comms, notices=notices)
