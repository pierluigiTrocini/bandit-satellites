# File di configurazione

Due file JSON separati, validati da `scripts/config_io.py` (chiavi sconosciute = errore; le chiavi che
iniziano con `_` sono commenti). Si importano dal viewer (modalità Live, «⇪ Importa», anche entrambi insieme)
oppure si modificano nell'editor del viewer («✎ Modifica»: a ogni modifica valida la simulazione riparte; se i file non sono validi resta in corso quella precedente), oppure da riga di comando (`--constellation`, `--scenario`). Un file omesso resta quello in uso.

## `constellation/v1` — cosa si simula

| Campo | Note |
|---|---|
| `name` | etichetta |
| `satellites[]` | `name` (identificatore), `orbit`, e blocchi opzionali `instrument`, `resources`, `downlink` che sovrascrivono quelli comuni |
| `instrument`, `resources`, `downlink` (livello alto) | valori comuni a tutti i satelliti |
| `ground_stations[]` | `name`, `lat`, `lon` [deg], opz. `elevation_m`, `min_elevation_deg` (default 10, per stazione); almeno una |

`orbit`:
- `type: "keplerian"`: `altitude_km`, `inclination_deg`, `raan_deg`, opz. `eccentricity`, `arg_perigee_deg`, `true_anomaly_deg`
  (semiasse = raggio equatoriale + `altitude_km`).
- `type: "sun_synchronous"`: `altitude_km`, `ltdn_hours` (ora solare media locale del nodo discendente), opz. `true_anomaly_deg`.
  Inclinazione (da precessione J2) e RAAN (dall'epoca dello scenario) sono calcolate.

`instrument`: `swath_km` **oppure** `min_elevation_deg` (la swath è convertita in elevazione minima a metà swath,
Terra sferica), `max_range_km`, `data_rate_mbps`, `power_draw_w`.
`resources`: `battery_capacity_wh`, `storage_capacity_gbit`, `mass_kg`, `panel_area_m2`, `panel_efficiency`, `wheel_torque_max_nm`.
`downlink`: `rate_mbps`, `power_draw_w`. Ogni campo omesso resta al default di BSK-RL.

## `scenario/v1` — come si simula

`seed` (default 4), `epoch` ISO 8601 UTC (default 2025-03-21T00:00:00Z), `duration_hours` (24; 0 = senza limite),
`round_s` (60, multiplo di `track_dt`), `track_dt` (10), `targets: {mode: "random", count: 600}` (uniformi sulla sfera).
`non_stationarity`: riservato, per ora assente o `null`.

## Limiti da dichiarare

- Lo strumento è un satellite **agile** che può riprendere qualunque target nella swath: Sentinel-2 reale è nadir-pointing.
- Orbita propagata da BSK-RL (gravità + drag), non da TLE/SGP4: valida per ore o pochi giorni.
- Un'immagine vale 1 s di strumento (`data_rate_mbps` × 1 s): la scala di `storage_capacity_gbit` è relativa a questa convenzione.
- I valori di `sentinel2.constellation.json` sono approssimati e vanno verificati sulle fonti ESA.
