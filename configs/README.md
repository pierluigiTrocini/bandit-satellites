# File di configurazione

Tre tipi di file JSON (costellazione, scenario, richieste), validati da `scripts/config_io.py` (chiavi sconosciute = errore; le chiavi
che iniziano con `_` sono commenti). Si importano dal viewer (modalità Live, «⇪ Importa», anche insieme), oppure si modificano
nell'editor del viewer («✎ Modifica»: a ogni modifica valida la simulazione riparte; se i file non sono validi resta in corso
quella precedente), oppure si passano da riga di comando (`--constellation`, `--scenario`, `--requests`). Un file omesso resta
quello in uso. Esempi: `sentinel2.constellation.json`, `default.scenario.json`, `stream.scenario.json`,
`requests.scenario.json` + `requests.example.json`.

## `constellation/v1` — cosa si simula

| Campo | Note |
|---|---|
| `name` | etichetta |
| `satellites[]` | `name` (identificatore), `orbit`, e blocchi opzionali `instrument`, `resources`, `downlink` che sovrascrivono quelli comuni |
| `instrument`, `resources`, `downlink` (livello alto) | valori comuni a tutti i satelliti |
| `ground_stations[]` | `name`, `lat`, `lon` [deg], opz. `elevation_m`, `min_elevation_deg` (default 10, per stazione), `roles`; almeno una |
| `comms` | collegamenti e uplink (sotto) |

`orbit`:
- `type: "keplerian"`: `altitude_km`, `inclination_deg`, `raan_deg`, opz. `eccentricity`, `arg_perigee_deg`, `true_anomaly_deg`
  (semiasse = raggio equatoriale + `altitude_km`).
- `type: "sun_synchronous"`: `altitude_km`, `ltdn_hours` (ora solare media locale del nodo discendente), opz. `true_anomaly_deg`.
  Inclinazione (da precessione J2) e RAAN (dall'epoca dello scenario) sono calcolate.

`instrument`: `swath_km` **oppure** `min_elevation_deg` (la swath è convertita in elevazione minima a metà swath,
Terra sferica), `max_range_km`, `data_rate_mbps`, `power_draw_w`.
`resources`: `battery_capacity_wh`, `storage_capacity_gbit`, `mass_kg`, `panel_area_m2`, `panel_efficiency`, `wheel_torque_max_nm`.
`downlink`: `rate_mbps`, `power_draw_w`. Ogni campo omesso resta al default di BSK-RL.

### Ruoli delle stazioni (`ground_stations[].roles`)
Lista non vuota di `"uplink"` (invia piano e richieste al satellite) e/o `"downlink"` (riceve i dati). Senza `roles` la stazione fa
entrambi. **Deve esserci sempre almeno una stazione di uplink e una di downlink**: se manca un ruolo, il simulatore sceglie una
stazione a caso (flusso casuale derivato dal seed: stessa scelta a parità di seed) e le assegna il ruolo mancante. Se le altre
stazioni coprono ancora l'altro ruolo la stazione *cambia* ruolo; se è l'unica stazione le fa entrambi. La decisione compare nel
riquadro **«Avvisi del simulatore»** del viewer (livello «warning») e in `trace.meta.notices`.

### Comunicazione (`comms`)
| Campo | Valori | Default | Effetto |
|---|---|---|---|
| `uplink` | `at_ground_contact` \| `continuous` | `at_ground_contact` | `at_ground_contact`: **modello B**, decisioni a terra a lotti; il piano sale al satellite quando passa su una stazione di uplink e lo esegue senza scelte fino al contatto successivo. `continuous`: decisioni a ogni round con informazione immediata. |
| `uplink_latency_s` | 0-86400 | 0 | Anticipo con cui una richiesta deve essere arrivata a terra per entrare nel piano di un contatto. |
| `min_uplink_contact_s` | 0-3600 | 0 | Un contatto più breve non basta a inviare il piano. |
| `crosslinks` | `none` \| `free` | `free` (più satelliti), `none` (uno) | Collegamenti tra satelliti (`free` = BSK-RL `FreeCommunication`). `line_of_sight` non è ancora implementato. |

I default dei parametri non specificati sono in un solo punto (`config_io.UPLINK_DEFAULTS`) e sono **neutri**: non rappresentano
un'operazione reale (le cifre citate in letteratura, p. es. ~25 s di contatto o ottimizzazione ~3 h prima del passaggio, sono
esempi storici). Il simulatore elenca nel riquadro avvisi i default che ha usato.

## `scenario/v1` — come si simula

`seed` (default 4), `epoch` ISO 8601 UTC (default 2025-03-21T00:00:00Z), `duration_hours` (24; 0 = senza limite),
`round_s` (60, multiplo di `track_dt`), `track_dt` (10), `planning.horizon_max_s` (21600: un piano non copre più di così),
`non_stationarity`: riservato, per ora assente o `null`.

### Richieste di imaging (`targets`)
| `mode` | Campi | Significato |
|---|---|---|
| `random` (default) | `count` (600) | Lista fissa di target uniformi sulla sfera, noti fin dall'inizio (priorità casuale in [0, 1)). |
| `stream` | `initial_count` (0), `arrival: {type: "poisson", rate_per_hour}` (obbligatorio), `spatial: {type: "uniform"}`, `priority: {type: "uniform", min, max}` (default (0, 1]), `validity_hours` (assente = non scade) | Flusso di richieste: arrivi di Poisson a tasso costante. |
| `file` | `path` (opz., per la riga di comando) | Richieste da un file `requests/v1`. |

Gli arrivi e le loro proprietà usano **flussi casuali propri** derivati dal seed e indipendenti dalle azioni: due algoritmi con lo
stesso seed vedono la stessa sequenza di richieste. Un flusso generato si può salvare (`export_trace.py --export-requests F`, o
`GET /live/requests.json` dal viewer) e rileggere con `mode: "file"`.

## `requests/v1` — elenco di richieste

```json
{ "schema": "requests/v1", "requests": [
  { "id": "R1", "target": { "lat": 45.07, "lon": 7.69, "name": "Torino" }, "priority": 0.9, "arrival_s": 0, "validity_s": 43200 } ] }
```
`id` (stabile, 1-64 caratteri tra lettere, cifre, `_ . -`; se manca lo assegna il simulatore), `target.lat/lon` (opz. `name`,
`elevation_m`), `priority` (> 0, senza unità), `arrival_s` (istante di arrivo a terra, default 0), scadenza opzionale con
`expires_s` (istante assoluto) **oppure** `validity_s` (durata dopo l'arrivo). Senza scadenza la richiesta non scade (con episodi
infiniti l'arretrato cresce). Esempio: `requests.example.json`.

## Come funziona il modello B (`uplink: at_ground_contact`)
1. Le richieste arrivano a **terra**; il satellite le conosce solo dopo averle ricevute a un contatto di uplink.
2. A ogni contatto il pianificatore (oggi `DemoPlanner`, **non** un bandit) sceglie le immagini e i downlink fino al contatto
   di uplink successivo (o fino a `planning.horizon_max_s`); il piano sale e il satellite lo esegue senza scelte.
3. Se un satellite non ha un contatto in corso all'avvio parte con un piano come se fosse stato caricato a t = 0.
4. La terra sa che un'immagine è stata fatta solo quando i dati sono stati **scaricati** (registro FIFO sulla memoria).
   Fino ad allora una richiesta ripresa risulta ancora aperta a terra: con `crosslinks: none` è possibile riprenderla due volte.
5. La protezione di bordo (batteria sotto il 20%, fino al 50%) prevale sul piano: ricarica.

## Limiti da dichiarare

- Lo strumento è un satellite **agile** che può riprendere qualunque target nella swath: Sentinel-2 reale è nadir-pointing.
- Orbita propagata da BSK-RL (gravità + drag), non da TLE/SGP4: valida per ore o pochi giorni.
- Un'immagine vale 1 s di strumento (`data_rate_mbps` × 1 s): la scala di `storage_capacity_gbit` è relativa a questa convenzione.
- I valori di `sentinel2.constellation.json` sono approssimati e vanno verificati sulle fonti ESA.
- **Finestre di accesso.** Negli episodi da file le finestre le calcola `scripts/ground.py::WindowEngine` (campionamento ogni 5 s con
  raffinamento degli estremi), non BSK-RL: il calcolo di BSK-RL perde i passaggi quando la finestra candidata è molto più lunga del
  passaggio, cioè con elevazione minima bassa e quota alta (misurato su Sentinel-2, stazioni a 5°: 2 passaggi su 24 per un
  satellite e 0 su 24 per l'altro). Finestre più brevi di 5 s possono mancare. Il percorso senza file (Walker a 500 km, stazioni
  di default a 10°) usa ancora BSK-RL, dove il difetto non compare (verificato su 20 e 22 passaggi).
- Il pianificatore dimostrativo ipotizza che ogni immagine occupi 90 s dall'apertura della finestra; con questa politica circa l'80%
  dei tentativi di imaging fallisce (il satellite non fa in tempo a puntare): i bracci «geometricamente fattibili» non sono tutti eseguibili.
- Lo stato del satellite visto a terra è quello reale al momento della pianificazione (telemetria ideale).
