# Decisioni di progetto del simulatore

Registro delle decisioni per rendere il simulatore realistico e adattabile a qualsiasi costellazione.
Va letto prima di implementare le feature qui elencate e aggiornato a ogni decisione.

- **Ultimo aggiornamento**: 2026-10-14
- **Codice al momento della stesura**: commit `1fcacd6` (config da JSON + editor nel viewer). Nulla di quanto è descritto
  nelle sezioni 3-6 è ancora implementato, salvo dove indicato.
- **Regole del repository da rispettare** (da `CLAUDE.md`): mai committare/pushare senza conferma; `np.random.Generator`
  derivati dal seed, mai `np.random` globale; versioni `bsk==2.12.0`, `bsk-rl==1.3.7` fisse; non inventare dati;
  dichiarare i limiti di astrazione; controllare prima se BSK-RL ha già la funzione.

## Legenda degli stati

| Stato | Significato |
|---|---|
| **FISSATO** | Deciso dall'utente. Si implementa così. |
| **PROPOSTO** | Proposta dell'assistente, non ancora approvata. Non implementare senza conferma. |
| **APERTO** | Decisione da prendere (domanda in sezione 9). |
| **FUORI SCOPE** | Escluso per ora. Non implementare, ma lasciare un punto di innesto dove indicato. |
| **IN ATTESA** | L'utente deve ancora rispondere. |

---

## 1. Principio generale (FISSATO)

1. Il simulatore **non conosce nessuna costellazione**: conosce solo il formato dei due file di configurazione
   (`constellation/v1` e `scenario/v1`, vedi `configs/README.md`). Sentinel-2 è solo un file di esempio
   (`configs/sentinel2.constellation.json`, valori approssimati e da verificare).
2. Ogni forma di realismo è un **modello attivabile e parametrizzabile da file**, mai un valore cablato nel codice per una
   costellazione specifica.
3. La scelta della costellazione definitiva è rinviata ai prossimi passi: nessuna feature può dipendere da Sentinel-2.
4. **Separazione dei file**: costellazione = *cosa* si simula (satelliti, strumenti, stazioni); scenario = *come*
   (seed, epoca, durata, flusso di richieste). Qualunque costellazione deve poter girare con qualunque scenario.
5. **Compatibilità**: i file `v1` esistenti devono restare validi. I campi nuovi sono opzionali e il loro default riproduce il
   comportamento attuale.
6. Resta valida la validazione attuale di `scripts/config_io.py`: chiavi sconosciute = errore (salvo le chiavi che iniziano
   con `_`, usate come commenti); errori raccolti tutti insieme con il percorso del campo; nessun valore del file viene
   eseguito.

## 2. Fuori scope in questa fase (FUORI SCOPE)

L'obiettivo ora è avere un **simulatore realistico come base per la ricerca**. Sono esclusi:

- reward, qualità dell'immagine e controllo qualità a terra (compreso il reward ritardato dopo il downlink e il registro
  FIFO delle immagini in memoria);
- modello di nuvolosità (vera/prevista), perché nel progetto serve solo a calcolare la qualità;
- il campo `instrument.constraints.cloud_sensitive` (rimosso dallo schema proposto);
- non stazionarietà controllabile: il campo `non_stationarity` dello scenario resta **riservato**; oggi è rifiutato se non è
  assente o `null` (`load_scenario`). Va mantenuto così.

Resta **dentro**: il flusso fisico dei dati (la memoria si riempie, il downlink la svuota, con i suoi ritardi) e il consumo di
risorse (batteria, memoria). Quello che si rimanda è solo l'attribuzione di un valore alle singole immagini.
Il simulatore continua a produrre i bracci fattibili e i costi in risorse; il reward verrà costruito sopra in una fase
successiva.

## 3. Schema della costellazione esteso (FISSATO: elenco dei campi approvato il 2026-10-14)

Tutti i campi sono **opzionali**. Possono stare al livello alto del file (valori comuni) o dentro ogni satellite
(override), come già avviene per `instrument`, `resources`, `downlink`.

### 3.1 Campi approvati

| Campo | Tipo / valori | Default (= comportamento attuale) | Effetto |
|---|---|---|---|
| `instrument.kind` | `optical` \| `sar` \| `generic` | `generic` | Seleziona quali vincoli si applicano (3.3). |
| `instrument.pointing` | `agile` \| `nadir` | `agile` | Modo di ripresa (sezione 6). Per ora **solo `agile` è implementato**. |
| `instrument.constraints.min_sun_elevation_deg` | numero [0, 90] | assente = nessun vincolo | Un'acquisizione è ammessa solo se il Sole è sopra questa altezza sull'orizzonte del target. Ha senso per `optical`. |
| `instrument.constraints.incidence_min_deg` | numero [0, 90] | assente | Angolo di incidenza minimo (banda tipica del SAR). |
| `instrument.constraints.incidence_max_deg` | numero [0, 90], `>= incidence_min_deg` | assente | Angolo di incidenza massimo. |
| `comms.crosslinks` | `none` \| `line_of_sight` \| `free` | `free` se più satelliti, `none` se uno | Collegamenti tra satelliti (3.4). |
| `comms.uplink` | `continuous` \| `at_ground_contact` | `continuous` | Come i satelliti ricevono informazioni e comandi dal segmento di terra (sezione 5). |
| `data.bits_per_acquisition` | numero > 0 [bit] | `instrument.data_rate_mbps` × 1e6 × 1 s | Volume di dati di un'immagine. Sostituisce la convenzione «1 s di strumento = 1 immagine». |
| `reference` | oggetto (3.5) | assente | Valori pubblici per la validazione. Non influenza la simulazione. |

### 3.2 Estensione proposta ma NON approvata (PROPOSTO)

- `instrument.pointing`: aggiungere il valore `roll` (può inclinarsi solo lateralmente). In attesa del parere dell'utente dopo
  la lettura delle fonti (sezione 6).
- `ground_stations[].roles` (**approvato e implementato il 2026-10-14**, con correzione automatica casuale e avviso nel viewer):
  lista di `uplink` e/o `downlink` (default entrambi). Motivazione: nella realtà i comandi e i dati di payload possono passare
  da stazioni diverse (sezione 7).
- `comms.uplink_latency_s` [s] (default 0, **approvato e implementato**): anticipo con cui una richiesta deve essere arrivata a
  terra per entrare nel piano di un contatto.
- `comms.min_uplink_contact_s` [s] (default 0, **approvato e implementato**): durata minima di un contatto perché l'invio vada a buon fine.

### 3.3 Vincoli per tipo di strumento (implementazione)

- **Illuminazione** (`optical`): per ogni finestra di accesso a un target, il Sole deve stare sopra
  `min_sun_elevation_deg` sul target. L'elevazione solare si calcola dall'epoca e dalla posizione del Sole (formula
  astronomica a bassa precisione, già presente in `config_io.mean_sun_ra_deg` per il Sole medio, o effemeridi Basilisk).
  - **Aggancio verificato in BSK-RL**: `AccessSatellite` ha `add_access_filter(fn)` e `find_next_opportunities(...,
    filter=...)`; `fn` riceve il dizionario dell'opportunità (`type`, `object`, `window=(t_open, t_close)`, `r_LP_P`...) e
    restituisce un booleano. Il filtro agisce **sull'intera finestra**: una finestra che attraversa il terminatore va gestita
    (decisione da prendere in implementazione: usare l'istante centrale della finestra, oppure tagliare la finestra).
  - Non è un vincolo di batteria: l'eclisse del *satellite* è già modellata da BSK-RL; qui si parla della luce *sul target*.
- **Banda di incidenza** (`sar`): BSK-RL conosce solo elevazione minima e distanza massima per l'accesso, non un minimo di
  incidenza. Servono un filtro nostro (stesso meccanismo di `add_access_filter`) e la conversione incidenza ↔ elevazione
  (Terra sferica, stessa geometria di `swath_to_min_elevation_deg`).
- Lo schema deve segnalare errore se un vincolo non ha senso per il `kind` (es. `min_sun_elevation_deg` con `sar`), oppure
  ignorarlo con un avviso: **da decidere in implementazione**, preferibile l'errore.

### 3.4 Comunicazione tra satelliti

- `free`: comportamento attuale (`FreeCommunication`): i satelliti condividono subito cosa è stato ripreso.
- `line_of_sight`: `LOSCommunication` di BSK-RL; richiede che il satellite usi un modello dinamico
  `LOSCommDynModel` (verificato in `bsk_rl/comm/communication.py`). Da provare con un test prima di promettere.
- `none`: nessun collegamento. L'informazione sulle immagini già fatte arriva ai satelliti solo tramite il segmento di terra
  (sezione 5).

### 3.5 Blocco `reference` (PROPOSTO nel formato)

Oggetto `{ "<nome_metrica>": { "value": numero, "unit": "stringa", "source": "stringa" } }`. Esempi di metriche: tempo di
rivisita medio, copertura giornaliera. L'assistente **non deve inventare i valori**: li fornisce l'utente da fonti
pubbliche. Usato solo dal modulo di validazione (sezione 8).

### 3.6 File da toccare

`scripts/config_io.py` (schema, validazione, `build_plan`), `scripts/constellation.py` (uso dei nuovi campi),
`configs/README.md` (documentazione dello schema), `configs/sentinel2.constellation.json` (esempio aggiornato, con i limiti
dichiarati). Test: validazione positiva e negativa di ogni campo; i file `v1` esistenti devono dare esattamente gli stessi
risultati di oggi (regressione: 31 immagini e reward 13.7281 per un satellite, seed 4, 3 orbite; 44/43/75 con 3 satelliti
Walker).

## 4. Scalabilità (FISSATO: approvata il 2026-10-14)

Obiettivo: sapere come cresce il costo con il numero di satelliti prima di pianificare le campagne Monte Carlo.

- **Misure note** (da sessioni precedenti): circa 25-30 mila secondi simulati al secondo per un satellite; circa 12,5 mila con
  1 satellite e 2000 target in modalità senza limite di tempo; 139 round/s con episodio finito di 24 h e 116 round/s senza
  limite (1 satellite, round da 60 s). Con 1-3 satelliti. **Non si sa come scala oltre.**
- **Benchmark da realizzare** (`scripts/bench_scaling.py`): numero di satelliti N ∈ {1, 3, 10, 30, 100}, stesso episodio
  (24 h, round 60 s, target e stazioni fissi, `FreeCommunication`), 3 ripetizioni. Misurare: secondi simulati al secondo,
  tempo di `reset`, memoria (RSS) a fine episodio, tempo per round in funzione di N.
- **Uscita**: tabella in `outputs/` e un paragrafo in questo file (sezione 10) con i risultati.
- **Nota**: il validatore limita oggi a 64 i satelliti per file (`config_io.load_constellation`). È una scelta mia, non un
  limite di BSK-RL: da rivedere dopo il benchmark.
- **Criterio di successo**: curve misurate e riproducibili; nessuna modifica del codice di simulazione.

## 5. Richieste di imaging come flusso (FISSATO e implementato il 2026-10-14; vedi 5.9)

### 5.1 Decisioni fissate

1. **Il caso statico resta** (FISSATO). Lista fissa di target noti a priori, come oggi (`targets.mode: "random"`).
2. **Arrivi Poisson stazionari per ora** (FISSATO): le richieste arrivano come processo di Poisson con tasso costante
   `rate_per_hour`. Il tasso variabile nel tempo è rinviato alla fase di non stazionarietà (FUORI SCOPE); il punto di innesto
   è una funzione `rate(t)` che oggi restituisce una costante.
3. **Le richieste sono descritte da un file strutturato** (FISSATO nel principio; formato APERTO, vedi 5.3): ogni richiesta
   ha una **priorità**, un **target** e una **scadenza eventuale**.
4. **I satelliti ricevono le informazioni sui target al contatto con la stazione** (FISSATO come ipotesi di realismo). La
   ricerca sul web è in sezione 7. Una richiesta arrivata a terra è invisibile al satellite finché questo non ha un contatto
   di uplink.

### 5.2 Ciclo di vita di una richiesta (implementato nel modello B, vedi 5.9)

```
arrivo a terra (t_a) -> [ritardo uplink_latency_s] -> in attesa di uplink
   -> primo contatto di uplink del satellite k a t_c >= t_a + latenza  => nota al satellite k (t_known[k] = t_c)
   -> ripresa (imaged) | scaduta (expired) | ritirata
```

- Ogni satellite ha il proprio insieme di richieste note: due satelliti possono conoscere la stessa richiesta in istanti diversi.
- Un braccio è **disponibile** per il satellite k a un round se: è noto a k, non è scaduto, non risulta già ripreso (secondo
  l'informazione che k possiede), e ha una finestra di accesso fattibile.
- Stato "ripresa": con `crosslinks: free` l'informazione è condivisa subito; con `none`, i satelliti la ricevono solo
  dal segmento di terra al contatto successivo, quindi **possono riprendere due volte lo stesso target** (il reward
  `UniqueImageReward` non lo premia, ma il costo in risorse c'è). Questo è realistico e va **tenuto, non nascosto**.
  Dettaglio da fissare (APERTO, 9.5): quando la terra viene a sapere che un'immagine è stata fatta (al downlink?).

### 5.3 Formato del file di richieste (FISSATO e implementato: terzo file `requests/v1`)

L'utente vuole inserire le richieste con un file strutturato. Proposta di bozza `requests/v1`:

```json
{
  "schema": "requests/v1",
  "name": "esempio",
  "requests": [
    { "id": "R001", "target": { "lat": 45.07, "lon": 7.69, "name": "Torino" },
      "priority": 0.8, "arrival_s": 0, "expires_s": 86400 },
    { "id": "R002", "target": { "lat": 41.9, "lon": 12.5 },
      "priority": 0.3, "arrival_s": 3600 }
  ]
}
```

- `id`: identificatore **stabile** scelto dall'utente (unico). Se manca, lo assegna il simulatore con un contatore. È l'identità
  del braccio per il bandit: **non usare `Target.id` di BSK-RL** (contiene `id(self)`, cambia a ogni esecuzione).
- `target`: `lat` [-90, 90], `lon` [-180, 180], opzionale `name`, opzionale `elevation_m`. (Fase successiva: aree o tessere al
  posto dei punti.)
- `priority`: numero in (0, 1] come la priorità attuale `UniformTargets` (`np.random.rand()`); **range da confermare**.
- `arrival_s`: istante di arrivo a terra [s dall'inizio dell'episodio], default 0 (richiesta già presente).
- `expires_s`: istante assoluto di scadenza [s], oppure `validity_s` come durata dopo l'arrivo; **uno dei due**; assenti = non
  scade. Da decidere quale forma preferire (9.2).
- Domanda aperta (9.1): il file di richieste è **un terzo tipo di file** (`requests/v1`) o un blocco dentro lo scenario? La
  proposta è un terzo file opzionale, indicato dallo scenario (`targets.mode: "file"`, `targets.path` o importato dal viewer),
  così lo stesso elenco si riusa tra scenari.
- Le richieste del file e quelle generate da Poisson hanno **lo stesso formato interno**; la modalità `stream` genera le stesse
  strutture con un generatore.

### 5.4 Blocco `targets` dello scenario esteso (FISSATO e implementato; `spatial` ammette per ora solo `uniform`)

```json
"targets": {
  "mode": "random | stream | file",
  "count": 600,                       // random: numero di target statici (come oggi)
  "initial_count": 0,                 // stream: arretrato presente a t=0
  "arrival": { "type": "poisson", "rate_per_hour": 20 },
  "spatial": { "type": "uniform | land | cities" },
  "priority": { "type": "uniform", "min": 0.0, "max": 1.0 },
  "validity_hours": 48,               // assente = non scade
  "path": "requests.json"             // file: elenco di richieste
}
```

Regole di validazione: `mode: random` ammette solo `count`; `stream` ammette `initial_count`, `arrival`, `spatial`,
`priority`, `validity_hours`; `file` ammette `path`. Ogni altra combinazione è un errore. Il default resta `random` con
`count: 600` (compatibilità).

### 5.5 Riproducibilità (FISSATO come requisito)

- Gli arrivi vengono da generatori `np.random.Generator` **propri**, derivati dal seed dello scenario con
  `np.random.SeedSequence(seed).spawn(n)`: flussi indipendenti per tempi di arrivo, posizione, priorità, scadenza.
- La sequenza di richieste è **indipendente dalle azioni** dei satelliti: gli algoritmi confrontati vedono la stessa
  realizzazione. Per generarla in modo pigro (a blocchi) senza dipendere dalla politica, ogni flusso avanza sempre dello
  stesso numero di estrazioni per blocco.
- BSK-RL imposta l'`np.random` globale in `reset()` (verificato): i nostri generatori non devono dipenderne.

### 5.6 Cosa dice il codice di BSK-RL sull'aggiunta di target a episodio in corso (verificato leggendo `bsk_rl/sats/access_satellite.py`)

- `add_location_for_access_checking(object, r_LP_P, min_elev, type, start_time=0.0)` aggiunge un bersaglio mentre l'episodio
  gira. La documentazione avverte: considerato solo nelle **chiamate future** di `calculate_additional_windows`, non
  retroattivamente.
- Le finestre si calcolano **a blocchi** di `generation_duration` (default 600 s) e **in modo pigro**: `find_next_opportunities`
  chiama `calculate_additional_windows` finché non trova `n` opportunità (fino a `max_lookahead=100` volte). Quindi
  `window_calculation_time` può essere **ben oltre il tempo simulato corrente**. Un target aggiunto al tempo t non ha
  finestre tra t e `window_calculation_time`, a meno di calcolarle noi. **Rischio principale.**
- Ogni blocco ricalcola le finestre di **tutti** i target registrati (`locations_for_access_checking`): in un episodio infinito la
  lista cresce e il costo sale. Le richieste scadute o riprese vanno **rimosse dalla lista** (è una lista Python semplice).
- Una azione `Image` accetta solo target "noti": l'insieme `data_store.data.known` (`UniqueImageData.known`) va aggiornato per i
  nuovi target, per ciascun satellite al momento in cui li conosce (5.2). Verificato: `Image` solleva `ValueError` se il target
  non è noto.
- `UniqueImageReward.initial_data` passa a `known` tutti i target dello scenario all'inizio: nel flusso i target
  vanno aggiunti dopo.
- `scenario.targets` è una lista fissa creata da `regenerate_targets()` in `reset`: la gestione dinamica sta nel nostro codice.

### 5.7 Prototipo preliminare ("spike") sui target dinamici: ESEGUITO il 2026-10-14

Script scartabili (non nel repository). Un satellite, episodio infinito, round da 60 s, 200 target base più target aggiunti a
episodio in corso. **Esiti verificati:**

1. **Ritardo di visibilità**: `env.reset()` calcola già le finestre in avanti (serve a riempire l'osservazione di BSK-RL), e le
   chiamate successive le estendono a blocchi di 600 s. A t = 3000 s il calcolo era arrivato a 7200 s: un target aggiunto a t
   non ha finestre per i 4200 s già calcolati. Senza correzione si perdono le finestre in quell'intervallo.
2. **Correzione**: calcolare noi le finestre del solo nuovo luogo in [t, `window_calculation_time`] con la stessa logica di
   `calculate_additional_windows` (metodi privati `_find_candidate_windows`, `_find_elevation_roots`, `_refine_window`,
   `_add_window`; versione `bsk-rl==1.3.7` fissata). Verifica contro una **scansione indipendente a 1 s** dell'elevazione su
   150 target: 21 finestre reali, 0 perse, 0 in più.
3. **Il calcolo standard di BSK-RL è affidabile**: confronto con la scansione a 1 s su 1285 finestre (4 seed, 1500 target):
   0 perse. (Un primo test sembrava mostrare finestre perse, ma il «riferimento» era viziato dallo stesso problema del punto 1.)
4. **`Image` su un target aggiunto dinamicamente funziona**: con pool della stessa dimensione (200 target, 5 seed, 12 ore) il
   tasso di successo dei tentativi è lo stesso: 20% per i target statici (77 immagini su 379 tentativi) e 20% per i dinamici
   (76 su 384). **Nota di ricerca**: con la politica ingenua «prendi la finestra che si chiude per prima» l'80% dei tentativi
   di imaging fallisce (il satellite non fa in tempo a puntare): i bracci «geometricamente fattibili» sono spesso non
   eseguibili; conferma l'utilità di una stima del tempo di transizione (sezione 6.3, punto 3).
5. **Costo**: circa 10 µs per luogo registrato e per blocco da 600 s (300 luoghi: 3 ms; 1200: 11 ms; 5200: 53 ms per blocco);
   tenendo solo il 10% dei luoghi il costo scala di conseguenza (6 ms a 5200). Il costo è proporzionale ai luoghi *attivi*:
   rimuovere i target scaduti o ripresi è sufficiente a limitarlo.
6. `known` (`data_store.data.known`) **non** filtra le opportunità: `find_next_opportunities` usa solo il filtro «non ancora
   ripreso» (`UniqueImageReward.create_data_store`). La conoscenza a bordo dei target è quindi una nostra responsabilità
   (nel modello B il satellite esegue solo i task del piano caricato).

### 5.8 Viewer e formato della traccia (implementato, vedi 5.9 e sezione 13)

- Il formato v2 ha una lista fissa `targets`. Servono: `targets[].id` stabile (stringa), `arrival_s`, `expires_s` e un
  messaggio incrementale nello stream (`targets_added`, `targets_removed` con motivo `imaged`/`expired`) dentro il delta di
  ogni round. Per il formato statico non cambia nulla.
- Il viewer mostra le richieste in attesa (non ancora note al satellite selezionato con un simbolo diverso) e quelle scadute.
  **Progettazione di dettaglio da discutere** dopo lo spike.
- Memoria: la storia del viewer cresce con la durata. Con le richieste dinamiche serve una finestra di conservazione
  (es. ultime N ore) o il solo mantenimento dei target attivi. Da decidere (9.7).

### 5.9 Implementazione del modello B e delle richieste (2026-10-14)

Moduli: `scripts/config_io.py` (schemi, default, ruoli), `scripts/request_stream.py` (sorgenti e target con identità stabile),
`scripts/ground.py` (finestre, contatti, pianificatore, esecuzione, registro dati), `scripts/constellation.py` (`Sim`).

- **Modalità di `Sim`**: `legacy` (nessun file: invariata, regressione 31 immagini / 13,7281 e 44/43/75), `round`
  (`comms.uplink: continuous`: decisione a ogni round ma con richieste dinamiche) e `plan` (modello B, **default** per i file).
- **Modello B**: ai contatti con una stazione di ruolo `uplink` (e durata ≥ `min_uplink_contact_s`) il pianificatore riceve
  `ctx = {k, t_up, t_plan = t_up − latenza, horizon_end, arms, downlinks}` e restituisce attività `image`/`downlink`; il piano
  sostituisce il precedente e il satellite lo esegue (`Ground.execute`). Orizzonte = contatto di uplink successivo, al massimo
  `planning.horizon_max_s`. I bracci hanno identità stabile (id richiesta). Le richieste assegnate a un piano di un altro
  satellite non sono riproposte finché la loro finestra non si chiude. `DemoPlanner` **non è un bandit**.
- **Ruoli delle stazioni** (interpretazione della richiesta dell'utente: nel testo «diventa di uplink (e viceversa)» ho letto
  «se manca il downlink una stazione scelta a caso diventa downlink, e viceversa»): `config_io.fix_station_roles`; flusso
  casuale derivato dal seed; la stazione *cambia* ruolo se le altre coprono ancora quello che lascia, altrimenti li fa entrambi;
  ogni decisione è un avviso `warning` in `trace.meta.notices`, mostrato nel riquadro «Avvisi del simulatore» del viewer. I
  default usati compaiono come avvisi `info`.
- **Finestre**: `ground.WindowEngine` sostituisce il calcolo di BSK-RL negli episodi da file (motivo in `configs/README.md`):
  verificato su Sentinel-2 contro una scansione a 1 s: 454 finestre di target su 455 (l'unica persa dura 4 s), 48 passaggi
  di stazione su 48 (BSK-RL: 2 su 48).
- **Verifiche eseguite** (episodio di 24 h, 2 satelliti, flusso 15/h con 20 iniziali, validità 12 h): ids unici; nessuna immagine
  prima dell'arrivo o dopo la scadenza; nessun duplicato; nessuna immagine di richieste non note al satellite; ritardo
  arrivo→conoscenza a bordo mediano 2394 s (max 6080 s); stesso seed → stessa traccia (hash identico); la sequenza di richieste
  coincide con quella di una sorgente isolata (indipendente dalla politica); export `requests/v1` rileggibile e replay da file
  identico (26 immagini). Velocità 207-228 round/s (episodio infinito di 25 h con scadenze: 207 round/s).
- **Sensibilità osservata** (12 h, flusso 20/h): decisione continua 18 immagini; piano al contatto 11; piano con latenza di 1 h 6;
  contatti più brevi del minimo 0.
- **Memoria**: `sat.opportunities` non viene potata: con episodi molto lunghi cresce (nota già in sezione 9.7).

## 6. Caso di studio: satelliti agili (AEOS) (FISSATO come direzione il 2026-10-14; dettagli APERTI)

**Decisione dell'utente**: usare il contesto dei satelliti agili di osservazione della Terra (AEOS, *agile Earth observation
satellites*) come caso di studio e adattare il simulatore a questo contesto. L'utente ha letto per ora solo gli abstract dei due
articoli di riferimento e approfondirà; la scelta della costellazione specifica resta rinviata.

Fonti lette dall'assistente (testo estratto con `pypdf`; letti: introduzione, formulazione del problema, vincoli operativi,
variazioni e direzioni future del survey; introduzione e vincoli dell'altro articolo; **non** letti i metodi di soluzione né gli
esperimenti):
- Wang, Wu, Xing, Pedrycz, *Agile Earth observation satellite scheduling over 20 years: formulations, methods and future
  directions*, arXiv:2003.06169v1 (2020): survey di 62 articoli dal 1997 al 2019.
- Wang, Han, Leus, *Scheduling multiple agile Earth observation satellites with multiple observations*, arXiv:1812.00203v2
  (versione del 26 ottobre 2025): scheduling di più AEOS con osservazioni multiple dello stesso target, risolto con generazione
  di colonne; esperimenti su una costellazione cinese di AEOS (gap di ottimalità medio < 3%).

### 6.1 Il problema AEOSSP (come lo definiscono i due articoli)

- **Agile vs convenzionale**: un satellite convenzionale (CEOS) manovra solo sul rollio; la sua finestra di osservazione (OTW)
  coincide con la finestra di visibilità (VTW). Un AEOS manovra su tre assi (guarda avanti e indietro lungo il beccheggio): la
  VTW è più lunga della OTW e può contenere più OTW candidate, anche due o più osservazioni nella stessa VTW. Lo scheduling CEOS è
  una *selezione di VTW*; quello AEOS è molto più complesso (NP-hard).
- **Obiettivo**: selezionare e schedulare le osservazioni massimizzando il profitto totale sotto vincoli operativi.
- **Vincoli**:
  - **Tempo di transizione** tra osservazioni consecutive = tempo di manovra d'assetto ΔV + tempo di stabilizzazione ΔS; dipende
    dalle OTW scelte ed è fortemente non lineare per un AEOS.
  - **Risorse a bordo**: energia e memoria per orbita (consumo per unità di tempo di imaging EI/MI e di manovra EM, capacità EC/MC).
  - Un solo target alla volta; numero massimo di osservazioni desiderate per target N_i (tipicamente 1).
  - Requisiti di illuminazione e risoluzione complicano il calcolo delle VTW.
- **Semplificazione tipica**: il **downlink non è modellato** (si assume che ci siano abbastanza stazioni o relè) e la memoria si
  considera svuotata a ogni orbita. Il survey indica lo **scheduling integrato osservazione + downlink** come direzione di
  ricerca aperta (sezione 4.2 del survey).
- **Piano fisso offline**: nella formulazione base «i task schedulati non possono essere cambiati».
- **Tre varianti**: (i) *profitto*: target puntuali, target ad area (scomposti in strisce lungo la traccia, profitto non lineare
  nell'area acquisita) e target con più osservazioni (stereo, serie temporali; profitto non lineare nel numero di osservazioni);
  (ii) *multi-obiettivo* (equità tra utenti, qualità, tempestività, bilanciamento, energia); (iii) *modello autonomo* (decisioni
  reattive a bordo con informazione in tempo reale: nuvolosità, emergenze, richieste che cambiano dinamicamente; formulato anche
  come processo decisionale semi-markoviano).
- **Direzioni future** indicate dal survey: scheduling sotto incertezza (nuvolosità; circa il 35% delle immagini di Landsat-7 era
  inutilizzabile per le nuvole, secondo una fonte citata nel survey) combinato con piattaforme autonome; scheduling integrato con il
  downlink; «super-AEOS» con controllo d'assetto in tempo reale (strisce non parallele alla traccia, push-broom attivo, traiettorie
  non lineari).
- **Esempi reali** (Tab. 1 del survey): GeoEye-1, Pléiades (costellazione di due), WorldView, HJ-1, JL-1, SuperView-1, DEIMOS-2,
  KOMPSAT-5. I parametri orbitali e di agilità di ciascuno **non sono stati raccolti**: vanno cercati da fonti pubbliche, non
  inventati.

### 6.2 Che cosa il simulatore ha già e che cosa manca (verificato nel codice di BSK-RL 1.3.7)

| Elemento AEOSSP | In BSK-RL / nel nostro simulatore | Stato |
|---|---|---|
| VTW (finestra di visibilità) | Finestre di accesso da `imageTargetMinimumElevation` (campo di regard) | Presente |
| Tempo di transizione ΔV + ΔS | Emerge dalla **dinamica** (controllo d'assetto, ruote), non da una formula: `Image` fa puntare il satellite e scatta «appena si è stabilizzato» entro `imageAttErrorRequirement` | Presente, ma non prevedibile in forma chiusa |
| Scelta dell'istante di osservazione nella VTW (OTW) | **Non è una variabile**: l'immagine è scattata appena possibile; la decisione è *quale target* a ogni round | Differenza di modello |
| Fallimento di un'azione | `Image` fallisce se il target esce dal campo di regard prima della stabilizzazione | Presente (conta anche come costo di risorse) |
| Energia e memoria | Batteria con eclissi, memoria, potenza dello strumento attiva finché il task è abilitato (anche in manovra) | Presente |
| Downlink | Modellato con finestre di stazione e velocità | **Presente (il survey lo cita come lacuna aperta)** |
| Piano fisso offline | Noi ridecidiamo a ogni round (retasking) | Differenza: più vicino al «modello autonomo» |
| Target puntuali, una osservazione per target (N_i = 1) | `UniqueImageReward` conta un'immagine unica per target | Presente |
| Più osservazioni dello stesso target (N_i > 1) | Non supportato: servirebbe tenere noi il conteggio per target | Da aggiungere come proprietà della richiesta |
| Target ad area (strisce) | Solo punti | Da aggiungere (fase successiva) |
| Richieste dinamiche e di emergenza | Flusso di richieste (sezione 5); emergenza già deciso come canale separato | In progettazione |
| Vincoli di illuminazione/risoluzione | Filtro sulle finestre (sezione 3.3) | Da aggiungere |
| `ContinuousImagingFSWModel` (scansione nadirale continua) | Esiste in BSK-RL: serve a modellare un CEOS puramente nadirale | Non studiato |

### 6.3 Implicazioni di progetto (PROPOSTE; da discutere)

1. **Valori di `instrument.pointing`**: con il caso agile, `agile` è il valore principale (già così). `nadir` e `roll` (CEOS)
   servirebbero come **confronto** con un satellite convenzionale; `ContinuousImagingFSWModel` è un punto di partenza per `nadir`.
   `roll` richiederebbe di limitare l'assetto al solo rollio: non è chiaro se BSK-RL lo permetta; **da studiare**.
2. **Proprietà del target/richiesta**: aggiungere `max_observations` (N_i, default 1) al formato `requests/v1` (sezione 5.3),
   come proprietà strutturale. Il significato di profitto per osservazioni multiple resta **fuori scope** (reward escluso).
3. **Feasibility di un braccio**: in AEOSSP le transizioni infeasibili sono escluse dai vincoli; in BSK-RL sono una possibile
   esito di un'azione. Esporre al bandit solo i bracci geometricamente raggiungibili in tempo richiede una **stima** del tempo di
   transizione (surrogato calibrato sulla dinamica di BSK-RL, p. es. `t = ts + α·sqrt(|Δθ|)` come in Kacker et al. con α = 4,0 s/√deg
   e ts = 10 s per quel loro satellite). Da misurare in BSK-RL su un campione di manovre.
4. **Oracolo offline per il regret** (domanda aperta del `CLAUDE.md`): la letteratura AEOSSP offre formulazioni esatte e
   quasi-esatte: MILP e generazione di colonne con gap < 3% (articolo MAS). Un oracolo sui nostri episodi richiederebbe un modello
   di transizione compatibile con quello di BSK-RL (punto 3): **non banale**, e il regret rispetto a un oracolo che usa un
   surrogato misura anche l'errore del surrogato. Da discutere, non decidere ora.
5. **Downlink integrato**: poiché il simulatore modella già il downlink, il caso di studio può posizionarsi nella lacuna
   «scheduling integrato» indicata dal survey (4.2). È un'osservazione per la tesi; il suo valore di contributo va valutato con il
   relatore.
6. **Costellazioni candidate**: scegliere tra quelle della Tab. 1 del survey o altre agili; servono parametri pubblici
   (orbita, campo di regard, agilità, memoria, batteria, downlink). Da raccogliere dall'utente.
7. **Sentinel-2** non è un AEOS (nadirale): il file di esempio resta valido come esempio di costellazione, ma non come
   caso di studio agile. Eventualmente affiancarlo da un esempio agile.

### 6.4 Non implementare prima di ulteriori decisioni

Nulla di questa sezione è implementato. Dipende dal feedback dell'utente dopo la lettura completa degli articoli.

## 7. Ricerca: informazioni sui target ricevute al contatto con la stazione

Ricerca web del 2026-10-14. **Limiti**: ho letto riassunti di ricerca e, per l'articolo di Kacker et al., il **testo completo**
(8 pagine, fornito dall'utente in PDF, estratto con `pypdf`). Le altre fonti sono in gran parte brevetti e articoli di conferenza,
letti solo nei riassunti; non ho trovato valori operativi attuali per specifici operatori commerciali. I tempi citati
sono **esempi, non standard**.

### 7.1 Cosa emerge (struttura comune del ciclo di pianificazione)

1. **Raccolta delle richieste** dai clienti; alcuni operatori permettono tempo esatto di osservazione o finestra preferita.
2. **Pianificazione dei contatti** con le stazioni a terra, calcolati dalla visibilità orbitale e prenotati; la prenotazione non è
   istantanea (le stazioni hanno bisogno di tempo per mettere in coda l'invio).
3. **Pianificazione delle acquisizioni** sotto vincoli (potenza, tempo di riorientamento tra puntamenti, memoria, termico).
4. **Generazione dei comandi** di basso livello, inclusi i downlink automatici.
5. **Uplink durante il passaggio sulla stazione**: i comandi e il piano di raccolta salgono al satellite nel contatto. Esempio
   storico (IKONOS, descrizione operativa): apparecchiature collegate all'antenna ~10 minuti prima del contatto; a contatto
   avvengono autenticazione e invio del piano di raccolta in una finestra di circa 25 s; ottimizzazione dei candidati ~3 ore prima del passaggio.
6. **Downlink e tracciamento** dei dati con programmi di ricezione.
7. **Ripianificazione** al cambiare delle previsioni; ripiego su una stazione alternativa se una è fuori servizio.

### 7.2 Stazioni di comando e di dati sono spesso diverse (caso Sentinel-2, dalle pagine ESA; dati del 2014-2017, da riverificare)

- Comandi e telemetria di servizio: principalmente in banda S dalla stazione di **Kiruna** (Svalbard non è la stazione di
  telecomando).
- Dati di payload: banda X sulle stazioni core di **Matera, Maspalomas, Svalbard**.
- Il collegamento laser verso EDRS (relè) è un'ulteriore via per i dati quando il satellite non è in vista di una stazione in banda X.
- Conseguenza di progetto: da qui `ground_stations[].roles` (3.2). Il file `sentinel2.constellation.json` attuale elenca solo le
  tre stazioni X-band: per questa ricerca andrebbe **aggiunta Kiruna con ruolo `uplink`** (coordinate da verificare).

### 7.3 Il ritardo di uplink come parametro (Kacker, Chien, Devaraj, Demir, Cahoy, IWPSS 2025; testo completo letto)

Contesto dello studio: tasking dinamico di un satellite agile in base a maschere di nuvolosità da satelliti meteorologici
geostazionari. Il tema è il meteo (fuori scope per noi), ma la parte sulla **latenza di uplink** è direttamente utile.

- **Tre modi di far arrivare un'informazione al satellite** (loro Fig. 1): (1) tramite **stazione a terra**: il dato si scarica,
  si estrae da un database e si invia al satellite nel contatto, con *buchi di comunicazione*; (2) **ricezione diretta** a bordo
  quasi in tempo reale; (3) **relè** (Iridium, GlobalStar, TDRS) in tempo reale. Con (2) e (3) le decisioni sono in tempo reale;
  con (1) **gli aggiornamenti arrivano circa ogni 45 minuti** per un satellite in orbita bassa (cadenza indicata in Fig. 1;
  nelle conclusioni parlano di informazione trasmessa «ogni mezza orbita»).
- **Parametrizzazione**: non assumono un'architettura precisa; variano il **ritardo di informazione da 0 a 2 ore** (valori
  0, 30, 60, 90, 120 minuti in Fig. 7), sia con collegamento in tempo reale sia con contatti intermittenti dalle stazioni.
- **Risultato rilevante per il modello**: il ritardo medio aggiuntivo dovuto ai buchi di contatto è circa **22,5 minuti**
  (coerente con metà dei 45 minuti, cioè con l'attesa media del prossimo contatto: è una nostra lettura, l'articolo non lo scrive); la modalità via stazione rende circa **2 punti
  percentuali** in meno del tempo reale, e le due si equivalgono circa a 2 ore di ritardo, perché i 22,5 minuti pesano meno.
  Un'informazione ritardata vale meno: già dopo 2 ore la maggior parte del potere predittivo della maschera di nuvolosità è
  persa. **Questa è una conferma che l'uplink al contatto introduce un ritardo realistico e non trascurabile.**
- **Scenario simulato** (loro Tab. 2): **un solo satellite** in orbita tipo ISS (400 km, 51,6°, periodo 5553,5 s), orizzonte di
  pianificazione 12 h, campo di regard 30°, tempo di assestamento 10 s, modello di riorientamento
  `t_slew = t_s + α·sqrt(|Δθ|)` con α = 4,0 s/√deg; **si pianifica solo fuori dall'eclisse**; domanda di immagini simulata con le
  **10.000 città più grandi del mondo** (dataset Simplemaps, lo stesso usato da `CityTargets` di BSK-RL), e un'utilità 1 per
  ogni immagine non coperta dalle nuvole.
- **Metodi di confronto**: schedulatore convenzionale, schedulatore onnisciente teorico (limite superiore, MILP ottimo con
  vincoli di agilità e di ripetizione) e schedulatori con informazione ritardata. Il MILP usa due vincoli: agilità (il
  riorientamento tra due task deve stare nel tempo disponibile) e ripetizione (un'unica acquisizione per richiesta). Utile come
  idea per l'**oracolo offline del regret**, decisione ancora aperta nel `CLAUDE.md`.
- **Classificazione delle missioni**: *monitoring* (acquisizione di quanta più immagine possibile senza bersagli specifici,
  copertura ampia, risoluzione più bassa) e *tasked* (bersagli specifici a risoluzione maggiore, che richiedono input esterno
  per guidare il tasking). Entrambi faticano con fenomeni imprevedibili. Rilevante per la scelta della costellazione (sezione 6).
- **Limiti dello studio**: risultati da un singolo istante di partenza (1 gennaio 2025) e un solo satellite; non generalizza
  su stagioni, regioni, orbite e configurazioni di stazioni, come dichiarano gli autori.

### 7.4 Riassunto sul ritardo (da fonti brevettuali e articoli di rete)

- I contatti con le stazioni sono brevi: un passaggio dura circa dieci minuti, con quattro-sei contatti al giorno per una singola
  stazione (stima citata da una fonte di rete, non verificata in dettaglio). Con più stazioni, l'articolo di Kacker indica
  aggiornamenti circa ogni 45 minuti per un satellite in LEO (7.3).
- Mitigazioni: stazioni distribuite, collegamenti intersatellitari, satelliti dedicati alla consegna dei comandi, elaborazione a bordo.

### 7.5 Cosa implica per il simulatore (PROPOSTO)

Due modi di modellare l'uplink (**l'utente ha scelto il modello B il 2026-10-14**: il bandit gira a terra):

- **A. Solo conoscenza** (PROPOSTO come primo passo): al contatto di uplink il satellite apprende le richieste nuove e quelle
  scadute/riprese; tra un contatto e l'altro **sceglie da solo** tra le richieste note, a ogni round. L'architettura a round resta
  invariata.
- **B. Piano completo**: al contatto sale un piano (sequenza di acquisizioni) che il satellite esegue senza scelte fino al contatto
  successivo. È più fedele al caso classico, ma **cambia la definizione di round**: il bandit deciderebbe a lotti per contatto.
  Da valutare dopo A.

Meccanica per A:
- Contatto di uplink di k = finestra aperta con una stazione che ha il ruolo `uplink` (le finestre stazione-satellite sono già
  calcolate da BSK-RL: `sat.opportunities` con `type == "ground_station"`, usate in `constellation.py::_choose`).
- Se la finestra dura meno di `min_uplink_contact_s`, il contatto non vale.
- Una richiesta con `arrival_s + uplink_latency_s <= t_c` diventa nota a k al primo contatto utile.
- Con `uplink: continuous` il contatto è sempre aperto (comportamento attuale: i target noti subito).
- L'attuale Walker/viewer con `FreeCommunication` continua a funzionare con `continuous`.

### 7.6 Fonti

Vedi sezione 11.

## 8. Altre feature approvate o in sospeso

- **Target modulari** (generatori `uniform`, `land`, `cities`, importazione da file): **PROPOSTO, in discussione con il punto 2**.
  Il generatore `uniform` è quello attuale; `land` usa la maschera di terra già costruita da `scripts/build_basemap.py`
  (`viewer/web/static/coast.json` e immagine); `cities` usa `bsk_rl.scene.CityTargets` (`worldcities.csv` incluso in BSK-RL).
  Va usato un `np.random.Generator` proprio (BSK-RL usa `np.random` globale per i target).
- **Risorse e volumi dati parametrici**: PROPOSTO. Il campo `data.bits_per_acquisition` è approvato (3.1); la capacità di
  memoria e di batteria reali vanno inseriti con i valori reali dalle fonti pubbliche, **non inventati**. Degrado graduale
  della batteria: domanda aperta di `CLAUDE.md` (fattibilità non verificata).
- **Vincoli per tipo di strumento**: l'elenco dei campi è approvato (3.1). Aggancio tecnico verificato (3.3).
- **Comunicazione realistica** (crosslink, contesa delle stazioni): PROPOSTO. Non è verificato se BSK-RL impedisca a due
  satelliti di usare la stessa antenna insieme.
- **Oracolo offline per il regret** (spunto, non una decisione): Kacker et al. usano un MILP con vincoli di agilità e di
  ripetizione come schedulatore onnisciente ottimo; per noi sarebbe un'opzione sulle istanze generate dal simulatore (7.3).
- **Validazione generica e orbite da TLE**: PROPOSTO. Modulo che per qualunque configurazione calcola copertura, tempo di
  rivisita e uso delle risorse, confrontandoli con `reference`. Un tipo di orbita `tle` è opzionale (propagazione BSK-RL, valida
  per ore o pochi giorni; richiede una fonte pubblica dei TLE).

## 9. Domande e decisioni (aggiornato il 2026-10-14)

Risposte dell'utente alla discussione sulle cinque domande aperte:

| # | Domanda | Decisione | Stato |
|---|---|---|---|
| 9.1 | File di richieste: terzo file o blocco dello scenario? | **Terzo file `requests/v1`**, con esportazione del flusso generato | FISSATO, implementato |
| 9.2 | Scadenza: assoluta o relativa? | **Entrambe** (`expires_s` o `validity_s`, esclusive), normalizzate in istante assoluto; assente = non scade | FISSATO, implementato |
| 9.3 | Priorità | **Qualsiasi numero > 0** nei file; default del generatore: uniforme in (0, 1]. Classi discrete: non richieste | FISSATO, implementato |
| 9.4 | Uplink A (solo conoscenza) o B (piano completo)? | **B**: il bandit gira a terra (con possibile decentralizzazione futura); decisioni a lotti al contatto di uplink | FISSATO, implementato (vedi 5.9) |
| 9.5 | Quando la terra sa che un'immagine è stata fatta? | Al **downlink** (registro FIFO sulla memoria); l'informazione va al satellite al contatto successivo | FISSATO (raccomandazione non contestata), implementato |
| 9.6 | Ruoli stazioni e parametri di uplink | **Default per i parametri non specificati** (neutri, in `config_io.UPLINK_DEFAULTS`); **ruoli** `uplink`/`downlink` per stazione con correzione automatica casuale e riquadro di avviso nel viewer (vedi 5.9) | FISSATO, implementato |
| 9.7 | Memoria storica del viewer con episodi infiniti | non discussa | APERTO |
| 9.8 | `pointing: nadir`/`roll` | L'utente resta sul caso **agile** e approfondirà | IN ATTESA |
| 9.9 | Richiesta ripresa: si ritira dall'insieme attivo? | Sì, quando la terra la sa ripresa (downlink) o scade | implementato |
| 9.10 | Costellazione agile del caso di studio | L'utente **non vuole decidere per ora** | APERTO |
| 9.11 | `max_observations` (N_i) | non discussa | APERTO (non implementato) |

## 10. Piano di implementazione (ordine e criteri di accettazione)

Ogni passo si chiude con: regressione identica sui percorsi esistenti, prova nel viewer, documentazione aggiornata, e **conferma
dell'utente prima di ogni commit**.

| Passo | Contenuto | Stato |
|---|---|---|
| A | Schema esteso: **fatti** `comms.*`, `ground_stations[].roles`, `scenario.targets` (random/stream/file), `scenario.planning`, `requests/v1`; **da fare** `instrument.kind/pointing/constraints`, `data.bits_per_acquisition`, `reference` (approvati ma senza effetto: non aggiunti per non creare campi che non fanno nulla) | parziale |
| B | Benchmark di scalabilità (sezione 4) | da fare |
| C | Spike dei target dinamici (5.7) | **fatto** |
| D | Generatori di target modulari (`land`, `cities`): il generatore `uniform` con `Generator` proprio c'è (`request_stream.py`) | parziale |
| E | Flusso di richieste con ciclo di vita, uplink al contatto, piano B | **fatto** (5.9) |
| F | Viewer: avvisi, richieste dinamiche, piano, editor delle richieste | **fatto** |
| G | Vincoli per strumento (illuminazione, incidenza) | da fare |
| H | Validazione generica e `reference` | da fare |
| I | Adattamento al caso agile: surrogato del tempo di transizione, `max_observations`, confronto `nadir`/`roll` | in attesa dell'utente |
| J | Pianificatore bandit al posto di `DemoPlanner` (stessa interfaccia `plan(ctx)`) | futuro |

## 11. Riferimenti

### 11.1 Agile / nadirale / scheduling (consultazione in corso da parte dell'utente)

- Survey del problema di scheduling per satelliti agili: *Agile Earth observation satellite scheduling over 20 years:
  formulations, methods and future directions*, arXiv:2003.06169 (2020). https://arxiv.org/abs/2003.06169
- Introduzione chiara alla differenza convenzionale/agile: *Scheduling multiple agile Earth observation satellites with multiple
  observations*, arXiv:1812.00203. https://arxiv.org/pdf/1812.00203
- Survey ampio: Ferrari, Cordeau, Delorme, Iori, Orosei, *Satellite scheduling problems: A survey of applications in Earth and
  outer space observation*, Computers & Operations Research, 2025. https://repository.tilburguniversity.edu/items/bd1b7ece-3609-49b0-8155-5d5670121643
- Lemaître, Verfaillie, Jouhaud, Lachiver, Bataille, *Selecting and scheduling observations of agile satellites*, Aerospace
  Science and Technology 6(5), 2002. DOI 10.1016/S1270-9638(02)01173-2 (trovato solo tramite bibliografie, testo non letto).

### 11.1b Satelliti convenzionali (rollio) e acquisizione nadirale sistematica (consultazione richiesta dall'utente il 2026-10-14)

**Scheduling di satelliti convenzionali (CEOS, solo rollio)** — il filone «classico», in gran parte sul satellite SPOT 5:
- Vasquez, Hao, *A "logic-constrained" knapsack formulation and a tabu algorithm for the daily photograph scheduling of an
  earth observation satellite*, Computational Optimization and Applications 20(2), 2001, pp. 137-157. https://hal.archives-ouvertes.fr/hal-00359386 (letto solo l'abstract)
- Vasquez, Hao, *Upper bounds for the SPOT 5 daily photograph scheduling problem*, J. Combinatorial Optimization 7(1), 2003,
  pp. 87-103. https://leria-info.univ-angers.fr/%7Ejinkao.hao/papers/JOCO03.pdf (letto solo l'abstract)
- Benchmark pubblico: 21 istanze SPOT 5 (da un simulatore CNES dell'ordine di acquisizione). https://data.mendeley.com/datasets/2kbzg9nw3b/1
- Bensana, Lemaître, Verfaillie, *Earth observation satellite management*, Constraints 4(3), 1999 (solo record bibliografico).
- Gabrel, Vanderpooten, *Enumeration and interactive selection of efficient paths in a multiple criteria graph for scheduling an
  earth observing satellite*, EJOR 139(3), 2002; Lin, Liao, Liu, Lee, *Daily imaging scheduling of an earth observation
  satellite* (ROCSAT-II), IEEE Trans. SMC-A 35(2), 2005; Wolfe, Sorensen, *Three scheduling algorithms applied to the earth
  observing systems domain*, Management Science 46(1), 2000 (tutti descritti come lavori CEOS nella rassegna dell'articolo
  arXiv:1812.00203, sezione 2.1; non letti).
- Con incertezza di nuvolosità (fuori scope per ora): Wang, Song, Leus, Han, *Robust Earth observation satellite scheduling with
  uncertainty of cloud coverage*, IEEE Trans. Aerospace and Electronic Systems 56(3), 2020; Wang, Demeulemeester, Qiu, *A pure
  proactive scheduling algorithm for multiple earth observation satellites under uncertainties of clouds*, Computers & OR, 2016
  (riferimenti presi dalla bibliografia di arXiv:1812.00203; non letti).

**Acquisizione nadirale sistematica (missioni «monitoring»)** — non è un problema di ottimizzazione formale ma un piano operativo:
- Crawford et al., *The Landsat 8/9 Long Term Acquisition Plan for Earth's continental landmasses and near-shore coastal zones,
  Version 1.0*, USGS EROS, 2025, DOI 10.5066/P1WXQ2JN. https://www.usgs.gov/data/landsat-89-long-term-acquisition-plan-earths-continental-landmasses-and-near-shore-coastal
  (letta la pagina descrittiva: priorità per scena WRS-2 ogni giorno, requisito di luce con Sole ≥ 5° al centro scena, limiti di
  contatto con le stazioni e di registratore, richieste di emergenza che prevalgono).
- Arvidson, Goward, Gasch, Williams, *Landsat-7 long-term acquisition plan: development and validation*, Photogrammetric
  Engineering & Remote Sensing 72(10), 2006, pp. 1137-1146, DOI 10.14358/PERS.72.10.1137. (Solo abstract via ricerca.)
- ESA, documentazione di Sentinel-2 e dei piani di osservazione (vedi 11.2).

**Nota**: non ho trovato una fonte di ottimizzazione formale dedicata al puro push-broom nadirale: nella letteratura è il caso
«convenzionale» più restrittivo o il piano operativo sistematico di cui sopra.

### 11.2 Uplink e contatti con le stazioni

- Kacker, Chien, Devaraj, Demir, Cahoy, *Leveraging Realtime Meteorological Data for Dynamic Tasking of Agile Earth-Observing
  Satellites*, 14th International Workshop on Planning and Scheduling for Space (IWPSS), Toulouse, aprile 2025 (MIT, JPL,
  Planet Labs; CC BY-NC-ND 4.0). https://ai.jpl.nasa.gov/public/documents/papers/kacker-iwpss-2025.pdf
  (testo completo letto, 8 pagine).
- ESA, pagine su Sentinel-2: dati e operazioni (comandi via Kiruna in banda S; dati X-band a Matera, Maspalomas, Svalbard):
  https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Data_flow ,
  https://sentinels.copernicus.eu/copernicus-operations/mission-planning ,
  https://sentinel.esa.int/en/web/sentinel/missions/sentinel-2 (dati del 2014-2017: verificare lo stato attuale).
- Letteratura sul ciclo di pianificazione e uplink (brevetti e conferenze, riassunti letti, non i testi completi):
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11954143 ,
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12617552 ,
  https://icaps07-satellite.icaps-conference.org/demos/4/ddnnc.pdf (ESA ground station allocation).

### 11.3 Da verificare (non confermato in questa sessione)

- Sentinel-2 nadirale con swath di 290 km, quota 786 km, passaggio 10:30: valori nel file di esempio, marcati "da verificare".
- Kleinberg et al., 2010, sui *sleeping bandits* come riferimento per bracci che compaiono e scompaiono: ricordato a memoria,
  **da cercare e verificare** prima di citarlo in tesi.

## 12. Fatti verificati su BSK-RL 1.3.7 (riferimento per l'implementazione)

| Fatto | Dove |
|---|---|
| `add_location_for_access_checking(object, r_LP_P, min_elev, type, start_time=0.0)`; solo per chiamate future di `calculate_additional_windows` | `bsk_rl/sats/access_satellite.py:73` |
| Finestre a blocchi di `generation_duration=600` s; calcolo pigro fino a trovare `n` opportunità (`max_lookahead=100`) | `access_satellite.py` (`calculate_additional_windows`, `find_next_opportunities`) |
| Ogni blocco ricalcola tutti i target in `locations_for_access_checking` | `access_satellite.py::calculate_additional_windows` |
| `add_access_filter(fn)` e parametro `filter` di `find_next_opportunities`: gancio per vincoli su finestre | `access_satellite.py` |
| `known` = target noti; `Image` solleva `ValueError` per target non noto | `bsk_rl/data/unique_image_data.py`, `access_satellite.py:664` |
| Reward assegnato al momento dell'immagine, non al downlink | `UniqueImageReward` |
| Stazioni: `groundStationsData` con `name`, `lat`, `long`, `elev` e `gsMinimumElevation`, `gsMaximumRange`, `groundLocationPlanetRadius` per singola stazione | `bsk_rl/sim/world.py::_create_ground_station` |
| `FreeCommunication`, `LOSCommunication` (richiede `LOSCommDynModel`), altre | `bsk_rl/comm/communication.py` |
| `CityTargets` e `worldcities.csv` | `bsk_rl/scene/targets.py`, `bsk_rl/_dat/` |
| `Target.id` contiene `id(self)`: instabile | `bsk_rl/scene/targets.py` |
| BSK-RL seeda `np.random` globale in `reset()`; i target usano `np.random.rand()` | `bsk_rl/gym.py`, `targets.py` |
| Nessun modello di nuvolosità né di illuminazione del target nel wheel | ricerca su `bsk_rl/` |
| `bsk_rl/data/nadir_data.py` esiste, **non letto** | `bsk_rl/data/` |
| Il calcolo delle finestre di BSK-RL **perde passaggi** quando la finestra candidata è molto più lunga del passaggio (elevazione minima bassa e quota alta): Sentinel-2 a 786 km, stazioni a 5°: 2/24 e 0/24 passaggi trovati; con 7 stazioni a 10° e 500 km li trova tutti (20/20, 22/22). Sui target a 45° e 78°: 0 perse su 1285 | `access_satellite.py::_find_elevation_roots` (ramo `minimize_scalar`), misurato 2026-10-14 |
| Un'azione `Image` può fallire (target fuori dal campo di regard prima della stabilizzazione): con la politica ingenua ~80% dei tentativi | spike 5.7 |
| `ImagingSatellite(initial_generation_duration=0, max_generation_duration_beyond_initial=0)` impedisce a BSK-RL di calcolare finestre | usato in `pilot_bskrl.make_sat(observe_opportunities=False)` |

## 13. Stato dell'implementazione esistente (non da rifare)

- `scripts/config_io.py`: validazione di `constellation/v1`, `scenario/v1`, `requests/v1`; `build_plan` (argomenti BSK-RL, default,
  ruoli delle stazioni con correzione casuale riproducibile, avvisi).
- `scripts/request_stream.py`: `PoissonSource`, `FileSource`, `RequestTarget`, esportazione `requests/v1`.
- `scripts/ground.py`: `WindowEngine`, `Ground` (contatti, piano, esecuzione, registro FIFO), `DemoPlanner`.
- `scripts/constellation.py`: `Sim` con modalità `legacy`/`round`/`plan`, traccia v2 estesa (`targets_removed`, `uplinks`,
  `meta.notices/mode/comms`; delta con `targets_added`/`targets_removed`/`uplinks`).
- `scripts/sim_server.py`: stream live, `POST /scenario` (anche `requests`), `GET /config`, `GET /requests.json`.
- Viewer: importazione e editor dei file (anche richieste), riquadro avvisi, richieste dinamiche (in attesa di uplink, nel piano,
  scadute), dati di piano nel riquadro satellite.
- `configs/`: `sentinel2.constellation.json`, `default.scenario.json`, `stream.scenario.json`, `requests.example.json`,
  `requests.scenario.json`, `README.md`.

## 14. Registro delle modifiche a questo file

- 2026-10-14: creazione. Fissati: principio generale, esclusione di reward/QC/meteo, elenco dei campi dello schema, scalabilità,
  caso statico, arrivi Poisson stazionari, uplink al contatto. In attesa: punto 3 (agile/nadirale). Ricerca web sull'uplink
  aggiunta in sezione 7.
- 2026-10-14 (aggiornamento): letto per intero l'articolo di Kacker et al. (PDF fornito dall'utente). Aggiunti in 7.3 i dati
  sulla cadenza di uplink (~45 min via stazione, ritardo medio ~22,5 min), la parametrizzazione 0-2 h, lo scenario simulato e il
  modello di riorientamento; corretta la descrizione dello scenario (un solo satellite, non una costellazione).
- 2026-10-14 (agile): fissato il contesto agile (AEOS) come caso di studio. Letti i due articoli (parti indicate in sezione 6);
  riscritta la sezione 6 con il confronto AEOSSP / BSK-RL e le proposte; aggiunte le domande 9.10 e 9.11.
- 2026-10-14 (riferimenti CEOS/nadir): aggiunta la sezione 11.1b. Discussione sulle domande aperte 9.1-9.6 in corso con l'utente.
- 2026-10-14 (implementazione B): decisioni 9.1-9.6 prese dall'utente e implementate (sezione 5.9); spike dei target dinamici
  eseguito (5.7); scoperto e aggirato il difetto di BSK-RL sulle finestre di stazione (sezione 12). Il punto 3 (agile) resta
  in attesa e la costellazione non è stata scelta.
