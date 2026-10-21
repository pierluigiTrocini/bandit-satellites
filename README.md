# Scheduling di costellazioni EO con Online Learning

Codice di supporto alla tesi magistrale di Pierluigi Trocini (Università della Calabria / FH OÖ Hagenberg):
scheduling di costellazioni di Earth Observation come bandit combinatorio con vincoli di risorsa in
ambiente non stazionario, costruito sopra [BSK-RL](https://github.com/AVSLab/bsk_rl).

**Stato:** lavoro in corso. Oggi il repository contiene l'infrastruttura di simulazione e un viewer web con
una politica dimostrativa; gli algoritmi di bandit (OMD/EXP3, Thompson Sampling), le baseline e l'oracolo per
il regret non ci sono ancora.

## Contenuto

- `scripts/`: episodio e politica dimostrativa (`constellation.py`), pianificazione a terra (`ground.py`),
  stream live (`sim_server.py`), esportazione tracce, validazione delle configurazioni.
- `configs/`: costellazioni e scenari in JSON (schema in `configs/README.md`).
- `viewer/`: viewer web 2D/3D delle tracce e dello stream live (servito da nginx via Docker).
- `docs/decisioni-simulatore.md`: scelte di modellazione e limiti di astrazione.

## Uso

Richiede [uv](https://docs.astral.sh/uv/) (Python 3.12) e, per il viewer, Docker.

```bash
uv sync
PYTHONPATH=scripts uv run python scripts/export_trace.py --sats 3 --planes 3
docker compose up -d        # viewer su http://localhost:8080
```

Per la mappa di base: `uv run python scripts/build_basemap.py` (scarica i dati Natural Earth).

## Licenza e crediti

Codice sotto licenza MIT (`LICENSE`). Dipende da BSK-RL (MIT) e Basilisk (ISC), non copiati nel repository;
la mappa di base deriva da Natural Earth (pubblico dominio). Dettagli in `NOTICE`. Il progetto non è affiliato
né approvato da AVS Lab.
