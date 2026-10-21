# Bandit Satellites - Online Learning for Earth Observation scheduling.

Master Thesis' Project (Università della Calabria / FH OÖ Hagenberg).
Optimized EO scheduling as combinatorial bandit with resources constraints in non-stationary environment.
The following code is a satellite simulator, based on [BSK-RL](https://github.com/AVSLab/bsk_rl).

## Usage
[uv](https://docs.astral.sh/uv/) (Python 3.12) and Docker required.

```bash
uv sync
PYTHONPATH=scripts uv run python scripts/export_trace.py --sats 3 --planes 3
docker compose up -d        # viewer su http://localhost:8080
```
For base Earth map: `uv run python scripts/build_basemap.py`

## License and credits
Code licensed under the MIT License (`LICENSE`). It depends on BSK-RL (MIT) and Basilisk (ISC), which are not included in the repository;
the base map is derived from Natural Earth (public domain). See `NOTICE` for details. This project is not affiliated with
or endorsed by AVS Lab.
