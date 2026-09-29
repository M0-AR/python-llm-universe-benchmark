# Zero to Hero (learn by running, 15 min)

1. **Read the map**: `README.md` → `docs/BEST_PRACTICES_2026_2027.md` → `docs/HIDDEN_PATTERNS.md`.
2. **Run offline**:
   ```bash
   pip install -r requirements.txt && pip install -e .
   pytest -q
   python scripts/run_all.py --out results/universe_report.json
   cat results/universe_report.json | head -n 80
   ```
3. **Prove reproducibility** (no hand-edit without verify):
   ```bash
   python scripts/verify_repro.py  # must print REPRODUCIBLE
   ```
4. **Break things on purpose** (learn hidden patterns):
   - Change `HELLO` tokenization in `src/universe_bench/tokenization_bias.py` → `pytest tests/test_tokenization_bias.py` must FAIL (good: test guards truth).
   - Add a canary to a test string → `test_contamination` catches it.
5. **Docker (shareable)**:
   ```bash
   docker compose config
   docker compose up --abort-on-container-exit bench
   # results/ gets versions.txt + pytest.txt + universe_report.json
   ```
6. **Next**: pick one PhD chapter in `docs/PHD_ROADMAP.md` and extend its suite YAML in `benchmarks/`.
