.PHONY: setup test bench verify compose-config compose-build compose-up lint
setup:
	pip install -r requirements.txt
	pip install -e .
test:
	PYTHONHASHSEED=0 pytest -q
bench:
	PYTHONHASHSEED=0 python scripts/run_all.py --out results/universe_report.json
verify: compose-config test bench
compose-config:
	docker compose config >/dev/null && echo "compose OK"
compose-build:
	docker compose build bench
compose-up:
	docker compose up --abort-on-container-exit bench
lint:
	python -m py_compile $$(find src scripts tests -name "*.py")
