.PHONY: setup lint format test test-cov demo docker-demo generate run validate deliver eval bench dashboard minio-up minio-down clean help

PYTHON ?= uv run python
UV ?= uv

setup:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check src/ tests/ configs/
	$(UV) run mypy src/ tests/

format:
	$(UV) run ruff format src/ tests/ configs/

test:
	$(UV) run pytest tests/ -v

test-cov:
	$(UV) run pytest tests/ -v --cov=src/episodeforge --cov-report=term-missing

demo:
	$(UV) run forge generate --episodes 2 --workers 1
	$(UV) run forge run --spec configs/clients/client_a_lerobot.v1.yaml
	$(UV) run forge spec lint configs/clients/client_a_lerobot.v1.yaml
	@echo "Demo pipeline finished successfully!"

docker-demo:
	docker compose run --rm forge-demo

generate:
	$(UV) run forge generate --episodes 10 --workers 2

run:
	$(UV) run forge run --spec configs/clients/client_a_lerobot.v1.yaml

validate:
	$(UV) run forge spec lint configs/clients/client_a_lerobot.v1.yaml
	$(UV) run forge spec lint configs/clients/client_a_lerobot.v2.yaml
	$(UV) run forge spec lint configs/clients/client_b_rlds.v1.yaml
	$(UV) run forge spec lint configs/clients/client_c_robomimic.v1.yaml
	$(UV) run forge spec lint configs/clients/client_d_egocentric_webdataset.v1.yaml

deliver:
	$(UV) run forge deliver data/releases/sample --spec configs/clients/client_a_lerobot.v1.yaml

eval:
	$(UV) run forge eval --suite quick

bench:
	$(UV) run forge bench

dashboard:
	$(UV) run streamlit run dashboard/app.py

minio-up:
	docker compose up -d minio createbuckets

minio-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} +
