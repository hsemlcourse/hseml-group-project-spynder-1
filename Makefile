.PHONY: lint lint-py lint-notebooks fix fix-py fix-notebooks test train run-api docker-build docker-up docker-down

lint: lint-py lint-notebooks

lint-py:
	python -m flake8 src tests

lint-notebooks:
	python -m nbqa flake8 notebooks

fix: fix-py fix-notebooks

fix-py:
	python -m ruff check src tests --fix
	python -m ruff format src tests

fix-notebooks:
	python -m ruff check notebooks --fix
	python -m ruff format notebooks

train:
	python src/modeling.py

run-api:
	uvicorn src.app:app --reload

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down
