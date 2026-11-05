.PHONY: install test run up down site
install:
	pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
test:
	pytest -q
run:
	uvicorn main:app --reload --port 8000
up:
	docker compose up --build -d
down:
	docker compose down
site:
	python -m http.server 8080 --directory docs
