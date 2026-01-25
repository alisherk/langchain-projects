.PHONY: init-db dev devworker redis help

help:
	@echo "Available commands:"
	@echo "  make init-db    - Initialize the database"
	@echo "  make dev        - Run the Flask development server"
	@echo "  make devworker  - Run the Celery worker"
	@echo "  make redis      - Run Redis server"

init-db:
	cd src/pdf && ../../.venv/bin/flask --app app.web init-db

dev:
	cd src/pdf && APP_ENV=development ../../.venv/bin/flask --app app.web run --debug --port 8000

devworker:
	cd src/pdf && APP_ENV=development ../../.venv/bin/watchmedo auto-restart --directory=./app --pattern=*.py --recursive -- ../../.venv/bin/celery -A app.celery.worker worker --concurrency=1 --loglevel=INFO --pool=solo

