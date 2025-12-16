.PHONY: init-db dev devworker redis help

help:
	@echo "Available commands:"
	@echo "  make init-db    - Initialize the database"
	@echo "  make dev        - Run the Flask development server"
	@echo "  make devworker  - Run the Celery worker"
	@echo "  make redis      - Run Redis server"

init-db:
	flask --app src.pdf.app.web init-db

dev:
	inv dev

devworker:
	inv devworker

redis:
	redis-server
