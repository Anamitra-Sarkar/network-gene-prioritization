.PHONY: test lint api dev frontend

test:
	pytest -v

lint:
	python -m py_compile data_pipeline/*.py backend/app/**/*.py 2>&1 | head -n 50

api:
	uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev
