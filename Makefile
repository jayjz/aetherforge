.PHONY: dev test chaos schema

dev:
	AETHER_ENGINE=mock uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

test:
	AETHER_ENGINE=mock AETHER_CHAOS=false pytest tests/ -v

chaos:
	AETHER_ENGINE=mock AETHER_CHAOS=true python scripts/chaos_monkey.py

schema:
	python scripts/generate_openapi.py --out openapi.json