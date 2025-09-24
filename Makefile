.PHONY: lint

lint:
	uv run ruff check --fix 
	uv run ruff format
	uv run pyright src/cerberus_dpl

