.PHONY: lint cerberus

lint:
	uv run ruff check --fix 
	uv run ruff format
	uv run pyright src/cerberus_dpl

cerberus:
	uv run cerberus --help