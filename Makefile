.PHONY: setup format lint typecheck test security architecture config verify

setup:
	uv python install 3.12
	uv sync --locked --python 3.12

format:
	uv run python scripts/verify.py format

lint:
	uv run python scripts/verify.py lint

typecheck:
	uv run python scripts/verify.py typecheck

test:
	uv run python scripts/verify.py test

security:
	uv run python scripts/verify.py security

architecture:
	uv run python scripts/verify.py architecture

config:
	uv run python scripts/verify.py config

verify:
	uv run python scripts/verify.py verify
