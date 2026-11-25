.PHONY: setup hooks format lint typecheck test security secrets architecture config verify

setup:
	uv python install 3.12
	uv sync --locked --python 3.12
	uv run pre-commit install

hooks:
	uv run pre-commit install

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

secrets:
	uv run python scripts/verify.py secrets # pragma: allowlist secret

architecture:
	uv run python scripts/verify.py architecture

config:
	uv run python scripts/verify.py config

verify:
	uv run python scripts/verify.py verify
