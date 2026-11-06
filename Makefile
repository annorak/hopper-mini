.PHONY: check demo

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

demo:
	@echo "Demo is not implemented yet; it arrives in M5.3."
	@exit 2
