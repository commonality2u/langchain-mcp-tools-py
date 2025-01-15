# NOTES: 
# - The command lines (recipe lines) must start with a TAB character.
# - Each command line runs in a separate shell without .ONESHELL:
.PHONY: clean install build pkg-check publish test run-example
.ONESHELL:

.venv:
	uv venv

install: .venv
	uv pip install .

build: clean install
	uv build
	@echo
	uvx twine check dist/*

check-pkg:
	uv pip show -f langchain-mcp-tools

publish: build
	# set PYPI_API_KEY from .env
	$(eval export $(shell grep '^PYPI_API_KEY=' .env ))

	# check if PYPI_API_KEY is set
	@if [ -z "$$PYPI_API_KEY" ]; then \
		echo "Error: PYPI_API_KEY environment variable is not set"; \
		exit 1; \
	fi

	uvx twine upload \
		--repository-url https://upload.pypi.org/legacy/ dist/* \
		--password ${PYPI_API_KEY}

test:
	python -m pytest tests/ -v

run-example: install
	uv run examples/example.py

clean:
	@echo "The following files will be removed:"
	@git clean -fdxn -e .env
	@echo "\nAre you sure you want to proceed? [Y/n] "
	@read ans; \
	if [ "$${ans:-Y}" = "Y" ]; then \
		echo "Proceeding with clean..."; \
		git clean -fdx -e .env; \
	else \
		echo "Clean cancelled."; \
		exit 1; \
	fi
