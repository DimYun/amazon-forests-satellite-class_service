APP_PORT := 5039
DOCKER_TAG := latest
DOCKER_IMAGE := planet

.PHONY: run_app
run_app:
	python3 -m uvicorn app:create_app --host='0.0.0.0' --port=$(APP_PORT)

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: run_unit_tests
run_unit_tests:
	PYTHONPATH=. pytest tests/unit/

.PHONY: run_integration_tests
run_integration_tests:
	PYTHONPATH=. pytest tests/integration/

.PHONY: run_all_tests
run_all_tests:
	make run_unit_tests
	make run_integration_tests

.PHONY: generate_coverage_report
generate_coverage_report:
	PYTHONPATH=. pytest --cov=src --cov-report html  tests/

.PHONY: lint
lint:
	PYTHONPATH=. flake8 src app.py
	PYTHONPATH=. black src app.py
	PYTHONPATH=. isort src app.py

.PHONY: build
build:
	docker build -f Dockerfile . -t $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker_run
docker_run:
	docker run -p 5039:5039 -d $(DOCKER_IMAGE):$(DOCKER_TAG)
