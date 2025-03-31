PYTHON ?= python3
PIP ?= pip3

# DOCKER_BASE_IMAGE = artefakt.dev.sbb.berlin:5000/sbb/ocrd_core:v2.68.0
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-tf2:v3.3.0
DOCKER_TAG = ocrd/eynollah

#MODEL := 'https://qurator-data.de/eynollah/2021-04-25/models_eynollah.tar.gz'
#MODEL := 'https://qurator-data.de/eynollah/2022-04-05/models_eynollah_renamed.tar.gz'
MODEL := 'https://qurator-data.de/eynollah/2022-04-05/models_eynollah.tar.gz'
#MODEL := 'https://github.com/qurator-spk/eynollah/releases/download/v0.3.0/models_eynollah.tar.gz'
#MODEL := 'https://github.com/qurator-spk/eynollah/releases/download/v0.3.1/models_eynollah.tar.gz'

PYTEST_ARGS ?= 

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    docker       Build Docker image"
	@echo "    build        Build Python source and binary distribution"
	@echo "    install      Install package with pip"
	@echo "    install-dev  Install editable with pip"
	@echo "    deps-test    Install test dependencies with pip"
	@echo "    models       Download and extract models to $(CURDIR)/models_eynollah"
	@echo "    smoke-test   Run simple CLI check"
	@echo "    test         Run unit tests"
	@echo ""
	@echo "  Variables"
	@echo "    DOCKER_TAG   Docker image tag for 'docker' [$(DOCKER_TAG)]"
	@echo "    PYTEST_ARGS  pytest args for 'test' (Set to '-s' to see log output during test execution, '-vv' to see individual tests. [$(PYTEST_ARGS)]"
	@echo "    MODEL        URL of 'models' archive to download for 'test' [$(MODEL)]"
	@echo ""

# END-EVAL


# Download and extract models to $(PWD)/models_eynollah
models: models_eynollah

models_eynollah: models_eynollah.tar.gz
	tar zxf models_eynollah.tar.gz

models_eynollah.tar.gz:
	wget $(MODEL)

build:
	$(PIP) install build
	$(PYTHON) -m build .

# Install with pip
install:
	$(PIP) install .

# Install editable with pip
install-dev:
	$(PIP) install -e .

deps-test:
	$(PIP) install -r requirements-test.txt

smoke-test: deps-test
	eynollah layout -i tests/resources/kant_aufklaerung_1784_0020.tif -o . -m $(CURDIR)/models_eynollah

# Run unit tests
test: deps-test
	EYNOLLAH_MODELS=$(CURDIR)/models_eynollah $(PYTHON) -m pytest tests  --durations=0 --continue-on-collection-errors $(PYTEST_ARGS)

# Build docker image
docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

.PHONY: models build install install-dev test smoke-test docker help
