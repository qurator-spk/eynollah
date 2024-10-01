EYNOLLAH_MODELS ?= $(PWD)/models_eynollah
export EYNOLLAH_MODELS

# DOCKER_BASE_IMAGE = artefakt.dev.sbb.berlin:5000/sbb/ocrd_core:v2.68.0
DOCKER_BASE_IMAGE = docker.io/ocrd/core:v2.68.0
DOCKER_TAG = ocrd/eynollah


# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    models       Download and extract models to $(PWD)/models_eynollah"
	@echo "    install      Install with pip"
	@echo "    install-dev  Install editable with pip"
	@echo "    test         Run unit tests"
	@echo ""
	@echo "  Variables"
	@echo ""

# END-EVAL


# Download and extract models to $(PWD)/models_eynollah
models: models_eynollah

models_eynollah: models_eynollah.tar.gz
	tar xf models_eynollah.tar.gz

models_eynollah.tar.gz:
	# wget 'https://qurator-data.de/eynollah/2021-04-25/models_eynollah.tar.gz'
	# wget 'https://qurator-data.de/eynollah/2022-04-05/models_eynollah_renamed.tar.gz'
	# wget 'https://qurator-data.de/eynollah/2022-04-05/models_eynollah_renamed_savedmodel.tar.gz'
	# wget 'https://github.com/qurator-spk/eynollah/releases/download/v0.3.0/models_eynollah.tar.gz'
	wget 'https://github.com/qurator-spk/eynollah/releases/download/v0.3.1/models_eynollah.tar.gz'

# Install with pip
install:
	pip install .

# Install editable with pip
install-dev:
	pip install -e .

smoke-test:
	eynollah -i tests/resources/kant_aufklaerung_1784_0020.tif -o . -m $(PWD)/models_eynollah

# Run unit tests
test:
	pytest tests

# Build docker image
docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

