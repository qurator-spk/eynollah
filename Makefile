PYTHON ?= python3
PIP ?= pip3
EXTRAS ?=

# DOCKER_BASE_IMAGE = artefakt.dev.sbb.berlin:5000/sbb/ocrd_core:v2.68.0
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-tf2:v3.3.0
DOCKER_TAG = ocrd/eynollah

#SEG_MODEL := https://qurator-data.de/eynollah/2021-04-25/models_eynollah.tar.gz
#SEG_MODEL := https://qurator-data.de/eynollah/2022-04-05/models_eynollah_renamed.tar.gz
SEG_MODEL := https://qurator-data.de/eynollah/2022-04-05/models_eynollah.tar.gz
#SEG_MODEL := https://github.com/qurator-spk/eynollah/releases/download/v0.3.0/models_eynollah.tar.gz
#SEG_MODEL := https://github.com/qurator-spk/eynollah/releases/download/v0.3.1/models_eynollah.tar.gz

BIN_MODEL := https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2021_03_09.zip

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
	@echo "    ocrd-test    Run OCR-D CLI check"
	@echo "    test         Run unit tests"
	@echo ""
	@echo "  Variables"
	@echo "    EXTRAS       comma-separated list of features (like 'OCR,plotting') for 'install' [$(EXTRAS)]"
	@echo "    DOCKER_TAG   Docker image tag for 'docker' [$(DOCKER_TAG)]"
	@echo "    PYTEST_ARGS  pytest args for 'test' (Set to '-s' to see log output during test execution, '-vv' to see individual tests. [$(PYTEST_ARGS)]"
	@echo "    SEG_MODEL    URL of 'models' archive to download for segmentation 'test' [$(SEG_MODEL)]"
	@echo "    BIN_MODEL    URL of 'models' archive to download for binarization 'test' [$(BIN_MODEL)]"
	@echo ""

# END-EVAL


# Download and extract models to $(PWD)/models_eynollah
models: models_eynollah default-2021-03-09

models_eynollah: models_eynollah.tar.gz
	tar zxf models_eynollah.tar.gz

models_eynollah.tar.gz:
	wget $(SEG_MODEL)

default-2021-03-09: $(notdir $(BIN_MODEL))
	unzip $(notdir $(BIN_MODEL))
	mkdir $@
	mv $(basename $(notdir $(BIN_MODEL))) $@

$(notdir $(BIN_MODEL)):
	wget $(BIN_MODEL)

build:
	$(PIP) install build
	$(PYTHON) -m build .

# Install with pip
install:
	$(PIP) install .$(and $(EXTRAS),[$(EXTRAS)])

# Install editable with pip
install-dev:
	$(PIP) install -e .$(and $(EXTRAS),[$(EXTRAS)])

deps-test: models_eynollah
	$(PIP) install -r requirements-test.txt

smoke-test: TMPDIR != mktemp -d
smoke-test: tests/resources/kant_aufklaerung_1784_0020.tif
	# layout analysis:
	eynollah layout -i $< -o $(TMPDIR) -m $(CURDIR)/models_eynollah
	fgrep -q http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 $(TMPDIR)/$(basename $(<F)).xml
	fgrep -c -e TextRegion -e ImageRegion -e SeparatorRegion $(TMPDIR)/$(basename $(<F)).xml
	# directory mode (skip one, add one):
	eynollah layout -di $(<D) -o $(TMPDIR) -m $(CURDIR)/models_eynollah
	test -s $(TMPDIR)/euler_rechenkunst01_1738_0025.xml
	# binarize:
	eynollah binarization -m $(CURDIR)/default-2021-03-09 $< $(TMPDIR)/$(<F)
	test -s $(TMPDIR)/$(<F)
	@set -x; test "$$(identify -format '%w %h' $<)" = "$$(identify -format '%w %h' $(TMPDIR)/$(<F))"
	$(RM) -r $(TMPDIR)

ocrd-test: TMPDIR != mktemp -d
ocrd-test: tests/resources/kant_aufklaerung_1784_0020.tif
	cp $< $(TMPDIR)
	ocrd workspace -d $(TMPDIR) init
	ocrd workspace -d $(TMPDIR) add -G OCR-D-IMG -g PHYS_0020 -i OCR-D-IMG_0020 $(<F)
	ocrd-eynollah-segment -w $(TMPDIR) -I OCR-D-IMG -O OCR-D-SEG -P models $(CURDIR)/models_eynollah
	result=$$(ocrd workspace -d $(TMPDIR) find -G OCR-D-SEG); \
	fgrep -q http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 $(TMPDIR)/$$result && \
	fgrep -c -e TextRegion -e ImageRegion -e SeparatorRegion $(TMPDIR)/$$result
	ocrd-sbb-binarize -w $(TMPDIR) -I OCR-D-IMG -O OCR-D-BIN -P model $(CURDIR)/default-2021-03-09
	ocrd-sbb-binarize -w $(TMPDIR) -I OCR-D-SEG -O OCR-D-SEG-BIN -P model $(CURDIR)/default-2021-03-09 -P operation_level region
	$(RM) -r $(TMPDIR)

# Run unit tests
test:
	EYNOLLAH_MODELS=$(CURDIR)/models_eynollah $(PYTHON) -m pytest tests  --durations=0 --continue-on-collection-errors $(PYTEST_ARGS)

# Build docker image
docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

.PHONY: models build install install-dev test smoke-test ocrd-test docker help
