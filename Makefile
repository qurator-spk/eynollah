EYNOLLAH_MODELS ?= $(PWD)/models_eynollah
export EYNOLLAH_MODELS

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
	# tar xf models_eynollah_renamed.tar.gz --transform 's/models_eynollah_renamed/models_eynollah/'
	# tar xf models_eynollah_renamed.tar.gz
	tar xf models_eynollah_renamed_savedmodel.tar.gz --transform 's/models_eynollah_renamed_savedmodel/models_eynollah/'

models_eynollah.tar.gz:
	# wget 'https://qurator-data.de/eynollah/2021-04-25/models_eynollah.tar.gz'
	# wget 'https://qurator-data.de/eynollah/2022-04-05/models_eynollah_renamed.tar.gz'
	# wget 'https://ocr-d.kba.cloud/2022-04-05.SavedModel.tar.gz'
	wget 'https://qurator-data.de/eynollah/2022-04-05/models_eynollah_renamed_savedmodel.tar.gz'

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
