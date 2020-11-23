# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    models       Download and extract models to $(PWD)/models_eynollah"
	@echo "    install      Install with pip"
	@echo "    install-dev  Install editable with pip"
	@echo ""
	@echo "  Variables"
	@echo ""

# END-EVAL


# Download and extract models to $(PWD)/models_eynollah
models: models_eynollah

models_eynollah: models_eynollah.tar.gz
	tar xf models_eynollah.tar.gz

models_eynollah.tar.gz:
	wget 'https://qurator-data.de/eynollah/models_eynollah.tar.gz'

# Install with pip
install:
	pip install .

# Install editable with pip
install-dev:
	pip install -e .

smoke-test:
	eynollah -i tests/resources/kant_aufklaerung_1784_0020.tif -o . -m $(PWD)/models_eynollah
