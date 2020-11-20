# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install      Install with pip"
	@echo "    install-dev  Install editable with pip"
	@echo ""
	@echo "  Variables"
	@echo ""

# END-EVAL

# Install with pip
install:
	pip install .

# Install editable with pip
install-dev:
	pip install -e .
