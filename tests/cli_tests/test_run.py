import pytest
from PIL import Image
from eynollah.cli import (
    layout as layout_cli,
    binarization as binarization_cli,
    enhancement as enhancement_cli,
)
from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES as NS

