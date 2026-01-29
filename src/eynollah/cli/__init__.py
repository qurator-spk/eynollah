# NOTE: For predictable order of imports of torch/shapely/tensorflow
#       this must be the first import of the CLI!
from ..eynollah_imports import imported_libs

from .cli_models import models_cli
from .cli_binarize import binarize_cli

from .cli import main
from .cli_binarize import binarize_cli
from .cli_enhance import enhance_cli
from .cli_extract_images import extract_images_cli
from .cli_layout import layout_cli
from .cli_ocr import ocr_cli
from .cli_readingorder import readingorder_cli

main.add_command(binarize_cli, 'binarization')
main.add_command(enhance_cli, 'enhancement')
main.add_command(layout_cli, 'layout')
main.add_command(readingorder_cli, 'machine-based-reading-order')
main.add_command(models_cli, 'models')
main.add_command(ocr_cli, 'ocr')
main.add_command(extract_images_cli, 'extract-images')
