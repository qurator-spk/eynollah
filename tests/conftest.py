from glob import glob
import os
import pytest
from pathlib import Path


@pytest.fixture()
def tests_dir():
    return Path(__file__).parent.resolve()

@pytest.fixture()
def model_dir(tests_dir):
    return os.environ.get('EYNOLLAH_MODELS_DIR', str(tests_dir.joinpath('..').resolve()))

@pytest.fixture()
def resources_dir(tests_dir):
    return tests_dir / 'resources/2files'

@pytest.fixture()
def image_resources(resources_dir):
    return [Path(x) for x in glob(str(resources_dir / '*.tif'))]

@pytest.fixture()
def eynollah_log_filter():
    return lambda logrec: logrec.name.startswith('eynollah')

@pytest.fixture
def eynollah_subcommands():
    return [
        'binarization',
        'layout',
        'ocr',
        'enhancement',
        'machine-based-reading-order',
        'models',
    ]

