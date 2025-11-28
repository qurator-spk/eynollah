from click import command
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .processor_binarize import SbbBinarizeProcessor

@command()
@ocrd_cli_options
def main(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbBinarizeProcessor, *args, **kwargs)
