from click import command
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .processor_layout import EynollahProcessor

@command()
@ocrd_cli_options
def main(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahProcessor, *args, **kwargs)
