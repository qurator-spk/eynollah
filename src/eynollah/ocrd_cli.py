from .processor import EynollahProcessor
from click import command
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

@command()
@ocrd_cli_options
def main(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahProcessor, *args, **kwargs)

if __name__ == '__main__':
    main()
