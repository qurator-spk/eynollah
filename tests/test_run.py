from os import environ
from pathlib import Path
from eynollah.cli import layout as eynollah_cli
from click.testing import CliRunner

testdir = Path(__file__).parent.resolve()

EYNOLLAH_MODELS = environ.get('EYNOLLAH_MODELS', str(testdir.joinpath('..', 'models_eynollah').resolve()))

def test_full_run(tmpdir, subtests, pytestconfig):
    args = [
        '-m', EYNOLLAH_MODELS,
        '-i', str(testdir.joinpath('resources/kant_aufklaerung_1784_0020.tif')),
        '-o', tmpdir,
        # subtests write to same location
        '--overwrite',
    ]
    if pytestconfig.getoption('verbose') > 0:
        args.extend(['-l', 'DEBUG'])
    runner = CliRunner()
    for options in [
            [], # defaults
            ["--allow_scaling", "--curved-line"],
            ["--allow_scaling", "--curved-line", "--full-layout"],
            ["--allow_scaling", "--curved-line", "--full-layout", "--reading_order_machine_based"],
            ["--allow_scaling", "--curved-line", "--full-layout", "--reading_order_machine_based",
             "--textline_light", "--light_version"],
            # -ep ...
            # -eoi ...
            # --do_ocr
            # --skip_layout_and_reading_order
    ]:
        with subtests.test(#msg="test CLI",
                           options=options):
            result = runner.invoke(eynollah_cli, args + options)
            print(result)
            print(result.output)
            assert result.exit_code == 0
            assert 'kant_aufklaerung_1784_0020.tif' in result.output
