from os import environ
from pathlib import Path
import logging
from PIL import Image
from eynollah.cli import layout as layout_cli, binarization as binarization_cli
from click.testing import CliRunner
from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES as NS

testdir = Path(__file__).parent.resolve()

EYNOLLAH_MODELS = environ.get('EYNOLLAH_MODELS', str(testdir.joinpath('..', 'models_eynollah').resolve()))
SBBBIN_MODELS = environ.get('SBBBIN_MODELS', str(testdir.joinpath('..', 'default-2021-03-09').resolve()))

def test_run_eynollah_layout_filename(tmp_path, subtests, pytestconfig, caplog):
    infile = testdir.joinpath('resources/kant_aufklaerung_1784_0020.tif')
    outfile = tmp_path / 'kant_aufklaerung_1784_0020.xml'
    args = [
        '-m', EYNOLLAH_MODELS,
        '-i', str(infile),
        '-o', str(outfile.parent),
        # subtests write to same location
        '--overwrite',
    ]
    if pytestconfig.getoption('verbose') > 0:
        args.extend(['-l', 'DEBUG'])
    caplog.set_level(logging.INFO)
    def only_eynollah(logrec):
        return logrec.name == 'eynollah'
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
            with caplog.filtering(only_eynollah):
                result = runner.invoke(layout_cli, args + options, catch_exceptions=False)
            print(result)
            assert result.exit_code == 0
            logmsgs = [logrec.message for logrec in caplog.records]
            assert str(infile) in logmsgs
            assert outfile.exists()
            tree = page_from_file(str(outfile)).etree
            regions = tree.xpath("//page:TextRegion", namespaces=NS)
            assert len(regions) >= 2, "result is inaccurate"
            regions = tree.xpath("//page:SeparatorRegion", namespaces=NS)
            assert len(regions) >= 2, "result is inaccurate"
            lines = tree.xpath("//page:TextLine", namespaces=NS)
            assert len(lines) == 31, "result is inaccurate" # 29 paragraph lines, 1 page and 1 catch-word line

def test_run_eynollah_layout_directory(tmp_path, pytestconfig, caplog):
    indir = testdir.joinpath('resources')
    outdir = tmp_path
    args = [
        '-m', EYNOLLAH_MODELS,
        '-di', str(indir),
        '-o', str(outdir),
    ]
    if pytestconfig.getoption('verbose') > 0:
        args.extend(['-l', 'DEBUG'])
    caplog.set_level(logging.INFO)
    def only_eynollah(logrec):
        return logrec.name == 'eynollah'
    runner = CliRunner()
    with caplog.filtering(only_eynollah):
        result = runner.invoke(layout_cli, args)
    print(result)
    assert result.exit_code == 0
    logmsgs = [logrec.message for logrec in caplog.records]
    assert len([logmsg for logmsg in logmsgs if logmsg.startswith('Job done in')]) == 2
    assert any(logmsg for logmsg in logmsgs if logmsg.startswith('All jobs done in'))
    assert len(list(outdir.iterdir())) == 2

def test_run_eynollah_binarization_filename(tmp_path, subtests, pytestconfig, caplog):
    infile = testdir.joinpath('resources/kant_aufklaerung_1784_0020.tif')
    outfile = tmp_path.joinpath('kant_aufklaerung_1784_0020.png')
    args = [
        '-m', SBBBIN_MODELS,
        str(infile),
        str(outfile),
    ]
    caplog.set_level(logging.INFO)
    def only_eynollah(logrec):
        return logrec.name == 'SbbBinarizer'
    runner = CliRunner()
    for options in [
            [], # defaults
            ["--no-patches"],
    ]:
        with subtests.test(#msg="test CLI",
                           options=options):
            with caplog.filtering(only_eynollah):
                result = runner.invoke(binarization_cli, args + options)
            print(result)
            assert result.exit_code == 0
            logmsgs = [logrec.message for logrec in caplog.records]
            assert any(True for logmsg in logmsgs if logmsg.startswith('Predicting'))
            assert outfile.exists()
            with Image.open(infile) as original_img:
                original_size = original_img.size
            with Image.open(outfile) as binarized_img:
                binarized_size = binarized_img.size
            assert original_size == binarized_size

def test_run_eynollah_binarization_directory(tmp_path, subtests, pytestconfig, caplog):
    indir = testdir.joinpath('resources')
    outdir = tmp_path
    args = [
        '-m', SBBBIN_MODELS,
        '-di', str(indir),
        '-do', str(outdir),
    ]
    caplog.set_level(logging.INFO)
    def only_eynollah(logrec):
        return logrec.name == 'SbbBinarizer'
    runner = CliRunner()
    with caplog.filtering(only_eynollah):
        result = runner.invoke(binarization_cli, args)
    print(result)
    assert result.exit_code == 0
    logmsgs = [logrec.message for logrec in caplog.records]
    assert len([logmsg for logmsg in logmsgs if logmsg.startswith('Predicting')]) == 2
    assert len(list(outdir.iterdir())) == 2
