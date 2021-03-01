from os import environ
from pathlib import Path
from ocrd_utils import pushd_popd
from tests.base import CapturingTestCase as TestCase, main
from qurator.eynollah.cli import main as eynollah_cli

testdir = Path(__file__).parent.resolve()

EYNOLLAH_MODELS = environ.get('EYNOLLAH_MODELS', str(testdir.joinpath('..', 'models_eynollah').resolve()))

class TestEynollahRun(TestCase):

    def test_full_run(self):
        with pushd_popd(tempdir=True) as tempdir:
            code, out, err = self.invoke_cli(eynollah_cli, [
                '-m', EYNOLLAH_MODELS,
                '-i', str(testdir.joinpath('resources/kant_aufklaerung_1784_0020.tif')),
                '-o', tempdir
            ])
            print(code, out, err)
            assert not code

if __name__ == '__main__':
    main(__file__)
