# pylint: disable=unused-import

from os.path import dirname, realpath
from os import chdir
import sys
import logging
import io
import collections
from unittest import TestCase as VanillaTestCase, skip, main as unittests_main
import pytest
from ocrd_utils import disableLogging, initLogging


def main(fn=None):
    if fn:
        sys.exit(pytest.main([fn]))
    else:
        unittests_main()


class TestCase(VanillaTestCase):

    @classmethod
    def setUpClass(cls):
        chdir(dirname(realpath(__file__)) + '/..')

    def setUp(self):
        disableLogging()
        initLogging()


class CapturingTestCase(TestCase):
    """
    A TestCase that needs to capture stderr/stdout and invoke click CLI.
    """

    @pytest.fixture(autouse=True)
    def _setup_pytest_capfd(self, capfd):
        self.capfd = capfd

    def invoke_cli(self, cli, args):
        """
        Substitution for click.CliRunner.invooke that works together nicely
        with unittests/pytest capturing stdout/stderr.
        """
        self.capture_out_err()  # XXX snapshot just before executing the CLI
        code = 0
        sys.argv[1:] = args  # XXX necessary because sys.argv reflects pytest args not cli args
        try:
            cli.main(args=args)
        except SystemExit as e:
            code = e.code
        out, err = self.capture_out_err()
        return code, out, err

    def capture_out_err(self):
        return self.capfd.readouterr()
