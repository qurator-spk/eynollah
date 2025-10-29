from typing import List
import pytest
import logging

from click.testing import CliRunner, Result
from eynollah.cli import main as eynollah_cli


@pytest.fixture
def run_eynollah_ok_and_check_logs(
    pytestconfig,
    caplog,
    model_dir,
    eynollah_subcommands,
    eynollah_log_filter,
):
    """
    Generates a Click Runner for `cli`, injects model_path and logging level
    to `args`, runs the command and checks whether the logs generated contain
    every fragment in `expected_logs`
    """

    def _run_click_ok_logs(
        subcommand: 'str',
        args: List[str],
        expected_logs: List[str],
    ) -> Result:
        assert  subcommand in eynollah_subcommands, f'subcommand {subcommand} must be one of {eynollah_subcommands}'
        args = [
                '-m', model_dir,
                subcommand,
                *args
        ]
        if pytestconfig.getoption('verbose') > 0:
            args = ['-l', 'DEBUG'] + args
        caplog.set_level(logging.INFO)
        runner = CliRunner()
        with caplog.filtering(eynollah_log_filter):
            result = runner.invoke(eynollah_cli, args, catch_exceptions=False)
        assert result.exit_code == 0, result.stdout
        if expected_logs:
            logmsgs = [logrec.message for logrec in caplog.records]
            assert any(logmsg.startswith(needle) for needle in expected_logs for logmsg in logmsgs), f'{expected_logs} not in {logmsgs}'
        return result

    return _run_click_ok_logs

