from typing import List
from click import Command
import pytest
import logging

from click.testing import CliRunner, Result

@pytest.fixture
def run_eynollah_ok_and_check_logs(
    pytestconfig,
    caplog,
    model_dir,
    eynollah_log_filter,
):
    """
    Generates a Click Runner for `cli`, injects model_path and logging level
    to `args`, runs the command and checks whether the logs generated contain
    every fragment in `expected_logs`
    """

    def _run_click_ok_logs(cli: Command, args: List[str], expected_logs: List[str]) -> Result:
        args = ['-m', model_dir] + args
        if pytestconfig.getoption('verbose') > 0:
            args.extend(['-l', 'DEBUG'])
        caplog.set_level(logging.INFO)
        runner = CliRunner()
        with caplog.filtering(eynollah_log_filter):
            result = runner.invoke(cli, args, catch_exceptions=False)
        assert result.exit_code == 0, result.stdout
        if expected_logs:
            logmsgs = [logrec.message for logrec in caplog.records]
            assert any(logmsg.startswith(needle) for needle in expected_logs for logmsg in logmsgs), f'{expected_logs} not in {logmsgs}'
        return result

    return _run_click_ok_logs

