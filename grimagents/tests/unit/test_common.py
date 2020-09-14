import datetime
import subprocess

import grimagents.common as common


def test_is_pipenv_present(monkeypatch):
    """Tests for determining if a virtual environment is accessible through Pipenv."""

    class MockProcess:
        def __init__(self, returncode):
            self.returncode = returncode

    process = MockProcess(0)

    def mock_run(command, universal_newlines=True, stdout='', stderr=''):
        return process

    monkeypatch.setattr(subprocess, 'run', mock_run)

    # A virtual environment is accessible (subprocess.run() results in a process with a return code of 0)
    assert common.is_pipenv_present() is True

    # A virtual environment is not accessible (subprocess.run() results in a process with a return code of something other than 0)
    process = MockProcess(1)
    assert common.is_pipenv_present() is False

    # Pipenv is not accessible
    def mock_run_file_not_found(command, universal_newlines=True, stdout='', stderr=''):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, 'run', mock_run_file_not_found)

    process = MockProcess(0)
    assert common.is_pipenv_present() is False


def test_get_timestamp(monkeypatch):
    """Tests for the return of a correctly formatted timestamp."""

    mock_time = datetime.datetime(2000, 1, 1, hour=0, minute=0, second=0)

    class mockdatetime:
        @classmethod
        def now(self):
            return mock_time

    monkeypatch.setattr('grimagents.common.datetime', mockdatetime)

    timestamp = common.get_timestamp()
    assert timestamp == '2000-01-01_00-00-00'


def test_get_human_readable_duration():
    """Tests for the return of a duration of seconds in human readable terms."""

    human_readable = common.get_human_readable_duration(-1)
    assert human_readable == '0 seconds'

    human_readable = common.get_human_readable_duration(10)
    assert human_readable == '10 seconds'

    human_readable = common.get_human_readable_duration(1000)
    assert human_readable == '16 minutes, 40 seconds'

    human_readable = common.get_human_readable_duration(10000)
    assert human_readable == '2 hours, 46 minutes, 40 seconds'


def test_add_nested_dict_value():
    """Tests that values are correctly nested in a dictionary under multiple keys
    and that existing, sibling key value pairs in the dictionary not destroyed.
    """

    recursiveKeys = [
        "reward_signals.extrinsic.strength",
        "reward_signals.extrinsic.gamma",
        "reward_signals.curiosity.strength",
        "reward_signals.curiosity.gamma",
    ]

    expectedDict = {
        'reward_signals': {
            'curiosity': {'gamma': 1.0, 'strength': 1.0},
            'extrinsic': {'gamma': 1.0, 'strength': 1.0},
        }
    }

    dictionary = {}

    for key in recursiveKeys:
        common.add_nested_dict_value(dictionary, key, 1.0)

    assert dictionary == expectedDict

    dictionary = {'reward_signals': {'extrinsic': {}}}

    for key in recursiveKeys:
        common.add_nested_dict_value(dictionary, key, 1.0)

    assert dictionary == expectedDict

    dictionary = {'reward_signals': {'extrinsic': {'gamma': 0.99}}}

    for key in recursiveKeys:
        common.add_nested_dict_value(dictionary, key, 1.0)

    assert dictionary == expectedDict
