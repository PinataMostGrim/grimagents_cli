from grimagents.commands import TrainingCommand
import grimagents.common


TEST_CONFIG = {
    "trainer-config-path": "config\\3DBall.yaml",
    "--env": "builds\\3DBall\\Unity Environment.exe",
    "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
    "--curriculum": "",
    "--keep-checkpoints": "",
    "--lesson": "",
    "--run-id": "3DBall",
    "--num-runs": "",
    "--save-freq": "",
    "--seed": "",
    "--base-port": "",
    "--num-envs": "",
    "--no-graphics": False,
    "--timestamp": False,
    "--log-filename": "",
}


def test_create_training_command():
    """Test for creating TrainingCommand with default configuration."""

    command = TrainingCommand(TEST_CONFIG)
    command_list = [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\Unity Environment.exe',
        '--export-path',
        'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--run-id',
        '3DBall',
        '--train',
    ]

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = command.get_command()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == command_list


def test_training_command_handles_no_graphics():
    """Test for correct handling of the --no-graphics argument."""

    # '--no-graphics' should be present
    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--no-graphics": True,
    }
    command = TrainingCommand(test_config)
    assert '--no-graphics' in command.get_command()

    # '--no-graphics' should not be present
    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--no-graphics": False,
    }
    command = TrainingCommand(test_config)
    assert '--no-graphics' not in command.get_command()


def test_training_command_excludes_timestamp():
    """Test for ensuring TrainingWrapperArguments excludes the '--timestamp' argument."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--timestamp": True,
    }
    command = TrainingCommand(test_config)
    assert '--timestamp' not in command.get_command()

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--timestamp": False,
    }
    command = TrainingCommand(test_config)
    assert '--timestamp' not in command.get_command()


def test_training_command_add_additional_args():
    """Test for TrainingCommand correctly setting additional arguments."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--run-id": "3DBall",
    }
    command = TrainingCommand(test_config)
    additional_args = ['--slow', '--load']
    command.set_additional_arguments(additional_args)
    command_list = [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\Unity Environment.exe',
        '--run-id',
        '3DBall',
        '--slow',
        '--load',
        '--train',
    ]

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = command.get_command()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == command_list


def test_training_command_override_args():
    """Test for TrainingCommand correctly overriding argument values."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
        "--curriculum": "",
        "--keep-checkpoints": "",
        "--lesson": "",
        "--run-id": "3DBall",
        "--num-runs": "",
        "--save-freq": "",
        "--seed": "",
        "--base-port": "",
        "--num-envs": "",
        "--no-graphics": False,
        "--timestamp": False,
        "--log-filename": "",
    }
    command = TrainingCommand(test_config)

    command.set_env('builds\\3DBall\\3DBall.exe')
    command.set_lesson('3')
    command.set_run_id('ball')
    command.set_num_envs('4')
    command.set_no_graphics_enabled(True)
    command.set_log_filename('3DBall.log')
    command.set_timestamp_enabled(True)

    command_string = command.get_command_as_string()
    assert '--env builds\\3DBall\\3DBall.exe' in command_string
    assert '--lesson 3' in command_string
    assert '--run-id ball' in command_string
    assert '--num-envs 4' in command_string
    assert '--no-graphics' in command_string
    assert '--log-filename 3DBall.log' in command_string
    assert '--run-id ball-' in command_string


def test_training_command_timestamp(monkeypatch):
    """Test for TrainingCommand correctly applying a timestamp and setting
    the log_filename."""

    def mock_return():
        return '2019-06-29_17-13-41'

    monkeypatch.setattr(grimagents.common, "get_timestamp", mock_return)

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--run-id": "3DBall",
        "--timestamp": True,
    }

    # Result if a log-filename key does not already exist
    command = TrainingCommand(test_config)
    command_string = command.get_command_as_string()

    assert '--run-id 3DBall-2019-06-29_17-13-41' in command_string
    assert '--log-filename 3DBall' in command_string

    # Result if an empty log-filename key already exists
    command.set_log_filename("")
    assert '--log-filename 3DBall' in command.get_command_as_string()

    # Result if a log-filename key exists and has content
    command.set_log_filename('ball')
    assert '--log-filename ball' in command.get_command_as_string()


def test_training_command_inference():
    """Tests for correct processing of the '--inference' argument.

    - Ensures get_command() can handle no additional args being set
    - Ensures the '--train' argument is removed
    - Ensures the '--export-path' argument is removed
    - Ensures the '--slow' argument is appended and not duplicated
    """

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
        "--inference": True,
    }

    # --train is removed, --slow is added, and no exceptions are caused by additional args not being set
    command = TrainingCommand(test_config)
    command_list = [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--run-id',
        '3DBall',
        '--slow',
    ]

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = command.get_command()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == command_list

    # '--slow' isn't duplicated
    command = TrainingCommand(test_config)
    additional_args = ['--slow']
    command.set_additional_arguments(additional_args)

    result = command.get_command()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == command_list
