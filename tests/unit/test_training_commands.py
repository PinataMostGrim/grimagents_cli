from argparse import Namespace

from grimagents.training_commands import TrainingWrapperArguments
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
}


def test_create_training_command():
    """Test for creating TrainingWrapperArguments with default configuration."""

    command = TrainingWrapperArguments(TEST_CONFIG)
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
    command = TrainingWrapperArguments(test_config)
    assert '--no-graphics' in command.get_command()

    # '--no-graphics' should not be present
    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--no-graphics": False,
    }
    command = TrainingWrapperArguments(test_config)
    assert '--no-graphics' not in command.get_command()


def test_training_command_excludes_timestamp():
    """Test for ensuring TrainingWrapperArguments excludes the '--timestamp' argument."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--timestamp": True,
    }
    command = TrainingWrapperArguments(test_config)
    assert '--timestamp' not in command.get_command()

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--timestamp": False,
    }
    command = TrainingWrapperArguments(test_config)
    assert '--timestamp' not in command.get_command()


def test_training_command_add_additional_args():
    """Test for TrainingWrapperArguments correctly setting additional arguments."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--run-id": "3DBall",
    }
    command = TrainingWrapperArguments(test_config)
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


def test_override_configuration_values():
    """Test for correct handling of TrainingWrapperArguments argument overrides.

    Ensures the following arguments are overridden:
        --trainer-config
        --env
        --lesson
        --run-id
        --num-envs
        --no-graphics
        --timestamp
    """

    args = Namespace(
        configuration_file='config\\3DBall_grimagents.json',
        new_window=False,
        trainer_config='config\\PushBlock_grimagents.json',
        env='builds\\PushBlock\\PushBlock.exe',
        lesson=2,
        run_id='PushBlock',
        base_port=5010,
        num_envs=2,
        inference=False,
        graphics=None,
        no_graphics=True,
        timestamp=None,
        no_timestamp=True,
        args=['--load', '--slow'],
    )

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--timestamp": True,
    }

    training_command = TrainingWrapperArguments(test_config)
    training_command.set_additional_arguments(args.args)
    training_command.apply_argument_overrides(args)

    result = training_command.get_command()
    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\PushBlock_grimagents.json',
        '--run-id',
        'PushBlock',
        '--env',
        'builds\\PushBlock\\PushBlock.exe',
        '--load',
        '--slow',
        '--lesson',
        '2',
        '--base-port',
        5010,
        '--num-envs',
        '2',
        '--no-graphics',
        '--train',
    ]


def test_training_command_set_methods():
    """Tests that TrainingWrapperArguments correctly sets argument values."""

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
    }
    command = TrainingWrapperArguments(test_config)

    command.set_env('builds\\3DBall\\3DBall.exe')
    command.set_lesson('3')
    command.set_run_id('ball')
    command.set_num_envs('4')
    command.set_no_graphics_enabled(True)
    command.set_timestamp_enabled(True)

    command_string = command.get_command_as_string()
    assert '--env builds\\3DBall\\3DBall.exe' in command_string
    assert '--lesson 3' in command_string
    assert '--run-id ball' in command_string
    assert '--num-envs 4' in command_string
    assert '--no-graphics' in command_string
    assert '--run-id ball-' in command_string


def test_training_command_timestamp(monkeypatch):
    """Test for TrainingWrapperArguments correctly applying a timestamp."""

    def mock_return():
        return '2019-06-29_17-13-41'

    monkeypatch.setattr(grimagents.common, "get_timestamp", mock_return)

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--run-id": "3DBall",
        "--timestamp": True,
    }

    command = TrainingWrapperArguments(test_config)
    command_string = command.get_command_as_string()

    assert '--run-id 3DBall-2019-06-29_17-13-41' in command_string


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
    command = TrainingWrapperArguments(test_config)
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
    command = TrainingWrapperArguments(test_config)
    additional_args = ['--slow']
    command.set_additional_arguments(additional_args)

    result = command.get_command()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == command_list
