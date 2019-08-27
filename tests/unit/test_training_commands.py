import pytest

from argparse import Namespace

import grimagents.common

from grimagents.training_commands import TrainingWrapperArguments


@pytest.fixture
def test_config():
    return {
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


def test_create_training_arguments(test_config):
    """Test for creating TrainingWrapperArguments with default configuration."""

    arguments = TrainingWrapperArguments(test_config)

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == [
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


def test_training_arguments_handles_no_graphics(test_config):
    """Test for correct handling of the --no-graphics argument."""

    # '--no-graphics' should be present
    test_config['--no-graphics'] = True
    arguments = TrainingWrapperArguments(test_config)
    assert '--no-graphics' in arguments.get_arguments()

    # '--no-graphics' should not be present
    test_config['--no-graphics'] = False
    arguments = TrainingWrapperArguments(test_config)
    assert '--no-graphics' not in arguments.get_arguments()


def test_training_arguments_excludes_timestamp(test_config):
    """Test for ensuring TrainingWrapperArguments excludes the '--timestamp' argument."""

    test_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(test_config)
    assert '--timestamp' not in arguments.get_arguments()

    test_config['--timestamp'] = False
    arguments = TrainingWrapperArguments(test_config)
    assert '--timestamp' not in arguments.get_arguments()


def test_training_arguments_add_additional_args(test_config):
    """Test for TrainingWrapperArguments correctly setting additional arguments."""

    arguments = TrainingWrapperArguments(test_config)
    additional_args = ['--slow', '--load']
    arguments.set_additional_arguments(additional_args)

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == ['pipenv', 'run', 'python', 'grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--env', 'builds\\3DBall\\Unity Environment.exe', '--export-path', 'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels', '--run-id', '3DBall', '--slow', '--load', '--train']


def test_override_configuration_values(test_config):
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

    test_config['--timestamp'] = True

    arguments = TrainingWrapperArguments(test_config)
    arguments.set_additional_arguments(args.args)
    arguments.apply_argument_overrides(args)

    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == ['pipenv', 'run', 'python', 'grimagents\\training_wrapper.py', 'config\\PushBlock_grimagents.json', '--env', 'builds\\PushBlock\\PushBlock.exe', '--export-path', 'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels', '--lesson', '2', '--run-id', 'PushBlock', '--base-port', 5010, '--num-envs', '2', '--no-graphics', '--load', '--slow', '--train']


def test_training_arguments_set_methods(test_config):
    """Tests that TrainingWrapperArguments correctly sets argument values."""

    arguments = TrainingWrapperArguments(test_config)

    arguments.set_env('builds\\3DBall\\3DBall.exe')
    arguments.set_lesson('3')
    arguments.set_run_id('ball')
    arguments.set_num_envs('4')
    arguments.set_no_graphics_enabled(True)
    arguments.set_timestamp_enabled(True)

    arguments_string = arguments.get_arguments_as_string()
    assert '--env builds\\3DBall\\3DBall.exe' in arguments_string
    assert '--lesson 3' in arguments_string
    assert '--run-id ball' in arguments_string
    assert '--num-envs 4' in arguments_string
    assert '--no-graphics' in arguments_string
    assert '--run-id ball-' in arguments_string


def test_training_arguments_timestamp(monkeypatch, test_config):
    """Test for TrainingWrapperArguments correctly applying a timestamp."""

    def mock_return():
        return '2019-06-29_17-13-41'

    monkeypatch.setattr(grimagents.common, "get_timestamp", mock_return)

    test_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(test_config)
    arguments_string = arguments.get_arguments_as_string()

    assert '--run-id 3DBall-2019-06-29_17-13-41' in arguments_string


def test_training_arguments_inference(test_config):
    """Tests for correct processing of the '--inference' argument.

    - Ensures get_arguments() can handle no additional args being set
    - Ensures the '--train' argument is removed
    - Ensures the '--export-path' argument is removed
    - Ensures the '--slow' argument is appended and not duplicated
    """

    test_config['--inference'] = True

    # --train is removed, --slow is added, and no exceptions are caused by additional args not being set
    arguments = TrainingWrapperArguments(test_config)
    arguments_list = ['pipenv', 'run', 'python', 'grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--env', 'builds\\3DBall\\Unity Environment.exe', '--run-id', '3DBall', '--slow']

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == arguments_list

    # '--slow' isn't duplicated
    arguments = TrainingWrapperArguments(test_config)
    additional_args = ['--slow']
    arguments.set_additional_arguments(additional_args)

    result = arguments.get_arguments()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == arguments_list
