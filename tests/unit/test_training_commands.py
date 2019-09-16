import pytest

from argparse import Namespace

import grimagents.command_util
import grimagents.common
import grimagents.config

from grimagents.training_commands import (
    Command,
    PerformTraining,
    ResumeTraining,
    TrainingWrapperArguments,
)


@pytest.fixture
def grim_config():
    return {
        'trainer-config-path': 'config\\3DBall.yaml',
        '--env': 'builds\\3DBall\\3DBall.exe',
        '--export-path': 'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--curriculum': '',
        '--keep-checkpoints': '',
        '--lesson': '',
        '--run-id': '3DBall',
        '--num-runs': '',
        '--save-freq': '',
        '--seed': '',
        '--base-port': '',
        '--num-envs': '',
        '--no-graphics': False,
        '--timestamp': False,
    }


@pytest.fixture
def namespace_args():

    return Namespace(
        configuration_file='config\\3DBall_grimagents.json',
        new_window=False,
        dry_run=False,
        trainer_config=None,
        env=None,
        lesson=None,
        run_id='3DBall',
        base_port=None,
        num_envs=None,
        inference=None,
        graphics=None,
        no_graphics=None,
        timestamp=None,
        no_timestamp=None,
        args=[],
    )


@pytest.fixture
def training_command_arguments():
    return [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\3DBall.exe',
        '--export-path',
        'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--run-id',
        '3DBall',
        '--train',
    ]


def test_perform_training_create_command(
    monkeypatch, namespace_args, grim_config, training_command_arguments
):
    """Tests that PerformTraining.create_command() returns a valid command line argument for training."""

    def mock_load_config(config_path):
        return grim_config

    monkeypatch.setattr(grimagents.config, 'load_grim_configuration_file', mock_load_config)

    perform_training = PerformTraining(namespace_args)
    result = perform_training.create_command()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == training_command_arguments


def test_perform_training_execute(monkeypatch, namespace_args, training_command_arguments):
    """Tests that PerformTraining.execute() initiates training.

    Ensures:
        - The training command is saved to history
        - The training command is sent to command_util for execution
    """

    def mock_create_command(self):
        return training_command_arguments

    def mock_save_to_history(command):
        assert command == training_command_arguments

    def mock_execute_command(command, new_window, show_command, dry_run):
        assert command == training_command_arguments

    monkeypatch.setattr(PerformTraining, 'create_command', mock_create_command)
    monkeypatch.setattr(grimagents.command_util, 'save_to_history', mock_save_to_history)
    monkeypatch.setattr(grimagents.command_util, 'execute_command', mock_execute_command)

    perform_training = PerformTraining(namespace_args)
    perform_training.execute()


def test_perform_training_command_dry_run(monkeypatch, namespace_args, grim_config):
    """Tests for the correct creation of a PerformTraining command with dry_run enabled.
    """

    namespace_args.dry_run = True

    def mock_load_config(config_path):
        return grim_config

    def mock_execute_command(command, new_window, show_command, dry_run):
        pass

    monkeypatch.setattr(grimagents.config, 'load_grim_configuration_file', mock_load_config)
    monkeypatch.setattr(grimagents.command_util, 'execute_command', mock_execute_command)

    perform_training = PerformTraining(namespace_args)
    perform_training.execute()
    assert perform_training.dry_run is True


def test_command_dry_run(monkeypatch):
    """Tests the correct assignment of the 'dry_run' argument for the following commands:
        - Command
        - ListTrainingOptions
        - StartTensorboard
        - ResumeTraining
    """

    no_dry_run_args = Namespace(new_window=False, dry_run=False, args=[])

    dry_run_args = Namespace(new_window=False, dry_run=True, args=[])

    def mock_execute_command(command, new_window, show_command, dry_run):
        pass

    monkeypatch.setattr(grimagents.command_util, 'execute_command', mock_execute_command)

    command = Command(no_dry_run_args)
    command.execute()
    assert command.dry_run is False

    command = Command(dry_run_args)
    command.execute()
    assert command.dry_run is True


def test_resume_training(monkeypatch):
    """Tests for correct parsing of ResumeTraining command.

    - Ensures --load argument is appended
    - Ensures --lesson argument is appended, if present in args
    """

    last_history = [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\3DBall.exe',
        '--run-id',
        '3DBall_01-2019-07-12_23-55-05',
        '--train',
    ]

    def mock_load_history():
        return last_history

    monkeypatch.setattr(grimagents.command_util, 'load_last_history', mock_load_history)

    # --load argument is appended and --lesson IS NOT present
    args = Namespace(new_window=False, dry_run=False, lesson=None, args=[])
    resume_training = ResumeTraining(args)

    assert resume_training.create_command() == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\3DBall.exe',
        '--run-id',
        '3DBall_01-2019-07-12_23-55-05',
        '--train',
        '--load',
    ]

    # --load argument is appended and --lesson IS present
    args = Namespace(new_window=False, dry_run=False, lesson=3, args=[])
    resume_training = ResumeTraining(args)

    assert resume_training.create_command() == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\3DBall.exe',
        '--run-id',
        '3DBall_01-2019-07-12_23-55-05',
        '--train',
        '--load',
        '--lesson',
        '3',
    ]


def test_create_training_arguments(grim_config):
    """Test for creating TrainingWrapperArguments with default configuration."""

    arguments = TrainingWrapperArguments(grim_config)

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
        'builds\\3DBall\\3DBall.exe',
        '--export-path',
        'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--run-id',
        '3DBall',
        '--train',
    ]


def test_training_arguments_handles_no_graphics(grim_config):
    """Test for correct handling of the --no-graphics argument."""

    # '--no-graphics' should be present
    grim_config['--no-graphics'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--no-graphics' in arguments.get_arguments()

    # '--no-graphics' should not be present
    grim_config['--no-graphics'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--no-graphics' not in arguments.get_arguments()


def test_training_arguments_excludes_timestamp(grim_config):
    """Test for ensuring TrainingWrapperArguments excludes the '--timestamp' argument."""

    grim_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--timestamp' not in arguments.get_arguments()

    grim_config['--timestamp'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--timestamp' not in arguments.get_arguments()


def test_training_arguments_add_additional_args(grim_config):
    """Test for TrainingWrapperArguments correctly setting additional arguments."""

    arguments = TrainingWrapperArguments(grim_config)
    additional_args = ['--slow', '--load']
    arguments.set_additional_arguments(additional_args)

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
        'builds\\3DBall\\3DBall.exe',
        '--export-path',
        'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--run-id',
        '3DBall',
        '--slow',
        '--load',
        '--train',
    ]


def test_override_configuration_values(grim_config):
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

    grim_config['--timestamp'] = True

    arguments = TrainingWrapperArguments(grim_config)
    arguments.set_additional_arguments(args.args)
    arguments.apply_argument_overrides(args)

    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\PushBlock_grimagents.json',
        '--env',
        'builds\\PushBlock\\PushBlock.exe',
        '--export-path',
        'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--lesson',
        '2',
        '--run-id',
        'PushBlock',
        '--base-port',
        5010,
        '--num-envs',
        '2',
        '--no-graphics',
        '--load',
        '--slow',
        '--train',
    ]


def test_training_arguments_set_methods(grim_config):
    """Tests that TrainingWrapperArguments correctly sets argument values."""

    arguments = TrainingWrapperArguments(grim_config)

    arguments.set_env('builds\\3DBall\\3DBallHard.exe')
    arguments.set_lesson('3')
    arguments.set_run_id('ball')
    arguments.set_num_envs('4')
    arguments.set_no_graphics_enabled(True)
    arguments.set_timestamp_enabled(True)

    arguments_string = arguments.get_arguments_as_string()
    assert '--env builds\\3DBall\\3DBallHard.exe' in arguments_string
    assert '--lesson 3' in arguments_string
    assert '--run-id ball' in arguments_string
    assert '--num-envs 4' in arguments_string
    assert '--no-graphics' in arguments_string
    assert '--run-id ball-' in arguments_string


def test_training_arguments_timestamp(monkeypatch, grim_config):
    """Test for TrainingWrapperArguments correctly applying a timestamp."""

    def mock_return():
        return '2019-06-29_17-13-41'

    monkeypatch.setattr(grimagents.common, "get_timestamp", mock_return)

    grim_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(grim_config)
    arguments_string = arguments.get_arguments_as_string()

    assert '--run-id 3DBall-2019-06-29_17-13-41' in arguments_string


def test_training_arguments_inference(grim_config):
    """Tests for correct processing of the '--inference' argument.

    - Ensures get_arguments() can handle no additional args being set
    - Ensures the '--train' argument is removed
    - Ensures the '--export-path' argument is removed
    - Ensures the '--slow' argument is appended and not duplicated
    """

    grim_config['--inference'] = True

    # --train is removed, --slow is added, and no exceptions are caused by additional args not being set
    arguments = TrainingWrapperArguments(grim_config)
    arguments_list = [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\3DBall.exe',
        '--run-id',
        '3DBall',
        '--slow',
    ]

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == arguments_list

    # '--slow' isn't duplicated
    arguments = TrainingWrapperArguments(grim_config)
    additional_args = ['--slow']
    arguments.set_additional_arguments(additional_args)

    result = arguments.get_arguments()
    result[3] = 'grimagents\\training_wrapper.py'

    assert result == arguments_list
