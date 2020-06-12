import pytest

from argparse import Namespace

import grimagents.command_util
import grimagents.common

from grimagents.training_commands import (
    Command,
    ListTrainingOptions,
    StartTensorboard,
    PerformTraining,
)


@pytest.fixture
def grim_config():
    return {
        'trainer-config-path': 'config/3DBall.yaml',
        '--env': 'builds/3DBall/3DBall.exe',
        '--export-path': 'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
    }


@pytest.fixture
def namespace_args():
    return Namespace(
        configuration_file='config/3DBall_grimagents.json',
        resume=False,
        dry_run=False,
        trainer_config=None,
        env=None,
        sampler=None,
        lesson=None,
        run_id='3DBall',
        base_port=None,
        num_envs=None,
        inference=None,
        graphics=None,
        no_graphics=None,
        timestamp=None,
        no_timestamp=None,
        multi_gpu=None,
        no_multi_gpu=None,
        env_args=None,
        additional_args=[],
    )


@pytest.fixture
def training_command_arguments():
    return [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--env',
        'builds/3DBall/3DBall.exe',
        '--export-path',
        'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--run-id',
        '3DBall',
    ]


def test_list_training_options(namespace_args):
    """Tests ListTrainingOptions.create_command() for the creation of a valid command."""

    namespace_args.list = True
    list_options = ListTrainingOptions(namespace_args)

    assert list_options.create_command() == ['pipenv', 'run', 'mlagents-learn', '--help']


def test_start_tensorboard(namespace_args):
    """Tests StartTensorboard.create_command() for the creation of a valid command."""

    namespace_args.tensorboard_start = True
    start_tensorboard = StartTensorboard(namespace_args)

    command = start_tensorboard.create_command()

    # Summaries folder will differ based on the system running the test
    command[3] = '--logdir=../../summaries'

    assert command == ['pipenv', 'run', 'tensorboard', '--logdir=../../summaries']


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
    result[3] = 'grimagents/training_wrapper.py'

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

    def mock_execute_command(command, show_command, dry_run):
        assert command == training_command_arguments

    monkeypatch.setattr(PerformTraining, 'create_command', mock_create_command)
    monkeypatch.setattr(grimagents.command_util, 'save_to_history', mock_save_to_history)
    monkeypatch.setattr(grimagents.command_util, 'execute_command', mock_execute_command)

    perform_training = PerformTraining(namespace_args)
    perform_training.execute()


def test_perform_training_command_dry_run(monkeypatch, namespace_args, grim_config):
    """Tests for the correct creation of a PerformTraining command with dry_run enabled."""

    namespace_args.dry_run = True

    def mock_load_config(config_path):
        return grim_config

    def mock_execute_command(command, show_command, dry_run):
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

    no_dry_run_args = Namespace(dry_run=False, args=[])

    dry_run_args = Namespace(dry_run=True, args=[])

    def mock_execute_command(command, show_command, dry_run):
        pass

    monkeypatch.setattr(grimagents.command_util, 'execute_command', mock_execute_command)

    command = Command(no_dry_run_args)
    command.execute()
    assert command.dry_run is False

    command = Command(dry_run_args)
    command.execute()
    assert command.dry_run is True
