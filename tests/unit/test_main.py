from argparse import Namespace
from grimagents.__main__ import Command, PerformTraining, ResumeTraining
from grimagents.training_commands import TrainingWrapperCommand

import grimagents.config


def test_perform_training_single_trainer_config(monkeypatch):
    """Tests for the correct creation of a training command.
    """

    args = Namespace(configuration_file='config\\3DBall_grimagents.json', new_window=False, args=[])

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        return training_command

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)

    perform_training = PerformTraining()

    command = perform_training.create_command(args)

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    command[3] = 'grimagents\\training_wrapper.py'

    assert command == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--run-id',
        '3DBall',
        '--env',
        'builds\\3DBall\\Unity Environment.exe',
        '--export-path',
        'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels',
        '--train',
    ]


def test_override_configuration_values(monkeypatch):
    """Test for correct creation of a training command with command line argument overrides.

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

    def mock_load_config(config_path):
        return test_config

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)

    training_command = TrainingWrapperCommand(test_config)
    training_command.set_additional_arguments(args.args)

    perform_training = PerformTraining()
    perform_training.override_configuration_values(training_command, args)

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
        '--num-envs',
        '2',
        '--no-graphics',
        '--train',
    ]


def test_command_dry_run(monkeypatch):
    """Tests the correct assignment of the 'dry_run' argument for the following commands:
        - Command
        - ListTrainingOptions
        - StartTensorboard
        - ResumeTraining
    """

    no_dry_run_args = Namespace(new_window=False, dry_run=False, args=[])

    dry_run_args = Namespace(new_window=False, dry_run=True, args=[])

    def mock_override_execute_command(command, new_window, show_command, dry_run):
        pass

    monkeypatch.setattr(grimagents.command_util, "execute_command", mock_override_execute_command)

    command = Command()
    command.execute(no_dry_run_args)
    assert command.dry_run is False

    command = Command()
    command.execute(dry_run_args)
    assert command.dry_run is True


def test_perform_training_command_dry_run(monkeypatch):
    """Test for the correct assignment of dry_run argument to PerformTraining."""

    no_dry_run_args = Namespace(
        configuration_file='config\\3DBall_grimagents.json',
        new_window=False,
        dry_run=False,
        args=[],
    )

    dry_run_args = Namespace(
        configuration_file='config\\3DBall_grimagents.json', new_window=False, dry_run=True, args=[]
    )

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--env": "builds\\3DBall\\Unity Environment.exe",
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        return training_command

    def mock_override_execute_command(command, new_window, show_command, dry_run):
        pass

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)
    monkeypatch.setattr(grimagents.command_util, "execute_command", mock_override_execute_command)

    perform_training = PerformTraining()
    perform_training.execute(no_dry_run_args)
    assert perform_training.dry_run is False

    perform_training = PerformTraining()
    perform_training.execute(dry_run_args)
    assert perform_training.dry_run is True


def test_resume_training(monkeypatch):
    """Tests for correct parsing of ResumeTraining command.

    - Ensures --load argument is appended
    - Ensures --lesson argument is appended, if present in args
    """

    last_history = [
        "pipenv",
        "run",
        "python",
        "grimagents\\training_wrapper.py",
        "config\\3DBall.yaml",
        "--env",
        "builds\\3DBall\\Unity Environment.exe",
        "--run-id",
        "3DBall_01-2019-07-12_23-55-05",
        "--train",
    ]

    def mock_load_history():
        return last_history

    monkeypatch.setattr(grimagents.command_util, "load_last_history", mock_load_history)

    resume_training = ResumeTraining()

    # --load argument is appended and --lesson IS NOT present
    args = Namespace(new_window=False, dry_run=False, lesson=None, args=[])
    assert resume_training.create_command(args) == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\Unity Environment.exe',
        '--run-id',
        '3DBall_01-2019-07-12_23-55-05',
        '--train',
        '--load',
    ]

    # --load argument is appended and --lesson IS present
    args = Namespace(new_window=False, dry_run=False, lesson=3, args=[])
    assert resume_training.create_command(args) == [
        'pipenv',
        'run',
        'python',
        'grimagents\\training_wrapper.py',
        'config\\3DBall.yaml',
        '--env',
        'builds\\3DBall\\Unity Environment.exe',
        '--run-id',
        '3DBall_01-2019-07-12_23-55-05',
        '--train',
        '--load',
        '--lesson',
        '3',
    ]
