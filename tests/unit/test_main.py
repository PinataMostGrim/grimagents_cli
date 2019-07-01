from argparse import Namespace
from grimagents.__main__ import PerformTraining

import grimagents.config
import pytest


def test_perform_training_single_trainer_config(monkeypatch):
    """Tests for the correct creation of a single command if only one trainer_config
    is defined in the grinagents configuration file.

    - Ensures the --export-path setting is preserved
    - Ensures a --base-port is added
    - Ensures only one training command is generated
    """

    args = Namespace(configuration_file='config\\3DBall_grimagents.json',
                     new_window=False,
                     args=[])

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

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', '3DBall', '--env', 'builds\\3DBall\\Unity Environment.exe', '--export-path', 'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels', '--base-port', '5010', '--train']

    with pytest.raises(StopIteration):
        next(generator)


def test_perform_training_multiple_trainer_configs(monkeypatch):
    """Tests for the correct creation of multiple commands if multiple trainer_configs
    are defined in the grimagents configuration file.

    - Ensures the '--export-path' setting is removed
    - Ensures the correct base ports are added
    - Ensures the run_id is unique for each training command
    - Ensures the correct number of commands are generated
    """

    args = Namespace(configuration_file='config\\3DBall_grimagents.json',
                     new_window=False,
                     args=[])

    test_config = {
        "trainer-config-path": [
            "config\\3DBall.yaml",
            "config\\2DBall.yaml",
            "config\\NoBall.yaml"
            ],
        "--run-id": "3DBall",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
        "--base-port": "5006",
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        return training_command

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)

    perform_training = PerformTraining()

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', '3DBall_00', '--env', 'builds\\3DBall\\Unity Environment.exe', '--base-port', '5006', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\2DBall.yaml', '--run-id', '3DBall_01', '--env', 'builds\\3DBall\\Unity Environment.exe', '--base-port', '5007', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\NoBall.yaml', '--run-id', '3DBall_02', '--env', 'builds\\3DBall\\Unity Environment.exe', '--base-port', '5008', '--train']

    with pytest.raises(StopIteration):
        next(generator)


def test_override_configuration_values(monkeypatch):
    """Test for correct creation of a training command with command line argument overrides.
    """
    args = Namespace(configuration_file='config\\3DBall_grimagents.json',
                     new_window=False,
                     env='',
                     lesson=2,
                     run_id='ball',
                     num_envs=2,
                     graphics=None,
                     no_graphics=True,
                     timestamp=None,
                     no_timestamp=True,
                     args=['--load', '--slow'])

    test_config = {
        "trainer-config-path": [
            "config\\3DBall.yaml",
            "config\\2DBall.yaml"
            ],
        "--run-id": "3DBall",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--base-port": "5006",
        "--timestamp": True
    }

    def mock_load_config(config_path):
        return test_config

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)

    perform_training = PerformTraining()

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', 'ball_00', '--base-port', '5006', '--lesson', '2', '--num-envs', '2', '--no-graphics', '--load', '--slow', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\2DBall.yaml', '--run-id', 'ball_01', '--base-port', '5007', '--lesson', '2', '--num-envs', '2', '--no-graphics', '--load', '--slow', '--train']

    with pytest.raises(StopIteration):
        next(generator)
