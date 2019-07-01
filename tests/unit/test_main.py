from argparse import Namespace
from grimagents.__main__ import PerformTraining

import grimagents.config
import pytest


def test_perform_training_single_trainer_config(monkeypatch):
    """Tests for the correct creation of a single command if only one trainer_config
    is defined in the grinagents configuration file.

    - Ensures the --env setting is preserved
    - Ensures a --base-port is added
    """

    args = Namespace(configuration_file='config\\3DBall_grimagents.json',
                     new_window=False,
                     args=[])

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "--env": "builds\\3DBall\\Unity Environment.exe",
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        return training_command

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)

    perform_training = PerformTraining()

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', '3DBall', '--env', 'builds\\3DBall\\Unity Environment.exe', '--base-port', '5010', '--train']

    with pytest.raises(StopIteration):
        next(generator)


def test_perform_training_multiple_trainer_configs(monkeypatch):
    """Tests for the correct creation of multiple commands if multiple trainer_configs
    are defined in the grimagents configuration file.

    - Ensures the '--env' setting is removed
    - Ensures the correct base ports are added
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
        "--base-port": "5006",
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        # create_command() ensures the '--env' is stripped by adding it as
        # a command line argument, which overrides configuration values.
        assert args.env == ''
        training_command.set_env('')
        return training_command

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)

    perform_training = PerformTraining()

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', '3DBall_00', '--base-port', '5006', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\2DBall.yaml', '--run-id', '3DBall_01', '--base-port', '5007', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\NoBall.yaml', '--run-id', '3DBall_02', '--base-port', '5008', '--train']

    with pytest.raises(StopIteration):
        next(generator)



    with pytest.raises(StopIteration):
        next(generator)
