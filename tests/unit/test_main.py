from argparse import Namespace
from grimagents.__main__ import PerformTraining

import grimagents.config
import pytest


@pytest.fixture
def fixture_perform_training_single_config(monkeypatch):
    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        return training_command

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)

    yield 'fixture_perform_training_single_config'


def test_perform_training_single_trainer_config(fixture_perform_training_single_config):
    """Tests for the correct creation of a single command if only one trainer_config
    is defined in the grinagents configuration file.
    """

    args = Namespace(configuration_file='config\\3DBall_grimagents.json',
                     new_window=False,
                     args=[])

    perform_training = PerformTraining()

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', '3DBall', '--train']

    with pytest.raises(StopIteration):
        next(generator)


def test_perform_training_multiple_trainer_configs(monkeypatch):
    """Tests for the correct creation of multiple commands if multiple trainer_configs
    are defined in the grimagents configuration file.
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
    }

    def mock_load_config(config_path):
        return test_config

    def mock_override_args(self, training_command, args):
        return training_command

    monkeypatch.setattr(grimagents.config, "load_grim_config_file", mock_load_config)
    monkeypatch.setattr(PerformTraining, "override_configuration_values", mock_override_args)

    perform_training = PerformTraining()

    generator = perform_training.create_command(args)
    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--run-id', '3DBall', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\2DBall.yaml', '--run-id', '3DBall', '--train']

    assert next(generator) == ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\NoBall.yaml', '--run-id', '3DBall', '--train']

    with pytest.raises(StopIteration):
        next(generator)
