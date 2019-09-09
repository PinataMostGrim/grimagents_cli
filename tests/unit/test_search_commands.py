import pytest
import subprocess

from argparse import Namespace
from pathlib import Path

import grimagents.config
import grimagents.command_util

from grimagents.search_commands import SearchCommand


@pytest.fixture
def grim_config():
    return {
        'trainer-config-path': 'config\\3DBall.yaml',
        '--env': 'builds\\3DBall\\3DBall.exe',
        '--run-id': '3DBall',
        '--seed': '',
        '--timestamp': True,
        'search': {
            'brain': {
                'name': '3DBallLearning',
                'hyperparameters': {
                    'batch_size': [64, 256],
                    'buffer_size_multiple': [50, 200],
                    'beta': [0.01, 0.0001],
                },
            }
        },
    }


@pytest.fixture
def trainer_config():
    return {
        'default': {
            'trainer': 'ppo',
            'batch_size': 1024,
            'beta': 0.005,
            'buffer_size': 10240,
            'epsilon': 0.2,
            'gamma': 0.99,
            'hidden_units': 128,
            'lambd': 0.95,
            'learning_rate': 0.0003,
            'max_steps': '5.0e4',
            'num_epoch': 3,
            'num_layers': 2,
            'time_horizon': 64,
            'use_curiosity': False,
            'curiosity_strength': 0.01,
            'curiosity_enc_size': 128,
        },
        '3DBallLearning': {
            'max_steps': '3.0e3',
            'normalize': True,
            'batch_size': 84,
            'buffer_size': 7392,
            'summary_freq': 1000,
            'time_horizon': 1000,
            'lambd': 0.99,
            'gamma': 0.995,
            'beta': 0.002,
            'use_curiosity': True,
        },
    }


@pytest.fixture
def intersect():
    return {'batch_size': 84, 'buffer_size_multiple': 88, 'beta': 0.002}


@pytest.fixture
def namespace_args():
    return Namespace(
        bayes_load=False,
        bayes_save=False,
        bayesian=None,
        configuration_file='config\\3DBall_grimagents.json',
        edit_config=None,
        export_index=None,
        parallel=False,
        random=4,
        resume=None,
        search_count=False,
    )


@pytest.fixture
def patch_load_grim_config(monkeypatch, grim_config):
    def mock_load_grim_config(file_path: Path):
        return grim_config

    monkeypatch.setattr(grimagents.config, "load_grim_configuration_file", mock_load_grim_config)


@pytest.fixture
def patch_load_trainer_config(monkeypatch, trainer_config):
    def mock_load_trainer_configuration(file_path: Path):
        return trainer_config

    monkeypatch.setattr(
        grimagents.config, "load_trainer_configuration_file", mock_load_trainer_configuration
    )


def test_search_command_init(
    patch_load_grim_config, patch_load_trainer_config, grim_config, trainer_config, namespace_args
):
    """Tests for the correct initialization of a SearchCommand object."""

    search_command = SearchCommand(namespace_args)

    assert search_command.grim_config_path == Path('config\\3DBall_grimagents.json')
    assert search_command.grim_config == grim_config

    assert search_command.trainer_config_path == Path('config\\3DBall.yaml')
    assert search_command.trainer_config == trainer_config

    assert search_command.search_config_path == search_command.trainer_config_path.with_name(
        'search_config.yaml'
    )

    assert search_command.search_counter == 0


def test_perform_search_with_configuration(
    monkeypatch,
    patch_load_grim_config,
    patch_load_trainer_config,
    grim_config,
    trainer_config,
    namespace_args,
    intersect,
):
    """Tests for the correct creation of search commands for grimagents."""

    def mock_write_yaml_file(yaml_data, file_path):
        assert yaml_data == trainer_config

    monkeypatch.setattr(grimagents.command_util, "write_yaml_file", mock_write_yaml_file)

    def mock_run(command):
        assert command == [
            'pipenv',
            'run',
            'python',
            '-m',
            'grimagents',
            'config\\3DBall_grimagents.json',
            '--trainer-config',
            'config\\search_config.yaml',
            '--run-id',
            '3DBall_00',
        ]

    monkeypatch.setattr(subprocess, 'run', mock_run)

    search_command = SearchCommand(namespace_args)
    search_command.perform_search_with_configuration(intersect, trainer_config)



