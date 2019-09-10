import pytest
import subprocess

from argparse import Namespace
from pathlib import Path

import grimagents.config
import grimagents.command_util

from grimagents.search_commands import (
    SearchCommand,
    PerformGridSearch,
    ExportGridSearchConfiguration,
)


@pytest.fixture
def test_file():
    return Path(__file__).parent / 'test_file'


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
        random=None,
        resume=None,
        search_count=False,
    )


@pytest.fixture
def patch_search_command(monkeypatch, grim_config, trainer_config):
    """Patches external methods used in initializing and performing searches with SearchCommand objects."""

    def mock_load_grim_config(file_path: Path):
        return grim_config

    monkeypatch.setattr(grimagents.config, "load_grim_configuration_file", mock_load_grim_config)

    def mock_load_trainer_configuration(file_path: Path):
        return trainer_config

    monkeypatch.setattr(
        grimagents.config, "load_trainer_configuration_file", mock_load_trainer_configuration
    )

    def mock_write_yaml_file(yaml_data, file_path):
        pass

    monkeypatch.setattr(grimagents.command_util, "write_yaml_file", mock_write_yaml_file)


@pytest.fixture
def patch_perform_grid_search(monkeypatch, trainer_config, intersect):
    """Patches external methods used by calling PerformGridSearch.execute()."""

    def mock_get_intersect_count(self):
        return 10

    monkeypatch.setattr(
        grimagents.parameter_search.GridSearch, 'get_intersect_count', mock_get_intersect_count
    )

    def mock_get_intersect(self, index):
        return intersect

    monkeypatch.setattr(grimagents.parameter_search.GridSearch, 'get_intersect', mock_get_intersect)

    def mock_get_brain_config_for_intersect(self, intersect):
        return trainer_config

    monkeypatch.setattr(
        grimagents.parameter_search.GridSearch,
        'get_brain_config_for_intersect',
        mock_get_brain_config_for_intersect,
    )

    def mock_perform_search_with_configuration(self, intersect, brain_config):
        pass

    monkeypatch.setattr(
        grimagents.search_commands.SearchCommand,
        'perform_search_with_configuration',
        mock_perform_search_with_configuration,
    )


@pytest.fixture
def fixture_cleanup_test_file(test_file):
    if test_file.exists():
        test_file.unlink()
    yield 'fixture_cleanup_test_file'
    if test_file.exists():
        test_file.unlink()


def test_search_command_init(patch_search_command, namespace_args, grim_config, trainer_config):
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
    monkeypatch, patch_search_command, namespace_args, grim_config, trainer_config, intersect
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


def test_perform_grid_search(
    patch_search_command,
    patch_perform_grid_search,
    namespace_args,
    test_file,
    fixture_cleanup_test_file,
):
    """Tests for the correct execution of a grid search."""

    search = PerformGridSearch(namespace_args)

    search.search_config_path = test_file
    with test_file.open('w') as f:
        f.write('Test search configuration')

    search.execute()

    # Correct number of searches are run
    assert search.search_counter == 9

    # Removal of test configuration file
    assert not test_file.exists()


def test_resume_perform_grid_search(
    monkeypatch, patch_search_command, patch_perform_grid_search, namespace_args
):
    """Tests for correct handling of the resume argument when executing a grid search."""

    # Raise exception when resuming at an index higher than the search configuration can accommodate
    namespace_args.resume = 11
    search = PerformGridSearch(namespace_args)
    with pytest.raises(IndexError):
        search.execute()

    # Correct number of searchs are run when resuming
    class Counter:
        def __init__(self):
            self.count = 0

        def increment_counter(self):
            self.count += 1

    search_counter = Counter()

    def mock_perform_search_with_configuration(self, intersect, brain_config):
        search_counter.increment_counter()

    monkeypatch.setattr(
        grimagents.search_commands.SearchCommand,
        'perform_search_with_configuration',
        mock_perform_search_with_configuration,
    )

    namespace_args.resume = 3
    search = PerformGridSearch(namespace_args)
    search.execute()

    assert search_counter.count == 7


def test_export_grid_search_configuration(
    monkeypatch, patch_search_command, patch_perform_grid_search, namespace_args, trainer_config
):
    """Tests exporting a grid search trainer configuration to file."""

    def mock_write_yaml_file(yaml_data, file_path):
        assert yaml_data == trainer_config

    monkeypatch.setattr(grimagents.command_util, "write_yaml_file", mock_write_yaml_file)

    namespace_args.export_index = 1
    export = ExportGridSearchConfiguration(namespace_args)
    export.execute()


