import bayes_opt.util
import pytest
import shutil
import subprocess

from argparse import Namespace
from bayes_opt import BayesianOptimization
from pathlib import Path

import grimagents.command_util
import grimagents.config
import grimagents.common
import grimagents.settings

from grimagents.search_commands import (
    SearchCommand,
    PerformGridSearch,
    ExportGridSearchConfiguration,
    PerformRandomSearch,
    PerformBayesianSearch,
)

from grimagents.parameter_search import ParameterSearch, GridSearch, BayesianSearch


class Counter:
    def __init__(self):
        self.count = 0

    def increment_counter(self):
        self.count += 1

    def reset_counter(self):
        self.count = 0


@pytest.fixture
def test_file():
    return Path(__file__).parent / 'test_file'


@pytest.fixture
def bayes_log_folder():
    return Path(__file__).parent / '3DBall_bayes'


@pytest.fixture
def grim_config():
    return {
        'trainer-config-path': 'config/3DBall.yaml',
        '--env': 'builds/3DBall/3DBall.exe',
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
    return {'batch_size': 84, 'beta': 0.002, 'buffer_size_multiple': 88}


@pytest.fixture
def bounds():
    return {'batch_size': [64, 256], 'buffer_size_multiple': [50, 200], 'beta': [0.01, 0.0001]}


@pytest.fixture
def namespace_args():
    return Namespace(
        bayes_load=False,
        bayes_save=False,
        bayesian=None,
        configuration_file='config/3DBall_grimagents.json',
        edit_config=None,
        export_index=None,
        random=None,
        resume=None,
        search_count=False,
    )


@pytest.fixture
def fixture_cleanup_bayes_log_folder(bayes_log_folder):
    """Fixture that ensures the test Bayesian search log folder is deleted before and after the test is run."""

    if bayes_log_folder.exists():
        shutil.rmtree(bayes_log_folder)
    yield 'fixture_cleanup_bayes_log_folder'
    if bayes_log_folder.exists():
        shutil.rmtree(bayes_log_folder)


@pytest.fixture
def patch_search_command(monkeypatch, grim_config, trainer_config):
    """Patches all external methods used by SearchCommand objects."""

    def mock_load_grim_config(file_path: Path):
        return grim_config

    def mock_load_trainer_configuration(file_path: Path):
        return trainer_config

    def mock_write_yaml_file(yaml_data, file_path):
        pass

    def mock_run(command):
        pass

    monkeypatch.setattr(grimagents.config, 'load_grim_configuration_file', mock_load_grim_config)

    monkeypatch.setattr(
        grimagents.config, 'load_trainer_configuration_file', mock_load_trainer_configuration
    )

    monkeypatch.setattr(grimagents.command_util, 'write_yaml_file', mock_write_yaml_file)
    monkeypatch.setattr(subprocess, 'run', mock_run)


@pytest.fixture
def patch_perform_search_with_configuration(monkeypatch):
    """Patches SearchCommand.perform_search_with_configuration()."""

    def mock_perform_search_with_configuration(self, brain_config):
        pass

    monkeypatch.setattr(
        SearchCommand, 'perform_search_with_configuration', mock_perform_search_with_configuration
    )


@pytest.fixture
def patch_perform_grid_search(monkeypatch, trainer_config, intersect):
    """Patches external methods used by PerformGridSearch objects."""

    def mock_get_intersect_count(self):
        return 10

    def mock_get_intersect(self, index):
        return intersect

    def mock_get_brain_config_for_intersect(self, intersect):
        return trainer_config

    monkeypatch.setattr(GridSearch, 'get_intersect_count', mock_get_intersect_count)
    monkeypatch.setattr(GridSearch, 'get_intersect', mock_get_intersect)

    monkeypatch.setattr(
        ParameterSearch, 'get_brain_config_for_intersect', mock_get_brain_config_for_intersect
    )


@pytest.fixture
def patch_perform_bayesian_search(monkeypatch, bounds, trainer_config):
    """Patches all external calls used by PerformBayesianSearch objects."""

    def mock_get_parameter_bounds(self, names, values):
        return bounds

    def mock_sanitize_parameter_values(self, dict):
        return {}

    def mock_get_brain_config_for_intersect(self, intersect):
        return trainer_config

    def mock_bayes_opt_load_logs(optimizer, logs):
        pass

    def mock_optimizer_subscribe(self, step, logger):
        pass

    def mock_optimizer_maximize(self, init_points, n_iter):
        pass

    monkeypatch.setattr(BayesianSearch, 'get_parameter_bounds', mock_get_parameter_bounds)

    monkeypatch.setattr(BayesianSearch, 'sanitize_parameter_values', mock_sanitize_parameter_values)

    monkeypatch.setattr(
        BayesianSearch, 'get_brain_config_for_intersect', mock_get_brain_config_for_intersect
    )

    monkeypatch.setattr(bayes_opt.util, 'load_logs', mock_bayes_opt_load_logs)
    monkeypatch.setattr(BayesianOptimization, 'subscribe', mock_optimizer_subscribe)
    monkeypatch.setattr(BayesianOptimization, 'maximize', mock_optimizer_maximize)


@pytest.fixture
def patch_get_optimizer_max(monkeypatch):
    def mock_get_optimizer_max(self, optimizer):
        return {'target': 1.756, 'params': {'batch_size': 144.0682249028942}}

    monkeypatch.setattr(PerformBayesianSearch, 'get_optimizer_max', mock_get_optimizer_max)


@pytest.fixture
def patch_get_last_mean_reward_from_log(monkeypatch):
    def mock_get_last_mean_reward_from_log(self):
        return float(1.358)

    monkeypatch.setattr(
        PerformBayesianSearch, 'get_last_mean_reward_from_log', mock_get_last_mean_reward_from_log
    )


@pytest.fixture
def patch_get_load_log_paths(monkeypatch, bayes_log_folder):
    def mock_get_load_log_paths(self):
        return [
            bayes_log_folder / 'file1.json',
            bayes_log_folder / 'file2.json',
            bayes_log_folder / 'file3.json',
        ]

    monkeypatch.setattr(PerformBayesianSearch, 'get_load_log_paths', mock_get_load_log_paths)


@pytest.fixture
def patch_get_save_log_path(monkeypatch):
    def mock_get_save_log_path(self):
        return Path(__file__).parent / '3DBall_bayes/3DBall_2019-09-13_03-41-44.json'

    monkeypatch.setattr(PerformBayesianSearch, 'get_save_log_path', mock_get_save_log_path)


@pytest.fixture
def patch_save_max_to_file(monkeypatch):
    def mock_save_max_to_file(self, max):
        pass

    monkeypatch.setattr(PerformBayesianSearch, 'save_max_to_file', mock_save_max_to_file)


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

    assert search_command.grim_config_path == Path('config/3DBall_grimagents.json')
    assert search_command.grim_config == grim_config

    assert search_command.trainer_config_path == Path('config/3DBall.yaml')
    assert search_command.trainer_config == trainer_config

    assert search_command.search_config_path == search_command.trainer_config_path.with_name(
        'search_config.yaml'
    )

    assert search_command.search_counter == 0


def test_search_command_get_run_id(patch_search_command, namespace_args):
    """Tests for the correct construction of a search run_id."""

    search_command = SearchCommand(namespace_args)
    assert search_command.get_search_run_id() == '3DBall_00'

    search_command.search_counter = 4
    assert search_command.get_search_run_id() == '3DBall_04'


def test_perform_search_with_configuration(
    monkeypatch, patch_search_command, namespace_args, grim_config, trainer_config, intersect
):
    """Tests that SearchCommand objects correctly perform searches with the specified trainer configurations.

    Ensures:
        - The correct configuration file is written for the search
        - The correct grimagents training command is generated
    """

    def mock_write_yaml_file(yaml_data, file_path):
        assert yaml_data == trainer_config

    def mock_run(command):
        assert command == [
            'pipenv',
            'run',
            'python',
            '-m',
            'grimagents',
            str(Path(namespace_args.configuration_file)),
            '--trainer-config',
            str(Path('config/search_config.yaml')),
            '--run-id',
            '3DBall_00',
        ]

    monkeypatch.setattr(grimagents.command_util, "write_yaml_file", mock_write_yaml_file)
    monkeypatch.setattr(subprocess, 'run', mock_run)

    search_command = SearchCommand(namespace_args)
    search_command.perform_search_with_configuration(trainer_config)


def test_perform_grid_search(
    patch_search_command,
    patch_perform_grid_search,
    patch_perform_search_with_configuration,
    namespace_args,
    test_file,
    fixture_cleanup_test_file,
):
    """Tests for the correct execution of a grid search.

    Ensures:
        - The correct number of searches are run
        - The search trainer configuration file is removed after training
    """

    search = PerformGridSearch(namespace_args)

    search.search_config_path = test_file
    with test_file.open('w') as f:
        f.write('Test search configuration')

    search.execute()

    assert search.search_counter == 9
    assert not test_file.exists()


def test_resume_perform_grid_search(
    monkeypatch,
    patch_search_command,
    patch_perform_search_with_configuration,
    patch_perform_grid_search,
    namespace_args,
):
    """Tests for correct handling of the resume argument when executing a grid search.

    Ensures:
        - An IndexError is raised if an invalid resume index is specified
        - The correct number of searches are run when resuming
    """

    namespace_args.resume = 11
    search = PerformGridSearch(namespace_args)
    with pytest.raises(IndexError):
        search.execute()

    search_counter = Counter()

    def mock_perform_search_with_configuration(self, brain_config):
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
    monkeypatch,
    patch_search_command,
    patch_perform_search_with_configuration,
    patch_perform_grid_search,
    namespace_args,
    trainer_config,
):
    """Tests exporting a grid search trainer configuration to file."""

    def mock_write_yaml_file(yaml_data, file_path):
        assert yaml_data == trainer_config

    monkeypatch.setattr(grimagents.command_util, 'write_yaml_file', mock_write_yaml_file)

    namespace_args.export_index = 1
    export = ExportGridSearchConfiguration(namespace_args)
    export.execute()


def test_perform_random_search(
    monkeypatch,
    patch_search_command,
    patch_perform_search_with_configuration,
    namespace_args,
    trainer_config,
    intersect,
    test_file,
    fixture_cleanup_test_file,
):
    """Tests for the correct execution of a random search.

    Ensures:
        - The correct number of searches are run
        - The search trainer configuration file is removed after training
    """

    def mock_get_randomized_intersect(self):
        return intersect

    def mock_get_brain_config_for_intersect(self, intersect):
        return trainer_config

    monkeypatch.setattr(
        grimagents.parameter_search.RandomSearch,
        'get_randomized_intersect',
        mock_get_randomized_intersect,
    )

    monkeypatch.setattr(
        grimagents.parameter_search.ParameterSearch,
        'get_brain_config_for_intersect',
        mock_get_brain_config_for_intersect,
    )

    namespace_args.random = 4
    search = PerformRandomSearch(namespace_args)

    search.search_config_path = test_file
    with test_file.open('w') as f:
        f.write('Test search configuration')

    search.execute()

    assert search.search_counter == 3
    assert not test_file.exists()


def test_perform_bayesian_search_init(patch_search_command, namespace_args):
    """Tests for the correct construction of a bayesian search trainer config output path."""

    search = PerformBayesianSearch(namespace_args)
    assert search.output_config_path == Path('config/bayes_config.yaml')


def test_perform_bayesian_search_execute(
    monkeypatch, patch_search_command, patch_perform_bayesian_search, patch_get_optimizer_max, patch_get_last_mean_reward_from_log, patch_get_load_log_paths, patch_get_save_log_path, patch_save_max_to_file, namespace_args,
):
    """Tests for the correct execution of a Bayesian search.

        - Ensures observation log loading respects the command line argument
        - Ensures obseration log saving respects the command line argument
        - Ensures the correct number of searches desired is communicated to the BayesianOptimization object
    """

    subscribe_counter = Counter()

    # As we are mocking optimizer.maximize(), perform_bayes_search() will never be called and does not need to be mocked.

    def mock_optimizer_maximize(self, init_points, n_iter):
        assert init_points == 2
        assert n_iter == 5

    monkeypatch.setattr(BayesianOptimization, 'maximize', mock_optimizer_maximize)

    def mock_bayes_opt_load_logs(optimizer, logs):
        assert type(logs) is list
        assert len(logs) == 3

    monkeypatch.setattr(bayes_opt.util, 'load_logs', mock_bayes_opt_load_logs)

    def mock_optimizer_subscribe(self, step, logger):
        subscribe_counter.increment_counter()

    monkeypatch.setattr(BayesianOptimization, 'subscribe', mock_optimizer_subscribe)

    namespace_args.bayesian = [2, 5]
    namespace_args.bayes_load = True
    namespace_args.bayes_save = True

    search = PerformBayesianSearch(namespace_args)
    search.execute()

    # The BayesianOptimization object calls subscribe() three times during maximization. A fourth call is made if PerformBayesianSearch object has decided to save optimization logs. As we are mocking the maximize() method, we expect only one call to subescribe().
    assert subscribe_counter.count == 1

    def mock_bayes_opt_load_logs(optimizer, logs):
        assert True is False

    monkeypatch.setattr(bayes_opt.util, 'load_logs', mock_bayes_opt_load_logs)

    subscribe_counter.reset_counter()

    namespace_args.bayesian = [2, 5]
    namespace_args.bayes_load = False
    namespace_args.bayes_save = False

    search = PerformBayesianSearch(namespace_args)
    search.execute()

    assert subscribe_counter.count == 0


def test_perform_bayes_search(
    monkeypatch,
    patch_search_command,
    patch_perform_bayesian_search,
    patch_get_load_log_paths,
    patch_get_last_mean_reward_from_log,
    patch_get_save_log_path,
    namespace_args,
    trainer_config,
):
    """Tests that PerformBayesianSearch objects correctly perform searches with the specified trainer configurations.

    Ensures:
        - The correct configuration file is written for the search
        - The correct grimagents training command is generated
    """

    def mock_write_yaml_file(yaml_data, file_path):
        assert yaml_data == trainer_config

    def mock_run(command):
        assert command == [
            'pipenv',
            'run',
            'python',
            '-m',
            'grimagents',
            str(Path(namespace_args.configuration_file)),
            '--trainer-config',
            str(Path('config/search_config.yaml')),
            '--run-id',
            '3DBall_00',
        ]

    monkeypatch.setattr(grimagents.command_util, 'write_yaml_file', mock_write_yaml_file)
    monkeypatch.setattr(subprocess, 'run', mock_run)

    namespace_args.bayesian = [1, 3]
    search = PerformBayesianSearch(namespace_args)
    search.perform_bayes_search(batch_size=84, beta=0.002, buffer_size_multiple=88)


def test_get_last_mean_reward_from_log(monkeypatch):
    """Tests for retrieval of the final mean reward of the last training run."""

    def mock_get_log_file_path():
        return Path()

    def mock_load_last_lines_from_file(log_file, number_of_lines):
        return [
            r'[2019-09-12 02:02:35,448][INFO] ---------------------------------------------------------------',
            r'[2019-09-12 02:02:35,450][INFO] Initiating \'3DBall_00-2019-09-12_02-02-34\'',
            r'[2019-09-12 02:03:14,850][INFO] Exporting brains:',
            r'[2019-09-12 02:03:14,855][INFO]     UnitySDK\Assets\ML-Agents\Examples\3DBall\ImportedModels\3DBallLearning.nn',
            r'[2019-09-12 02:03:14,855][INFO] ',
            r'Training run \'3DBall_00-2019-09-12_02-02-34\' ended after 39 seconds',
            r'[2019-09-12 02:03:14,857][INFO] Training completed successfully',
            r'[2019-09-12 02:03:14,858][INFO] Final Mean Reward: 1.358',
            r'[2019-09-12 02:03:14,858][INFO] ---------------------------------------------------------------',
        ]

    monkeypatch.setattr(grimagents.settings, 'get_log_file_path', mock_get_log_file_path)

    monkeypatch.setattr(
        grimagents.command_util, 'load_last_lines_from_file', mock_load_last_lines_from_file
    )

    assert PerformBayesianSearch.get_last_mean_reward_from_log() == float(1.358)


def test_save_max_to_file(
    monkeypatch,
    patch_search_command,
    patch_perform_bayesian_search,
    patch_get_last_mean_reward_from_log,
    patch_get_load_log_paths,
    patch_get_save_log_path,
    namespace_args,
    trainer_config,
):
    """Tests that a BayesianOptimization object's max property is correctly converted into a trainer configuration dictionary."""

    def mock_write_yaml_file(yaml_data, file_path):
        assert yaml_data == trainer_config

    monkeypatch.setattr(grimagents.command_util, 'write_yaml_file', mock_write_yaml_file)

    max = {
        'target': 1.595,
        'params': {
            'batch_size': 144.0682249028942,
            'beta': 0.0028687875149226343,
            'buffer_size_multiple': 50.017156222601734,
        },
    }

    search = PerformBayesianSearch(namespace_args)
    search.save_max_to_file(max)


def test_get_log_folder_path(
    monkeypatch,
    patch_search_command,
    namespace_args,
    bayes_log_folder,
    fixture_cleanup_bayes_log_folder,
):
    """Test for the correct generation of a Bayesian search log folder path and ensure the folder is created if it doesn't exist.

    Ensures:
        - The correct log folder path is returned
        - The log folder is created if it did not previously exist
    """

    search = PerformBayesianSearch(namespace_args)
    search.trainer_config_path = Path(__file__)

    assert search.get_log_folder_path() == bayes_log_folder
    assert bayes_log_folder.exists()


def test_get_save_log_path(monkeypatch, patch_search_command, namespace_args, bayes_log_folder):
    """Tests for the correct Bayesian search log file path creation."""

    def mock_get_timestamp():
        return '2019-09-13_03-41-44'

    def mock_get_log_folder_path(self):
        return bayes_log_folder

    monkeypatch.setattr(grimagents.common, 'get_timestamp', mock_get_timestamp)

    monkeypatch.setattr(PerformBayesianSearch, 'get_log_folder_path', mock_get_log_folder_path)

    log_path = Path(__file__).parent / '3DBall_bayes/3DBall_2019-09-13_03-41-44.json'

    search = PerformBayesianSearch(namespace_args)
    search.trainer_config_path = Path(__file__)

    assert search.get_save_log_path() == log_path


def test_get_load_log_paths(
    patch_search_command, namespace_args, bayes_log_folder, fixture_cleanup_bayes_log_folder
):
    """Tests that log files are loaded from the Bayesian search log folder.

    Ensures:
        - Loaded log file paths are correct
        - Non-json files are not loaded from the log folder
    """

    bayes_log_folder.mkdir()

    log_paths = [
        bayes_log_folder / 'file1.json',
        bayes_log_folder / 'file2.json',
        bayes_log_folder / 'file3.json',
        bayes_log_folder / 'file4',
    ]

    for filename in log_paths:
        log_file = bayes_log_folder / filename
        with log_file.open('w') as f:
            f.write('Test log file')

    search = PerformBayesianSearch(namespace_args)
    search.trainer_config_path = Path(__file__)
    retrieved_log_paths = search.get_load_log_paths()

    for pair in zip(log_paths, retrieved_log_paths):
        assert pair[0] == pair[1]

    assert len(retrieved_log_paths) == 3
