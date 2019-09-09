import json
import pytest
import yaml

from pathlib import Path

import grimagents.config as config
import grimagents.command_util as command_util


GRIM_CONFIG_FILE = 'grim_config.json'
TRAINER_CONFIG_FILE = 'trainer_config.yaml'
CURRICULUM_FILE = 'curriculum.json'

VALID_GRIM_CONFIG = {"trainer-config-path": "config\\3DBall.yaml", "--run-id": "3DBall"}

INVALID_GRIM_CONFIG = {"--env": "builds\\3DBall\\3DBall.exe", "--run-id": "3DBall"}

TRAINER_CONFIG = {
    'default': {
        'trainer': 'ppo',
        'batch_size': 1024,
        'beta': 5.0e-3,
        'buffer_size': 10240,
        'epsilon': 0.2,
        'gamma': 0.99,
        'hidden_units': 128,
        'lambd': 0.95,
        'learning_rate': 3.0e-4,
        'max_steps': 5.0e4,
        'num_epoch': 3,
        'num_layers': 2,
        'time_horizon': 64,
        'sequence_length': 64,
    }
}


def get_grim_config_file_path():
    return Path(__file__).parent / GRIM_CONFIG_FILE


def get_trainer_config_file_path():
    return Path(__file__).parent / TRAINER_CONFIG_FILE


def get_curriculum_file_path():
    return Path(__file__).parent / CURRICULUM_FILE


def delete_file(file: Path):
    if file.exists():
        file.unlink()


@pytest.fixture
def grim_config():
    return get_grim_config_file_path()


@pytest.fixture
def trainer_config():
    return get_trainer_config_file_path()


@pytest.fixture
def curriculum_config():
    return get_curriculum_file_path()


@pytest.fixture
def fixture_cleanup_grim_config():
    delete_file(get_grim_config_file_path())
    yield 'fixture_cleanup_grim_config'
    delete_file(get_grim_config_file_path())


@pytest.fixture
def fixture_cleanup_trainer_config():
    delete_file(get_trainer_config_file_path())
    yield 'fixture_cleanup_trainer_config'
    delete_file(get_trainer_config_file_path())


@pytest.fixture
def fixture_cleanup_curriculum():
    delete_file(get_curriculum_file_path())
    yield 'fixture_cleanup_curriculum'
    delete_file(get_curriculum_file_path())


def test_create_grim_config_file(grim_config, fixture_cleanup_grim_config):
    """Test for creating a new grimagents configuration file."""

    config.create_grim_config_file(grim_config)
    assert grim_config.exists()

    data = config.get_default_grim_config()
    with grim_config.open('r') as f:
        file_data = json.load(f)

    assert data == file_data


def test_load_grim_configuration_file(grim_config, fixture_cleanup_grim_config):
    """Test for loading a valid grimagents configuration file."""

    with grim_config.open(mode='w') as f:
        json.dump(VALID_GRIM_CONFIG, f, indent=4)

    file_data = config.load_grim_configuration_file(grim_config)
    assert file_data == VALID_GRIM_CONFIG


def test_invalid_configuration_error(grim_config, fixture_cleanup_grim_config):
    """Test for raising an InvalidConfigurationError for an invalid grimagents
    configuration file.
    """

    with grim_config.open(mode='w') as f:
        json.dump(INVALID_GRIM_CONFIG, f, indent=4)

    with pytest.raises(config.InvalidConfigurationError):
        config.load_grim_configuration_file(grim_config)


def test_configuration_validation():
    """Test for validating or rejecting grimagent configurations."""

    configuration = {"--env": "builds\\3DBall\\3DBall.exe"}
    assert config.validate_grim_configuration(configuration) is False

    configuration['trainer-config-path'] = 'config\\3DBall.yaml'
    assert config.validate_grim_configuration(configuration) is False

    configuration['--run-id'] = '3DBall'
    assert config.validate_grim_configuration(configuration) is True


def test_load_trainer_configuration(trainer_config, fixture_cleanup_trainer_config):
    """Tests for the correct loading of a trainer configuration dictionary from file. """

    with trainer_config.open(mode='w') as f:
        yaml.dump(TRAINER_CONFIG, f, indent=4)

    loaded_configuration = config.load_trainer_configuration_file(trainer_config)
    assert loaded_configuration['default']['buffer_size'] == 10240


def test_create_trainer_config_file(trainer_config, fixture_cleanup_trainer_config):
    """Test for creating default trainer configuration files."""

    config.create_trainer_configuration_file(trainer_config)
    assert trainer_config.exists()


def test_create_curriculum_file(curriculum_config, fixture_cleanup_curriculum):
    """Test for creating a default curriculum file."""

    config.create_curriculum_file(curriculum_config)
    assert curriculum_config.exists()


def test_no_add_search_entry(grim_config, fixture_cleanup_grim_config, monkeypatch):
    """Test for correct handling of the 'add_search' flag in config.edit_grim_config_file()
    """

    def mock_open_file(file_path):
        pass

    monkeypatch.setattr(command_util, "open_file", mock_open_file)

    config.edit_grim_config_file(grim_config, add_search=False)

    with grim_config.open('r') as f:
        file_data = json.load(f)

    assert 'search' not in file_data

    config.edit_grim_config_file(grim_config, add_search=True)
    with grim_config.open('r') as f:
        file_data = json.load(f)

    assert 'search' in file_data


def test_no_overwrite_search_entry(grim_config, fixture_cleanup_grim_config, monkeypatch):
    """Tests to ensure config.edit_grim_config_file(add_search=True) does not overwrite an existing search entry.
    """

    def mock_open_file(file_path):
        pass

    monkeypatch.setattr(command_util, "open_file", mock_open_file)

    configuration = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--run-id": "3DBall",
        "search": [],
    }

    with grim_config.open('w') as f:
        json.dump(configuration, f)

    config.edit_grim_config_file(grim_config, add_search=True)

    with grim_config.open('r') as f:
        file_data = json.load(f)

    assert 'brain' not in file_data['search']
