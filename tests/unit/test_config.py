import json
import pytest
import yaml

from pathlib import Path

import grimagents.config as config
import grimagents.command_util as command_util


def delete_file(file: Path):
    if file.exists():
        file.unlink()


@pytest.fixture
def valid_grim_config():
    return {"trainer-config-path": "config\\3DBall.yaml", "--run-id": "3DBall"}


@pytest.fixture
def invalid_grim_config():
    return {"--env": "builds\\3DBall\\3DBall.exe", "--run-id": "3DBall"}


@pytest.fixture
def trainer_config():
    return {
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


@pytest.fixture
def grim_config_path():
    return Path(__file__).parent / 'grim_config.json'


@pytest.fixture
def trainer_config_path():
    return Path(__file__).parent / 'trainer_config.yaml'


@pytest.fixture
def curriculum_config_path():
    return Path(__file__).parent / 'curriculum.json'


@pytest.fixture
def fixture_cleanup_grim_config(grim_config_path):
    delete_file(grim_config_path)
    yield 'fixture_cleanup_grim_config'
    delete_file(grim_config_path)


@pytest.fixture
def fixture_cleanup_trainer_config(trainer_config_path):
    delete_file(trainer_config_path)
    yield 'fixture_cleanup_trainer_config'
    delete_file(trainer_config_path)


@pytest.fixture
def fixture_cleanup_curriculum(curriculum_config_path):
    delete_file(curriculum_config_path)
    yield 'fixture_cleanup_curriculum'
    delete_file(curriculum_config_path)


def test_create_grim_config_file(grim_config_path, fixture_cleanup_grim_config):
    """Test for creating a new grimagents configuration file."""

    config.create_grim_config_file(grim_config_path)
    assert grim_config_path.exists()

    data = config.get_default_grim_config()
    with grim_config_path.open('r') as f:
        file_data = json.load(f)

    assert data == file_data


def test_load_grim_configuration_file(
    grim_config_path, valid_grim_config, fixture_cleanup_grim_config
):
    """Test for loading a valid grimagents configuration file."""

    with grim_config_path.open(mode='w') as f:
        json.dump(valid_grim_config, f, indent=4)

    file_data = config.load_grim_configuration_file(grim_config_path)
    assert file_data == valid_grim_config


def test_invalid_configuration_error(
    grim_config_path, invalid_grim_config, fixture_cleanup_grim_config
):
    """Test for raising an InvalidConfigurationError for an invalid grimagents
    configuration file.
    """

    with grim_config_path.open(mode='w') as f:
        json.dump(invalid_grim_config, f, indent=4)

    with pytest.raises(config.InvalidConfigurationError):
        config.load_grim_configuration_file(grim_config_path)


def test_configuration_validation():
    """Test for validating or rejecting grimagent configurations."""

    configuration = {"--env": "builds\\3DBall\\3DBall.exe"}
    assert config.validate_grim_configuration(configuration) is False

    configuration['trainer-config-path'] = 'config\\3DBall.yaml'
    assert config.validate_grim_configuration(configuration) is False

    configuration['--run-id'] = '3DBall'
    assert config.validate_grim_configuration(configuration) is True


def test_load_trainer_configuration(
    trainer_config_path, trainer_config, fixture_cleanup_trainer_config
):
    """Tests for the correct loading of a trainer configuration dictionary from file. """

    with trainer_config_path.open(mode='w') as f:
        yaml.dump(trainer_config, f, indent=4)

    loaded_configuration = config.load_trainer_configuration_file(trainer_config_path)
    assert loaded_configuration['default']['buffer_size'] == 10240


def test_create_trainer_config_file(trainer_config_path, fixture_cleanup_trainer_config):
    """Test for creating default trainer configuration files."""

    config.create_trainer_configuration_file(trainer_config_path)
    assert trainer_config_path.exists()


def test_create_curriculum_file(curriculum_config_path, fixture_cleanup_curriculum):
    """Test for creating a default curriculum file."""

    config.create_curriculum_file(curriculum_config_path)
    assert curriculum_config_path.exists()


def test_no_add_search_entry(grim_config_path, fixture_cleanup_grim_config, monkeypatch):
    """Test for correct handling of the 'add_search' flag in config.edit_grim_config_file()
    """

    def mock_open_file(file_path):
        pass

    monkeypatch.setattr(command_util, "open_file", mock_open_file)

    config.edit_grim_config_file(grim_config_path, add_search=False)

    with grim_config_path.open('r') as f:
        file_data = json.load(f)

    assert 'search' not in file_data

    config.edit_grim_config_file(grim_config_path, add_search=True)
    with grim_config_path.open('r') as f:
        file_data = json.load(f)

    assert 'search' in file_data


def test_no_overwrite_search_entry(grim_config_path, fixture_cleanup_grim_config, monkeypatch):
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

    with grim_config_path.open('w') as f:
        json.dump(configuration, f)

    config.edit_grim_config_file(grim_config_path, add_search=True)

    with grim_config_path.open('r') as f:
        file_data = json.load(f)

    assert 'brain' not in file_data['search']
