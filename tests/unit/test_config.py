import json
from pathlib import Path
import pytest

import grimagents.config as config
import grimagents.command_util as command_util


GRIM_CONFIG_FILE = 'grim_config.json'
TRAINER_CONFIG_FILE = 'trainer_config.yaml'
CURRICULUM_FILE = 'curriculum.json'

VALID_CONFIGURATION = {"trainer-config-path": "config\\3DBall.yaml", "--run-id": "3DBall"}

INVALID_CONFIGURATION = {"--env": "builds\\3DBall\\Unity Environment.exe", "--run-id": "3DBall"}


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
def fixture_grim_config_file():
    delete_file(get_grim_config_file_path())
    yield 'fixture_grim_config_file'
    delete_file(get_grim_config_file_path())


@pytest.fixture
def fixture_trainer_config_file():
    delete_file(get_trainer_config_file_path())
    yield 'fixture_trainer_config_file'
    delete_file(get_trainer_config_file_path())


@pytest.fixture
def fixture_curriculum_file():
    delete_file(get_curriculum_file_path())
    yield 'fixture_curriculum_file'
    delete_file(get_curriculum_file_path())


def test_create_grim_config_file(fixture_grim_config_file):
    """Test for creating a new grimagents configuration file."""

    path = get_grim_config_file_path()
    config.create_grim_config_file(path)
    assert path.exists()

    data = config.get_default_grim_config()
    with path.open('r') as f:
        file_data = json.load(f)

    assert data == file_data


def test_load_grim_config_file(fixture_grim_config_file):
    """Test for loading a valid grimagents configuration file."""

    path = get_grim_config_file_path()
    with path.open(mode='w') as f:
        json.dump(VALID_CONFIGURATION, f, indent=4)

    file_data = config.load_grim_config_file(path)
    assert file_data == VALID_CONFIGURATION


def test_invalid_configuration_error(fixture_grim_config_file):
    """Test for raising an InvalidConfigurationError for an invalid grimagents
    configuration file.
    """

    path = get_grim_config_file_path()
    with path.open(mode='w') as f:
        json.dump(INVALID_CONFIGURATION, f, indent=4)

    with pytest.raises(config.InvalidConfigurationError):
        config.load_grim_config_file(path)


def test_configuration_validation():
    """Test for validating or rejecting grimagent configurations."""

    configuration = {"--env": "builds\\3DBall\\Unity Environment.exe"}
    assert config.validate_grim_configuration(configuration) is False

    configuration['trainer-config-path'] = 'config\\3DBall.yaml'
    assert config.validate_grim_configuration(configuration) is False

    configuration['--run-id'] = '3DBall'
    assert config.validate_grim_configuration(configuration) is True


def test_create_trainer_config_file(fixture_trainer_config_file):
    """Test for creating default trainer configuration files."""

    path = get_trainer_config_file_path()
    config.create_trainer_configuration_file(path)
    assert path.exists()


def test_create_curriculum_file(fixture_curriculum_file):
    """Test for creating a default curriculum file."""

    path = get_curriculum_file_path()
    config.create_curriculum_file(path)
    assert path.exists()


def test_no_add_search_entry(fixture_grim_config_file, monkeypatch):
    """Test for correct handling of the 'add_search' flag in config.edit_grim_config_file()
    """

    def mock_open_file(file_path):
        pass

    monkeypatch.setattr(command_util, "open_file", mock_open_file)

    path = get_grim_config_file_path()
    config.edit_grim_config_file(path, add_search=False)

    with path.open('r') as f:
        file_data = json.load(f)

    assert 'search' not in file_data

    config.edit_grim_config_file(path, add_search=True)
    with path.open('r') as f:
        file_data = json.load(f)

    assert 'search' in file_data


def test_no_overwrite_search_entry(fixture_grim_config_file, monkeypatch):
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

    path = get_grim_config_file_path()
    with path.open('w') as f:
        json.dump(configuration, f)

    config.edit_grim_config_file(path, add_search=True)

    with path.open('r') as f:
        file_data = json.load(f)

    assert 'brain' not in file_data['search']
