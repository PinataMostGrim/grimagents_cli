import json
from pathlib import Path
import pytest

import grimagents.config as config


GRIM_CONFIG_FILE = 'grim_config.json'

VALID_CONFIGURATION = {
    "trainer-config-path": "config\\3DBall.yaml",
    "--run-id": "3DBall",
}

INVALID_CONFIGURATION = {
    "--env": "builds\\3DBall\\Unity Environment.exe",
    "--run-id": "3DBall",
}


def get_grim_config_file_path():
    return Path(__file__).parent / GRIM_CONFIG_FILE


def delete_grim_config_file():
    file = get_grim_config_file_path()
    if file.exists():
        file.unlink()


@pytest.fixture
def fixture_grim_config_file():
    yield 'fixture_grim_config_file'
    delete_grim_config_file()


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
