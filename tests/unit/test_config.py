from pathlib import Path
import pytest

import grimagents.config as config


GRIM_CONFIG_FILE = 'grim_config.json'


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
