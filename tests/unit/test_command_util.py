import json
import pytest
import yaml

from pathlib import Path

import grimagents.command_util as command_util


def get_test_file_path():
    return Path(__file__).parent / 'test_file'


def get_nested_test_file_path():
    return Path(__file__).parent / 'test_folder' / 'test_file'


def delete_file(file: Path):
    if file.exists():
        file.unlink()


def delete_folder(folder: Path):
    """Deletes an empty folder."""

    if folder.exists() and folder.is_dir():
        folder.rmdir()


@pytest.fixture
def test_file():
    return get_test_file_path()


@pytest.fixture
def nested_test_file():
    return get_nested_test_file_path()


@pytest.fixture
def test_dictionary():
    return {'alpha': 10, 'beta': 20, 'gamma': 20}


@pytest.fixture
def patch_get_history_file(monkeypatch):
    def mock_get_history_file_path():
        return get_test_file_path()

    monkeypatch.setattr(command_util.settings, 'get_history_file_path', mock_get_history_file_path)


@pytest.fixture
def fixture_cleanup_test_file(test_file):
    delete_file(test_file)
    yield 'fixture_cleanup_test_file'
    delete_file(test_file)


@pytest.fixture
def fixture_cleanup_nested_test_file(nested_test_file):
    delete_file(nested_test_file)
    delete_folder(nested_test_file.parent)
    yield 'fixture_cleanup_nested_test_file'
    delete_file(nested_test_file)
    delete_folder(nested_test_file.parent)


def test_write_file(test_file, fixture_cleanup_test_file):
    """Tests functionality for writing text to file. Ensures:

        - Text is written to file
        - The file is overwritten if flag is set
        - The file is not overwritten if flag is not set
    """

    # Write text to file
    text = 'Sample text'
    command_util.write_file(text, test_file)

    with test_file.open(mode='r') as f:
        content = f.read()

    assert content == 'Sample text'

    # Overwrite file
    text = 'Other sample text'
    command_util.write_file(text, test_file, overwrite=True)

    with test_file.open(mode='r') as f:
        content = f.read()

    assert content == 'Other sample text'

    # Refuse to overwrite file
    text = 'Original sample text'
    command_util.write_file(text, test_file)

    with test_file.open(mode='r') as f:
        content = f.read()

    assert content != 'Original sample text'


def test_write_json_file(test_dictionary, nested_test_file, fixture_cleanup_nested_test_file):
    """Tests writing JSON data to file. Ensures:

        - A folder path to the file is created if it doesn't exist
        - JSON is written to the correct file
    """

    command_util.write_json_file(test_dictionary, nested_test_file)

    assert nested_test_file.exists()

    with nested_test_file.open(mode='r') as f:
        content = json.load(f)

    assert content == test_dictionary


def test_load_json_file(test_dictionary, test_file, fixture_cleanup_test_file):
    """Tests loading JSON data from file. Ensures:

        - JSON is loaded from file
        - FileNotFoundError is raised
        - JSONDecodeError is raised
    """

    # FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        command_util.load_json_file(test_file)

    # JSON is loaded from file
    with test_file.open(mode='w') as f:
        json.dump(test_dictionary, f, indent=4)

    content = command_util.load_json_file(test_file)

    assert content == test_dictionary

    # JSONDecodeError is raised
    with test_file.open(mode='w') as f:
        f.write('This is not JSON data')

    with pytest.raises(json.decoder.JSONDecodeError):
        command_util.load_json_file(test_file)


def test_write_yaml_file(test_dictionary, nested_test_file, fixture_cleanup_nested_test_file):
    """Tests writing Yaml data to file. Ensures:

        - A folder path to the file is created if it doesn't exist
        - Yaml is written to the correct file
    """

    command_util.write_yaml_file(test_dictionary, nested_test_file)

    assert nested_test_file.exists()

    with nested_test_file.open(mode='r') as f:
        content = yaml.safe_load(f)

    assert content == test_dictionary


def test_load_yaml_file(test_dictionary, test_file, fixture_cleanup_test_file):
    """Tests loading Yaml data from file. Ensures:

        - Yaml is loaded from file
        - FileNotFoundError is raised
    """

    # FileNotFoundError
    with pytest.raises(FileNotFoundError):
        command_util.load_yaml_file(test_file)

    # Yaml is loaded from file
    with test_file.open(mode='w') as f:
        yaml.dump(test_dictionary, f, indent=4)

    content = command_util.load_yaml_file(test_file)

    assert content == test_dictionary


def test_create_history_file(test_file, patch_get_history_file, fixture_cleanup_test_file):
    """Tests creating a new history file."""

    command_util.create_history_file()

    assert test_file.exists()

    with test_file.open(mode='r') as f:
        content = json.load(f)

    assert content['history'] == []


def test_load_history(test_file, patch_get_history_file, fixture_cleanup_test_file):
    """Tests loading history from file. Ensures:

        - If history file doesn't exist, it is created
        - The history dictionary is loaded from file
    """

    assert not test_file.exists()

    content = command_util.load_history()

    assert test_file.exists()
    assert content == {'history': []}


def test_load_last_history(test_file, patch_get_history_file, fixture_cleanup_test_file):
    """Tests loading the last training command from history. Ensures:

        - Loads the last command
        - CommandUtilError is raised if the history is empty
    """

    # CommandUtilError is raised
    with pytest.raises(command_util.CommandUtilError):
        command_util.load_last_history()

    # The last command is loaded
    history = {'history': [['second', 'command'], ['first', 'command']]}

    with test_file.open(mode='w') as f:
        json.dump(history, f, indent=4)

    assert command_util.load_last_history() == ['second', 'command']


def test_load_last_lines_from_file(test_file, fixture_cleanup_test_file):
    """Tests the last lines of a file are correctly read. Ensures that requesting more lines than exist in the file is gracefully handled.
    """

    lines = 'first line\nsecond line\nthird line\nfourth line'
    with test_file.open(mode='w') as f:
        f.write(lines)

    assert command_util.load_last_lines_from_file(test_file, 1) == ['fourth line']

    assert command_util.load_last_lines_from_file(test_file, 2) == ['third line', 'fourth line']

    assert command_util.load_last_lines_from_file(test_file, 5) == [
        'first line',
        'second line',
        'third line',
        'fourth line',
    ]
