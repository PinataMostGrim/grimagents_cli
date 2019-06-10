"""Holds grimagents package settings."""

from pathlib import Path


# Relative path to the project's root folder from setting.py's location.
_PROJECT_ROOT_RELATIVE_PATH = '..\\..'

# Project relative path to the summaries folder.
_SUMMARIES_RELATIVE_PATH = 'summaries'

# Relative path to the grim-agents log folder.
_LOG_FOLDER_PATH = 'grim-agents\\logs'

# Relative path to the grim-agents trainer wrapper script.
_TRAINING_WRAPPER_PATH = 'grim-agents\\grimagents\\training_wrapper.py'


def get_project_folder_absolute():
    """Returns the absolute path for the project's root folder."""

    relative_path = Path(__file__).parent / _PROJECT_ROOT_RELATIVE_PATH
    return relative_path.resolve()


def get_summaries_folder():
    """Returns project relative path to the summaries folder"""

    return Path(_SUMMARIES_RELATIVE_PATH)


def get_log_folder():
    """Returns log folder's path relative to the project's root folder."""

    return Path(_LOG_FOLDER_PATH)


def get_log_folder_absolute():
    """Returns log folder as an absolute path."""

    return get_project_folder_absolute() / get_log_folder()


def get_training_wrapper_path():
    """Returns project relative path to the training wrapper."""

    return Path(_TRAINING_WRAPPER_PATH)
