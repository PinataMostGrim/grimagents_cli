"""Holds grimagents package settings."""

from pathlib import Path


# Relative path from settings.py to the MLAgents project root folder.
_PROJECT_ROOT_RELATIVE_PATH = '..\\..'

# Relative path from project root to the summaries folder.
_SUMMARIES_RELATIVE_PATH = 'summaries'

# Relative path from project root to the log folder.
_LOG_FOLDER_PATH = 'grim-agents\\logs'

# Relative path from project root to the file that stores the last run command.
_RUN_HISTORY_FILE = 'grim-agents\\grimagents\\history'


def get_project_folder_absolute():
    """Returns an absolute path for the MLAgents' project root folder."""

    relative_path = Path(__file__).parent / _PROJECT_ROOT_RELATIVE_PATH
    return relative_path.resolve()


def get_summaries_folder():
    """Returns project relative path to the summaries folder."""

    return Path(_SUMMARIES_RELATIVE_PATH)


def get_log_folder():
    """Returns log folder's path relative to the project's root folder."""

    return Path(_LOG_FOLDER_PATH)


def get_log_folder_absolute():
    """Returns log folder as an absolute path."""

    return get_project_folder_absolute() / get_log_folder()


def get_training_wrapper_path():
    """Returns absolute path to the training wrapper."""

    return Path(__file__).parent / 'training_wrapper.py'


def get_history_file_path():
    return get_project_folder_absolute() / _RUN_HISTORY_FILE
