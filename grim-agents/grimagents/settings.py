"""Holds grimagents package settings."""

from pathlib import Path


# This is a relative path to the project's root folder from setting.py's location
_PROJECT_ROOT_RELATIVE_PATH = '..\\..'

# This is a relative path to the grim-agents log folder
_LOG_FOLDER_PATH = 'grim-agents\\logs'


def get_project_folder_absolute():
    """Returns the absolute path for the project's root folder."""
    relative_path = Path(__file__).parent / _PROJECT_ROOT_RELATIVE_PATH
    return relative_path.resolve()


def get_log_folder():
    """Returns log folder's path relative to the project's root folder."""
    return Path(_LOG_FOLDER_PATH)


def get_log_folder_absolute():
    """Returns log folder as an absolute path"""
    return get_project_folder_absolute() / get_log_folder()
