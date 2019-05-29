"""Holds grimagents package settings."""

from pathlib import Path


_PROJECT_ROOT_RELATIVE_PATH = '..\\..'


def get_project_root_folder():
    """Returns the absolute path for the project's root folder."""
    relative_path = Path(__file__).parent / _PROJECT_ROOT_RELATIVE_PATH
    return relative_path.resolve()
