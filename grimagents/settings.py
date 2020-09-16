"""Holds grimagents path settings."""

from pathlib import Path


def get_summaries_folder():
    """Returns absolute path to the summaries folder."""

    return (Path(__file__).parent / '../../results').resolve()


def get_log_file_path():
    """Returns absolute path to the log folder."""

    return (Path(__file__).parent / '../logs/grimagents.log').resolve()


def get_training_wrapper_path():
    """Returns path to the training wrapper."""

    return Path(__file__).parent / 'training_wrapper.py'


def get_history_file_path():
    """Returns path to the training command history file."""

    return Path(__file__).parent / 'history'
