"""Facilitates executing system commands and opening files."""

import json
import logging
import os
import subprocess

from io import StringIO
from pathlib import Path
from subprocess import PIPE, CREATE_NEW_CONSOLE

from . import settings as settings


TRAINING_HISTORY_COUNT = 10


command_log = logging.getLogger('grimagents.command_util')


class CommandUtilError(Exception):
    pass


def open_file(file_path: Path):
    """Opens a file using the default system application.

    Args:
      file_path: Path: The file to open.
    """

    command_log.info(f'Opening \'{file_path}\'')

    try:
        # Note: Open file in Windows
        os.startfile(str(file_path))
    except AttributeError:
        # Note: Open file in OSX / Linux
        command = ['open', str(file_path)]
        execute_command(command)




def write_json_file(json_data, file_path: Path):

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    command_log.info(f'Creating file \'{file_path}\'')
    with file_path.open(mode='w') as f:
        json.dump(json_data, f, indent=4)


def load_json_file(file_path: Path):

    try:
        with file_path.open('r') as f:
            data = json.load(f)
    except FileNotFoundError as exception:
        command_log.error(f'File \'{file_path}\' not found')
        raise exception

    return data


def execute_command(command: list, cwd=None, new_window=False, show_command=True):
    """Executes a command in terminal. Optionally opens a new window or
    echos the provided command."""

    if show_command:
        command_log.info(' '.join(command))

    if new_window:
        command = ['cmd', '/K'] + command
        subprocess.Popen(command, cwd=cwd, creationflags=CREATE_NEW_CONSOLE)
    else:
        subprocess.run(command, cwd=cwd)


def execute_command_and_capture(command: list, cwd=None, show_command=True):
    """Executes a command in terminal, captures output,
    prints it, and returns it.
    """

    if show_command:
        command_log.info(' '.join(command))

    with subprocess.Popen(
        command, cwd=cwd, stdout=PIPE, bufsize=1, universal_newlines=True
    ) as p, StringIO() as buf:

        for line in p.stdout:
            command_log.info(line, end='')
            buf.write(line)

        output = buf.getvalue()

    return output


def create_history_file():
    """Creates or overwrites the training command history file."""

    history_file = settings.get_history_file_path()
    history = {"history": []}
    write_json_file(history, history_file)


def load_history():
    """Loads a list of training commands executed from the history file.

    Returns:
      The training command history list.
    """

    history_file = settings.get_history_file_path()

    if not history_file.exists():
        create_history_file()

    return load_json_file(history_file)


def save_to_history(command: list):
    """Saves a training command to the history file.

    Args:
      command: list: Training command to save.
    """

    history_file = settings.get_history_file_path()
    dict = load_history()

    while len(dict['history']) >= TRAINING_HISTORY_COUNT:
        dict['history'].pop()

    dict['history'].insert(0, command)

    with history_file.open(mode='w') as file:
        json.dump(dict, file, indent=4)


def load_last_history():
    """Loads the last training command executed from the history file.

    Returns:
      The last training command executed as a list of arguments.

    Raises:
      CommandUtilError: The history file is empty.
    """

    dict = load_history()
    try:
        return dict['history'][0]
    except IndexError:
        command_log.error('History file is empty')
        raise CommandUtilError('History file is empty')
