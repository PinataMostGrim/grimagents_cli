"""Facilitates executing system commands and opening files."""

import logging
import os
import subprocess

from io import StringIO
from pathlib import Path
from subprocess import PIPE, CREATE_NEW_CONSOLE


command_log = logging.getLogger('grimagents.command_util')


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

    source: https://stackoverflow.com/a/25755038
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
