"""Facilitates executing system commands and opening files."""

import os

from io import StringIO
from pathlib import Path
from subprocess import Popen, PIPE


def open_file(file_path: Path):
    """Opens a file using the default system application.

    Args:
      file_path: Path: The file to open.
    """

    print(f'Opening \'{file_path}\'')
    try:
        # Note: Open file in Windows
        os.startfile(str(file_path))
    except AttributeError:
        # Note: Open file in OSX / Linux
        command = ['open', str(file_path)]
        execute_command(command)


def execute_command(command: list, cwd=None):
    """Executes a command in terminal, captures its output,
    prints it, and returns it.

    source: https://stackoverflow.com/a/25755038
    """

    print(' '.join(command))

    with Popen(command, cwd=cwd, stdout=PIPE, bufsize=1, universal_newlines=True) as p, StringIO() as buf:

        for line in p.stdout:
            print(line, end='')
            buf.write(line)

        output = buf.getvalue()

    # print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
    # print(p.returncode)

    return output
