import asyncio
import sys
import os

from asyncio.subprocess import PIPE
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

# from hard_pi
# - prints stdout to terminal, displays stream in realtime
# def execute_command(command: list):
#     """
#     - executes command
#     - does not consume stderr

#     source: https://stackoverflow.com/a/17698359
#     """
#     print(' '.join(command))

#     with Popen(
#         command, stdout=PIPE, bufsize=1, universal_newlines=True
#     ) as p:
#         for line in p.stdout:
#             print(line, end='')

#     print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
#     print(p.returncode)


# from stack exchange (jfs)
# - prints stdout to terminal, displays stream in realtime, and saves to variable
def execute_command(command: list):
    """
    - executes a command
    - prints stdout to terminal
    - saves terminal output to variable

    source: https://stackoverflow.com/a/25755038
    """
    print(' '.join(command))

    with Popen(command, stdout=PIPE, bufsize=1, universal_newlines=True) as p, StringIO() as buf:
        for line in p.stdout:
            print(line, end='')
            buf.write(line)
        output = buf.getvalue()

    print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(p.returncode)

    return output


# from stack exchange
# - currently not working
# def execute_commands(commands: str):
#     """
#     source: https://stackoverflow.com/a/43697819
#     """

#     with Popen(
#         "cmd.exe", shell=False, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE
#     ) as p:
#         out, err = p.communicate(commands)

#         print(out.encode('utf-8'))

#         # for line in p.stdout:
#         #     print(line, end='')

#     print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
#     print(p.returncode)


def execute_command_shell(command: str):
    """
    - executes a command
    - expects commands in the format of a newline delimited string
    """
    print(command)

    with Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True) as p:

        for line in p.stdout:
            print(line, end='')

    print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(p.returncode)


# from stack exchange (jfs)
# - captures and displays both stdout and stderr as they appear and displays in realtime
# def execute_command(command: list):
#     """
#     - executes command
#     source: https://stackoverflow.com/questions/17190221/subprocess-popen-cloning-stdout-and-stderr-both-to-terminal-and-variables/25960956#25960956
#     """

#     print(' '.join(command))

#     cmd = tuple(command)
#     # run the event loop
#     if os.name == 'nt':
#         loop = asyncio.ProactorEventLoop()  # for subprocess' pipes on Windows
#         asyncio.set_event_loop(loop)
#     else:
#         loop = asyncio.get_event_loop()
#     rc, *output = loop.run_until_complete(read_and_display(*cmd))
#     loop.close()


# @asyncio.coroutine
# def read_and_display(*cmd):
#     """Capture cmd's stdout, stderr while displaying them as they arrive
#     (line by line).

#     """
#     # start process
#     process = yield from asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)

#     # read child's stdout/stderr concurrently (capture and display)
#     try:
#         stdout, stderr = yield from asyncio.gather(
#             # using sys.stdout.buffer.write circumvents realtime output of text.
#             # read_stream_and_display(process.stdout, sys.stdout.buffer.write),
#             # read_stream_and_display(process.stderr, sys.stderr.buffer.write),
#             read_stream_and_display(process.stdout, sys.stdout.write),
#             read_stream_and_display(process.stderr, sys.stderr.write),
#         )
#     except Exception:
#         process.kill()
#         raise
#     finally:
#         # wait for the process to exit
#         rc = yield from process.wait()
#     return rc, stdout, stderr


# @asyncio.coroutine
# def read_stream_and_display(stream, display):
#     """Read from stream line by line until EOF, display, and capture the lines.

#     """
#     output = []
#     while True:
#         line = yield from stream.readline()
#         if not line:
#             break
#         output.append(line)
#         # a string is required for the std.stdout.write method, whereas sys.stdout.buffer.write can accept bytes.
#         # display(line)  # assume it doesn't block
#         display(line.decode('utf-8'))  # assume it doesn't block
#     return b''.join(output)
