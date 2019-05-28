#!/usr/bin/env python3

import asyncio
import sys
import os

from asyncio.subprocess import PIPE
from io import StringIO
from subprocess import Popen, PIPE

# start cmd /K pipenv run mlagents-learn config\mkv_config.yaml --run-id=MKV_04 --train
# /C carry out command and close
# /K carry out command and remain


# from subtreeutil
# - prints stdout upon completion, buffers in memory
# def execute_command(command: list):
#     """
#     - executes command
#     - prints output once execution is complete
#     - does not handle large output very well as it is buffered in memory
#     """

#     print(' '.join(command))

#     with Popen(command, stdout=PIPE, stderr=PIPE) as process:
#         o, e = process.communicate()

#     if o:
#         print(o.decode('utf-8'))
#     if e:
#         print(e.decode('utf-8'))

#     print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
#     print(p.returncode)


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


def main():
    # command = ['cmd', '/K', 'python', 'mock_trainer.py']
    # command = ['pipenv', 'run', 'python', 'mock_trainer.py']
    command = ['python', 'mock_trainer.py']

    execute_command(command)


if __name__ == '__main__':
    main()
