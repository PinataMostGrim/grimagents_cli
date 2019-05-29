#!/usr/bin/env python3
"""CLI application that wraps 'mlagents-learn' and facilitates training logging.

# - use command_util to launch an mlagents-learn process
# - log output to file
# - record the runtime of the training
# - parse args for '--run-id'
"""


import argparse
from datetime import datetime
import sys
import time

from pathlib import Path
from subprocess import Popen, PIPE
import settings as settings


_TRAINER_RELATIVE_PATH = 'grim-agents\\mock_trainer.py'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def main():
    args = parse_args(sys.argv[1:])

    cwd = settings.get_project_folder_absolute()
    trainer_path = _TRAINER_RELATIVE_PATH

    log_folder = settings.get_log_folder_absolute()
    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=True)

    run_id = Path(args.run_id)
    if not run_id.suffix == '.log':
        run_id = run_id.with_suffix('.log')

    log_path = log_folder / run_id.name

    # Note: As these arguments are being passed direction into popen,
    # the trainer path does not need to be enclosed in quotes to support
    # paths with spaces in them.
    command = ['pipenv', 'run', 'python', f"{trainer_path}"] + sys.argv[1:]
    print(' '.join(command))

    with Popen(
        command, stdout=PIPE, cwd=cwd, bufsize=1, universal_newlines=True
    ) as p, log_path.open('a') as f:

        started = f'Training started at {datetime.now().strftime(TIME_FORMAT)}'
        print(started)
        f.write(started)

        start_time = time.perf_counter()

        for line in p.stdout:
            print(line, end='')
            f.write(line)

        end_time = time.perf_counter()

        finished = f'Training finished at {datetime.now().strftime(TIME_FORMAT)} ({end_time - start_time} seconds)\n\n'
        print(finished)
        f.write(finished)

    print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(p.returncode)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='training_wrapper',
        description='CLI application that wraps mlagents-learn with logging',
    )

    # parser.add_argument('--log-path', type=str, help='Log-path help')
    parser.add_argument('--run-id', type=str, help='Run-id help', required=True)

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main()
