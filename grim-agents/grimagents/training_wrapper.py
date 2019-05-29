#!/usr/bin/env python3
"""CLI application that wraps 'mlagents-learn' and facilitates training logging."""

# - use command_util to launch an mlagents-learn process
# - log output to file
# - record the runtime of the training
# - parse args for '--run-id'

import argparse
import time
import sys

from argparse import Namespace
from pathlib import Path
from subprocess import Popen, PIPE
import settings as settings


_TRAINER_RELATIVE_PATH = '..\\mock_trainer.py'


def main():
    # args = parse_args(sys.argv[1:])
    # log_path = Path(args.log_path)
    # run_id = args.run_id

    print(settings.get_project_root_folder())
    print(type(settings.get_project_root_folder()))
    return

    trainer_path = Path(__file__).parent / _TRAINER_RELATIVE_PATH
    log_path = Path('mock_training.log')

    # Note: As these arguments are being passed direction into popen,
    # the trainer path does not need to be enclosed in quotes to support
    # paths with spaces in them.

    # Note: We use absolute path here for the trainer script as we
    # do not know what the current working directory will be.
    command = ['pipenv', 'run', 'python', f"{trainer_path.resolve()}"] + sys.argv[1:]
    print(' '.join(command))

    with Popen(command, stdout=PIPE, bufsize=1, universal_newlines=True) as p, \
            log_path.open('w') as f:

        start_time = time.perf_counter()

        for line in p.stdout:
            print(line, end='')
            f.write(line)

        end_time = time.perf_counter()
        print(f'Training finished in {end_time - start_time} seconds.')

    print('━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(p.returncode)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='training_wrapper',
        description='CLI application that wraps mlagents-learn with logging')

    parser.add_argument('--log-path', type=str, help='Log-path help')
    parser.add_argument('--run-id', type=str, help='Run-id help')

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main()
