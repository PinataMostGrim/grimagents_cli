#!/usr/bin/env python3
"""CLI application that wraps 'mlagents-learn' and logs training output.

Executes training with the current working directory set to the the project's root folder.
"""

import argparse
import logging
import logging.config
import sys
import time

from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE

import settings as settings


_TRAINER_RELATIVE_PATH = 'grim-agents\\mock_trainer.py'


def main():

    args = parse_args(sys.argv[1:])
    cwd = settings.get_project_folder_absolute()
    run_id = args.run_id

    if args.timestamp:
        timestamp = get_timestamp()
        run_id = f'{run_id}-{timestamp}'

    # We use run_id from args in order to exclude the optional timestamp from the log file name.
    configure_log(args.run_id)
    training_log = logging.getLogger('training_wrapper')

    # Note: As these arguments are being passed directly into popen,
    # the trainer path does not need to be enclosed in quotes to support
    # paths with spaces in them.

    # trainer_path = _TRAINER_RELATIVE_PATH
    # command = ['pipenv', 'run', 'python', f"{trainer_path}", args.trainer_config_path, '--run-id', run_id] + args.args

    command = ['pipenv', 'run', 'mlagents-learn', args.trainer_config_path, '--run-id', run_id] + args.args
    try:
        with Popen(command, stdout=PIPE, cwd=cwd, bufsize=1, universal_newlines=True) as p:

            training_log.info('==================================================')
            training_log.info(f'{" ".join(command[2:])}')
            training_log.info(f'Initiating training run \'{run_id}\'')

            start_time = time.perf_counter()

            for line in p.stdout:
                training_log.info(line.rstrip())

            end_time = time.perf_counter()

            training_log.info('')
            training_log.info(
                f'Training run \'{run_id}\' ended after {end_time - start_time:.0f} seconds'
            )

    except KeyboardInterrupt:
        training_log.warning('KeyboardInterrupt, aborting')
        raise

    finally:
        if p.returncode == 0:
            training_log.info('Training completed successfully.')
        else:
            training_log.warning(f'Training was not completed successfully. (error code {p.returncode})')

        training_log.info('==================================================')
        logging.shutdown()


def get_timestamp():
    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def parse_args(argv):

    # It is important to keep command line argument parity with mlagents-learn.
    # As intermixed parsing was not introduced into ArgParser until Python 3.7,
    # we need to separate parsing into two parsers to accomplish this.

    # The issue with ArgParse in Python 3.6 is that optional arguments will get
    # collected by argparse.REMAINDER if they are not placed before positional
    # arguments. As this places an idiosyncratic restriction on the wrapper's
    # command line argument positioning, another solution needs to be found.

    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument(
        '--run-id', metavar='<run-id>', type=str, default='ppo', help='Run-id help'
    )
    wrapper_parser.add_argument('--timestamp', action='store_true', help='Timestamp help')
    # wrapper_parser.add_argument('--log-path', type=str, help='Log-path help')

    parser = argparse.ArgumentParser(
        prog='training_wrapper',
        description='CLI application that wraps mlagents-learn with logging',
        parents=[wrapper_parser],
    )

    parser.add_argument('trainer_config_path', type=str, help='Trainer config path help')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Additional arguments help')

    wrapper_args, extra_args = wrapper_parser.parse_known_args(argv)
    args = parser.parse_args(extra_args, wrapper_args)

    return args


def configure_log(run_id: str):
    """Configures logging for a training session."""

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "display": {"style": "{", "format": "{message}"},
            "timestamp": {"style": "{", "format": "[{asctime}][{levelname}] {message}"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "display",
            },
            "file": {"class": "logging.FileHandler", "filename": "", "formatter": "timestamp"},
        },
        "loggers": {"training_wrapper": {"handlers": ["console", "file"]}},
        "root": {"level": "INFO"},
    }

    log_folder = settings.get_log_folder_absolute()
    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=True)

    run_id = Path(run_id)
    if not run_id.suffix == '.log':
        run_id = run_id.with_suffix('.log')

    log_path = log_folder / run_id.name

    log_config['handlers']['file']['filename'] = log_path
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    main()
