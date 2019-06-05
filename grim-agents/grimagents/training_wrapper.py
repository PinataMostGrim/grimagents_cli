#!/usr/bin/env python3
"""
CLI application that wraps 'mlagents-learn' with quality of life improvements.
- Initiates training with mlagents-learn
- Logs training output to file
- Optionally timestamps run-id
- Optionally exports trained models to another location

Notes:
- The training process is executed with the project's root folder set as the current working directory.
"""

import argparse
import logging
import logging.config
import re
import sys
import time

from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE

import settings as settings


training_log = logging.getLogger('training_wrapper')


def main():

    args = parse_args(sys.argv[1:])
    cwd = settings.get_project_folder_absolute()
    run_id = args.run_id

    if args.timestamp:
        timestamp = get_timestamp()
        run_id = f'{run_id}-{timestamp}'

    # Note: We use run_id from args in order to exclude the optional timestamp from the log file name.
    configure_log(args.run_id)

    brain_regex = re.compile(r'\A.*DONE: wrote (.*\.nn) file.')
    exported_brains = []

    # Note: As these arguments are being passed directly into popen,
    # the trainer path does not need to be enclosed in quotes to support
    # paths with spaces in them.
    command = [
        'pipenv',
        'run',
        'mlagents-learn',
        args.trainer_config_path,
        '--run-id',
        run_id,
    ] + args.args
    try:
        with Popen(command, stdout=PIPE, cwd=cwd, bufsize=1, universal_newlines=True) as p:

            training_log.info('==================================================')
            training_log.info(f'{" ".join(command[2:])}')
            training_log.info(f'Initiating training run \'{run_id}\'')

            start_time = time.perf_counter()

            for line in p.stdout:
                line = line.rstrip()
                training_log.info(line)

                match = brain_regex.search(line)
                if match:
                    exported_brains.append(match.group(1))

            end_time = time.perf_counter()

            training_log.info('')
            training_log.info(
                f'Training run \'{run_id}\' ended after {end_time - start_time:.0f} seconds'
            )

        if args.export_path:
            export_brains(exported_brains, Path(args.export_path))

    except KeyboardInterrupt:
        training_log.warning('KeyboardInterrupt, aborting')
        raise

    finally:
        if p.returncode == 0:
            training_log.info('Training completed successfully.')
        else:
            training_log.warning(
                f'Training was not completed successfully. (error code {p.returncode})'
            )

        training_log.info('==================================================')
        training_log.info('')
        logging.shutdown()


def get_timestamp():
    """Fetch the current time as a string.

    Returns:
      The current time as a string.
    """

    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def export_brains(brains: list, export_path: Path):
    """Exports a list of trained models to a path.

    Args:
      brains: list: Trained models to export.
      export_path: Path: Destination folder to export brains into.
    """

    training_log.info('Exporting brains')

    if not export_path.exists():
        export_path.mkdir(parents=True, exist_ok=True)

    for brain in brains:
        source = Path(brain)
        if not source.exists():
            continue

        destination = export_path / source.name
        destination.write_bytes(source.read_bytes())

        training_log.info(f'Exported {destination}')


def parse_args(argv):
    """Builds a Namespace object with parsed arguments.

    Args:
      argv: list: Arguments to parse.

    Returns:
      A Namespace object containing parsed arguments.
    """

    # It is important to keep command line argument in parity with mlagents-learn.
    # As intermixed parsing was not introduced into ArgParser until Python 3.7,
    # we need to separate parsing into two parsers to accomplish this.

    # The issue with ArgParse in Python 3.6 is that optional arguments will get
    # collected by argparse.REMAINDER if they are not placed before positional
    # arguments. As this places an idiosyncratic restriction on the wrapper's
    # command line argument positioning, another solution needs to be found.

    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument(
        '--run-id',
        metavar='<run-id>',
        type=str,
        default='ppo',
        help='Run id for the training session',
    )
    wrapper_parser.add_argument(
        '--timestamp',
        action='store_true',
        help='Append a timestamp to the run-id. Timestamp will not be applied to log file name.',
    )
    wrapper_parser.add_argument(
        '--export-path', type=str, help='Export trained models to this path'
    )

    parser = argparse.ArgumentParser(
        prog='training_wrapper',
        description='CLI application that wraps mlagents-learn with quality of life improvements.',
        parents=[wrapper_parser],
    )

    parser.add_argument(
        'trainer_config_path', type=str, help='Configuration file that holds brain hyperparameters.'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments passed on to mlagents-learn (ex. --slow, --debug, --load).',
    )

    wrapper_args, extra_args = wrapper_parser.parse_known_args(argv)
    args = parser.parse_args(extra_args, wrapper_args)

    return args


def configure_log(run_id: str):
    """Configures logging for a training session.

    Args:
      run_id: str: The training session's run-id
    """

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
