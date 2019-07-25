#!/usr/bin/env python3
"""CLI application that wraps 'mlagents-learn' with several quality of life improvements.

Features:
- Logs mlagents-learn output to file
- Optionally exports trained models to another location after training finishes (for example, into a Unity project)

Notes:
- training_wrapper can be executed as a stand-alone script for logging and export features
- Potentially works with Linux (untested)
- See readme.md for more documentation
"""

import argparse
import logging
import logging.config
import re
import sys
import time

from pathlib import Path
from subprocess import Popen, PIPE

import grimagents.settings as settings
import grimagents.common as common


training_log = logging.getLogger('grimagents.training_wrapper')


def main():

    args = parse_args(sys.argv[1:])
    run_id = args.run_id

    log_filename = args.log_filename if args.log_filename else run_id
    configure_logging(log_filename)

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
        with Popen(command, stdout=PIPE, bufsize=1, universal_newlines=True) as p:

            training_log.info(f'{" ".join(command[2:])}')
            training_log.info('-' * 63)
            training_log.info(f'Initiating \'{run_id}\'')

            start_time = time.perf_counter()

            for line in p.stdout:
                line = line.rstrip()
                training_log.info(line)

                match = brain_regex.search(line)
                if match:
                    exported_brains.append(match.group(1))

    except KeyboardInterrupt:
        training_log.warning('KeyboardInterrupt, aborting')
        raise

    finally:
        if args.export_path:
            export_brains(exported_brains, Path(args.export_path))

        end_time = time.perf_counter()
        training_duration = common.get_human_readable_duration(end_time - start_time)

        training_log.info(f'\nTraining run \'{run_id}\' ended after {training_duration}.')

        if p.returncode == 0:
            training_log.info('Training completed successfully.')
        else:
            training_log.warning(
                f'Training was not completed successfully. (error code {p.returncode})'
            )

        training_log.info('-' * 63)
        training_log.info('')
        logging.shutdown()


def export_brains(brains: list, export_path: Path):
    """Exports a list of trained policies into a folder."""

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
    """Builds a Namespace object with parsed arguments."""

    # It is important to keep command line argument in parity with mlagents-learn.
    # As intermixed parsing was not introduced into ArgParser until Python 3.7,
    # we need to separate parsing into two parsers to accomplish this.

    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument(
        '--run-id',
        metavar='<run-id>',
        type=str,
        default='ppo',
        help='Run id for the training session',
    )
    wrapper_parser.add_argument(
        '--export-path', type=str, help='Export trained models to this path'
    )
    wrapper_parser.add_argument(
        '--log-filename', type=str, help='Write log output to this file. Defaults to run-id.'
    )

    parser = argparse.ArgumentParser(
        prog='training_wrapper',
        description='CLI application that wraps mlagents-learn with logging to file and automatic exporting of trained policy.',
        parents=[wrapper_parser],
    )

    parser.add_argument(
        'trainer_config_path', type=str, help='Configuration file that holds brain hyperparameters'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments passed on to mlagents-learn (ex. --slow, --debug, --load)',
    )

    wrapper_args, extra_args = wrapper_parser.parse_known_args(argv)
    args = parser.parse_args(extra_args, wrapper_args)

    return args


def configure_logging(log_filename: str):
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
                "stream": "ext://sys.stdout",
                "formatter": "display",
            },
            "file": {"class": "logging.FileHandler", "filename": "", "formatter": "timestamp"},
        },
        "loggers": {"grimagents.training_wrapper": {"handlers": ["console", "file"]}},
        "root": {"level": "INFO"},
    }

    log_file = settings.get_log_file_path()

    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log_config['handlers']['file']['filename'] = log_file
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    main()
