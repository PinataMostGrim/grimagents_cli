#!/usr/bin/env python3
"""CLI application that wraps 'mlagents-learn' with more automation.

Features:
- Optionally copies trained models to another location after training finishes (for example, into a Unity project)

Notes:
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

    configure_logging()

    if not common.is_pipenv_present():
        training_log.error(
            'No virtual environment is accessible by Pipenv from this directory, unable to run mlagents-learn'
        )
        return

    args = parse_args(sys.argv[1:])
    run_id = args.run_id

    exported_brain_regex = re.compile(r'Exported (.*\.nn) file')
    exported_brains = []

    mean_reward_regex = re.compile(r"(Mean Reward: )([^ ]+)\. ")
    mean_reward = 0

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
        with Popen(command, stdout=sys.stderr, stderr=PIPE, bufsize=2, universal_newlines=True) as p:

            training_log.info(f'{" ".join(command[2:])}')
            training_log.info('-' * 63)
            training_log.info(f'Initiating \'{run_id}\'')

            start_time = time.perf_counter()

            for line in p.stderr:
                # Print intercepted line so it is visible in the console
                line = line.rstrip()
                print(line)

                # Store last mean reward
                match = mean_reward_regex.search(line)
                if match:
                    mean_reward = match.group(2)

                # Search for exported brains
                match = exported_brain_regex.search(line)
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

        training_log.info(f'\nTraining run \'{run_id}\' ended after {training_duration}')

        if p.returncode == 0:
            training_log.info('Training completed successfully')
        else:
            training_log.warning(
                f'Training was not completed successfully (error code {p.returncode})'
            )

        training_log.info(f'Final Mean Reward: {mean_reward}')
        training_log.info('-' * 63)
        logging.shutdown()


def parse_args(argv):
    """Builds a Namespace object with parsed arguments."""

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

    parser = argparse.ArgumentParser(
        prog='grimwrapper',
        description='CLI application that wraps mlagents-learn with automatic exporting of trained policies.',
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


def configure_logging():
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


def export_brains(brains: list, export_path: Path):
    """Exports a list of trained policies into a folder."""

    training_log.info('Exporting brains:')

    if not export_path.exists():
        export_path.mkdir(parents=True, exist_ok=True)

    for brain in brains:
        source = Path(brain)
        if not source.exists():
            continue

        destination = export_path / source.name
        destination.write_bytes(source.read_bytes())

        training_log.info(f'Exported {destination}')


if __name__ == '__main__':
    main()
