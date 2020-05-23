"""
CLI application that wraps 'mlagents-learn' with more automation.

Features:
- Displays estimated time remaining in training run
- Optionally copies trained policies to another location after training finishes (for example, into a Unity project)

See readme.md for more information.
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


class TrainingRunInfo:
    def __init__(self):

        self.step = 0
        self.steps_remaining = 0
        self.max_steps = 0
        self.time_elapsed = 0
        self.time_remaining = 0
        self.mean_reward = 0
        self.exported_brains = []

        self.steps_regex = re.compile(r'Step: ([\d]+)\. ')
        self.time_regex = re.compile(r'Time Elapsed: ([\.\d]+) s')
        self.max_steps_regex = re.compile(r'max_steps:\t(.+)$')
        self.mean_reward_regex = re.compile(r'(Mean Reward: )([^ ]+)\. ')
        self.exported_brain_regex = re.compile(r'Exported (.*\.nn) file')

    def update_from_training_output(self, line):

        if self.max_steps == 0:
            match = self.max_steps_regex.search(line)
            if match:
                self.max_steps = int(float(match.group(1)))

        match = self.steps_regex.search(line)
        if match:
            self.step = int(match.group(1))
            self.steps_remaining = self.max_steps - self.step

        match = self.time_regex.search(line)
        if match:
            self.time_elapsed = float(match.group(1))

        match = self.mean_reward_regex.search(line)
        if match:
            self.mean_reward = float(match.group(2))

        match = self.exported_brain_regex.search(line)
        if match:
            self.exported_brains.append(Path(match.group(1)))

        if self.max_steps != 0 and self.step != 0:
            self.time_remaining = (self.time_elapsed / self.step) * self.steps_remaining
            self.time_remaining = max(self.time_remaining, 0.0)

    def line_has_time_elapsed(self, line):

        return self.time_regex.search(line) is not None


def main():

    configure_logging()

    if not common.is_pipenv_present():
        training_log.error(
            'No virtual environment is accessible by Pipenv from this directory, unable to run mlagents-learn'
        )
        sys.exit(1)

    argv = get_argvs()
    args = parse_args(argv)

    run_id = args.run_id
    training_info = TrainingRunInfo()

    command = [
        'pipenv',
        'run',
        'mlagents-learn',
        args.trainer_config_path,
        '--run-id',
        run_id,
    ] + args.args

    try:
        with Popen(
            command, stdout=sys.stderr, stderr=PIPE, bufsize=2, universal_newlines=True
        ) as p:

            training_log.info(f'{" ".join(command[2:])}')
            training_log.info('-' * 63)
            training_log.info(f'Initiating \'{run_id}\'')

            start_time = time.perf_counter()

            for line in p.stderr:
                # Print intercepted line so it is visible in the console
                line = line.rstrip()
                print(line)

                training_info.update_from_training_output(line)

                if training_info.line_has_time_elapsed(line):
                    print(
                        f'Estimated time remaining: {common.get_human_readable_duration(training_info.time_remaining)}'
                    )

    except KeyboardInterrupt:
        training_log.warning('KeyboardInterrupt, aborting')
        raise

    finally:
        training_log.info('-' * 63)
        if args.export_path:
            export_brains(training_info.exported_brains, Path(args.export_path))

        end_time = time.perf_counter()
        training_duration = common.get_human_readable_duration(end_time - start_time)

        training_log.info(f'\nTraining run \'{run_id}\' ended after {training_duration}')

        if p.returncode == 0:
            training_log.info('Training completed successfully')
        else:
            training_log.warning(
                f'Training was not completed successfully (error code {p.returncode})'
            )

        training_log.info(f'Final Mean Reward: {training_info.mean_reward}')
        training_log.info('-' * 63)
        logging.shutdown()


def get_argvs():

    return sys.argv[1:]


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
        '--export-path', type=str, help='Export trained policies to this path'
    )

    parser = argparse.ArgumentParser(
        prog='grimwrapper',
        description='CLI application that wraps mlagents-learn with automatic exporting of trained policies and exposes more training information in the console.',
        parents=[wrapper_parser],
    )

    parser.add_argument(
        'trainer_config_path', type=str, help='Configuration file that holds brain hyperparameters'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments passed on to mlagents-learn (ex. --debug, --load)',
    )

    wrapper_args, extra_args = wrapper_parser.parse_known_args(argv)
    return parser.parse_args(extra_args, wrapper_args)


def configure_logging():
    """Configures logging for a training session."""

    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'display': {'style': '{', 'format': '{message}'},
            'timestamp': {'style': '{', 'format': '[{asctime}][{levelname}] {message}'},
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
                'formatter': 'display',
            },
            'file': {'class': 'logging.FileHandler', 'filename': '', 'formatter': 'timestamp'},
        },
        'loggers': {'grimagents.training_wrapper': {'handlers': ['console', 'file']}},
        'root': {'level': 'INFO'},
    }

    log_file = settings.get_log_file_path()

    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log_config['handlers']['file']['filename'] = log_file
    logging.config.dictConfig(log_config)


def export_brains(brain_paths: list, export_path: Path):
    """Copies a list of brain files into a target folder.

    Parameters:
        brain_paths: list: A list of Path objects pointing each brain files that should be exported
        export_path: Path: A folder to export brains files into
    """

    training_log.info('Exporting brains:')

    if not export_path.exists():
        export_path.mkdir(parents=True, exist_ok=True)

    for brain in brain_paths:
        if not brain.exists():
            continue

        destination = export_path / brain.name
        destination.write_bytes(brain.read_bytes())

        training_log.info(f'\t{destination}')


if __name__ == '__main__':
    main()
