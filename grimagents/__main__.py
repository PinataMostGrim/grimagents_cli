#!/usr/bin/env python3
"""CLI application that loads training arguments from a configuration file and sends
them to the grimagents.training_wrapper. This script aims to automate several
repetitive training tasks.

Features:
- Load training arguments from a configuration file
- Override loaded configuration arguments with command line arguments
- Quickly resume the last training run
- Optionally time-stamp the training run-id
- Optionally launch training in a new console window

See training_wrapper.py for its feature list and readme.md for more documentation.
"""

import argparse
import logging
import logging.config
import sys

from argparse import Namespace
from pathlib import Path

import grimagents.config as config_util
import grimagents.command_util as command_util
import grimagents.settings as settings
import grimagents.common as common

from grimagents.training_commands import TrainingWrapperCommand


main_log = logging.getLogger('grimagents.main')


class Command:
    def __init__(self):
        self.new_window = False
        self.show_command = True

    def execute(self, args: Namespace):
        self.dry_run = args.dry_run
        self.command = self.create_command(args)
        command_util.execute_command(
            self.command,
            new_window=self.new_window,
            show_command=self.show_command,
            dry_run=self.dry_run,
        )

    def create_command(self, args):
        return ['cmd', '/K', 'echo', self.__class__.__name__, repr(args)]


class ListTrainingOptions(Command):
    """Outputs mlagents-learn usage options."""

    def create_command(self, args):
        return ['pipenv', 'run', 'mlagents-learn', '--help']


class EditGrimConfigFile(Command):
    """Opens a grimagents configuration file for editing or creates one if
    a file does not already exist."""

    def execute(self, args):
        file_path = Path(args.edit_config)
        config_util.edit_grim_config_file(file_path)


class EditTrainerConfigFile(Command):
    """Opens a trainer configuration file for editing or creates one if
    a file does not already exist.
    """

    def execute(self, args):
        file_path = Path(args.edit_trainer_config)
        config_util.edit_trainer_configuration_file(file_path)


class EditCurriculumFile(Command):
    """Opens a curriculum file for editing or creates one if a file does
    not already exist.
    """

    def execute(self, args):
        file_path = Path(args.edit_curriculum)
        config_util.edit_curriculum_file(file_path)


class StartTensorboard(Command):
    """Starts a new instance of tensorboard server in a new terminal window."""

    def create_command(self, args):
        self.new_window = True
        log_dir = f'--logdir={settings.get_summaries_folder()}'
        return ['pipenv', 'run', 'tensorboard', log_dir]


class PerformTraining(Command):
    """Launches the training wrapper script with arguments loaded from a configuration file."""

    def execute(self, args: Namespace):

        self.show_command = False
        self.dry_run = args.dry_run
        self.new_window = args.new_window
        self.command = self.create_command(args)

        command_util.save_to_history(self.command)
        command_util.execute_command(
            self.command,
            new_window=self.new_window,
            show_command=self.show_command,
            dry_run=self.dry_run,
        )

    def create_command(self, args):

        config_path = Path(args.configuration_file)
        config = config_util.load_grim_config_file(config_path)

        training_command = TrainingWrapperCommand(config)
        training_command.apply_argument_overrides(args)
        training_command.set_additional_arguments(args.args)

        return training_command.get_command()


class ResumeTraining(Command):
    """Launches the training wrapper script with the arguments used
    by the last training command executed."""

    def create_command(self, args):

        self.show_command = False
        self.new_window = args.new_window

        command = command_util.load_last_history()

        if '--load' not in command:
            command.append('--load')
        if args.lesson:
            command.append('--lesson')
            command.append(str(args.lesson))

        return command


def main():

    args = parse_args(sys.argv[1:])

    if not common.is_pipenv_present():
        main_log.error(
            'No virtual environment is accessible by Pipenv from this directory, unable to run mlagents-learn'
        )
        return

    if args.list:
        ListTrainingOptions().execute(args)
    elif args.edit_config:
        EditGrimConfigFile().execute(args)
    elif args.edit_trainer_config:
        EditTrainerConfigFile().execute(args)
    elif args.edit_curriculum:
        EditCurriculumFile().execute(args)
    elif args.tensorboard_start:
        StartTensorboard().execute(args)
    elif args.resume:
        ResumeTraining().execute(args)
    else:
        PerformTraining().execute(args)


def parse_args(argv):
    """Builds a Namespace object out of parsed arguments."""

    # Parser for arguments that apply exclusively to the grimagents cli
    options_parser = argparse.ArgumentParser(add_help=False)
    options_parser.add_argument(
        '--list', '-l', action='store_true', help='List mlagents-learn training options'
    )
    options_parser.add_argument(
        '--edit-config',
        dest='edit_config',
        metavar='<file>',
        type=str,
        help='Open a grimagents configuration file for editing',
    )
    options_parser.add_argument(
        '--edit-trainer-config',
        metavar='<file>',
        type=str,
        help='Open a trainer configuration file for editing',
    )
    options_parser.add_argument(
        '--edit-curriculum', metavar='<file>', type=str, help='Open a curriculum file for editing'
    )
    options_parser.add_argument(
        '--new-window', '-w', action='store_true', help='Run process in a new console window'
    )
    options_parser.add_argument(
        '--tensorboard-start', '-s', action='store_true', help='Start tensorboard server'
    )
    options_parser.add_argument('--resume', '-r', action='store_true', help='Resume the last run')
    options_parser.add_argument(
        '--dry-run', '-n', action='store_true', help='Print command without executing'
    )

    # Parser for arguments that may override configuration values
    overrides_parser = argparse.ArgumentParser(add_help=False)
    overrides_parser.add_argument('--trainer-config', type=str)
    overrides_parser.add_argument('--env', type=str)
    overrides_parser.add_argument('--lesson', type=int)
    overrides_parser.add_argument('--run-id', type=str)
    overrides_parser.add_argument('--num-envs', type=int)
    overrides_parser.add_argument(
        '--inference',
        action='store_true',
        help='Load environment in inference mode instead of training',
    )

    graphics_group = overrides_parser.add_mutually_exclusive_group()
    graphics_group.add_argument('--graphics', action='store_true')
    graphics_group.add_argument('--no-graphics', action='store_true')

    timestamp_group = overrides_parser.add_mutually_exclusive_group()
    timestamp_group.add_argument(
        '--timestamp',
        action='store_true',
        help='Append timestamp to run-id. Overrides configuration setting.',
    )
    timestamp_group.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Do not append timestamp to run-id. Overrides configuration setting.',
    )

    # Parser for arguments that are passed on to the training wrapper
    parser = argparse.ArgumentParser(
        prog='grimagents',
        description='CLI application that wraps Unity ML-Agents with quality of life improvements.',
        parents=[options_parser, overrides_parser],
    )

    parser.add_argument(
        'configuration_file', type=str, help='Configuration file to extract training arguments from'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments applied to training (ex. --slow, --debug, --load)',
    )

    args, unparsed_args = options_parser.parse_known_args()
    args, unparsed_args = overrides_parser.parse_known_args(unparsed_args, args)

    if len(argv) == 0:
        parser.print_help()
        sys.exit(0)

    if len(unparsed_args) > 0:
        args = parser.parse_args(unparsed_args, args)

    return args


def configure_logging():
    """Configures logging for the grim-agents CLI."""

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
        "loggers": {
            "grimagents.main": {"handlers": ["console", "file"]},
            "grimagents.config": {"handlers": ["console", "file"]},
            "grimagents.command_util": {"handlers": ["console", "file"]},
        },
        "root": {"level": "INFO"},
    }

    log_file = settings.get_log_file_path()

    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log_config['handlers']['file']['filename'] = log_file
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    configure_logging()
    main()
    logging.shutdown()
