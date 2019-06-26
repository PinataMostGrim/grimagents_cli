"""CLI application that loads training arguments from a configuration file and sends
them to the mlagents-learn training wrapper. This script aims to automate several
repetitive training tasks.

Features:
- Load training arguments from a configuration file
- Override loaded configuration arguments with command line arguments
- Easily resume the last training run
- Optionally time-stamp the training run-id
- Optionally launch training in a new console window

See training_wrapper.py for its feature list.

Requirements:
- Windows
- Pipenv accessible through the PATH environment variable
- Virtual environment setup for the MLAgents project
"""

import argparse
import logging
import logging.config
import sys

from argparse import Namespace
from pathlib import Path

from . import config as config_util
from . import command_util as command_util
from . import helpers as helpers
from . import settings as settings


class Command:
    def __init__(self):
        self.cwd = settings.get_project_folder_absolute()
        self.new_window = False
        self.show_command = True

    def execute(self, args: Namespace):
        self.command = self.create_command(args)
        command_util.execute_command(
            self.command, self.cwd, new_window=self.new_window, show_command=self.show_command
        )

    def create_command(self, args):
        return ['cmd', '/K', 'echo', self.__class__.__name__, repr(args)]


class ListTrainingOptions(Command):
    """Outputs mlagents-learn usage options."""

    def create_command(self, args):
        return ['pipenv', 'run', 'mlagents-learn', '--help']


class EditGrimConfigFile(Command):
    """Opens a GrimAgents configuration file for editing."""

    def execute(self, args):
        config_path = Path(args.edit_config)
        config_util.edit_grim_config_file(config_path)


class StartTensorboard(Command):
    """Starts a new instance of tensorboard server in a new terminal window."""

    def create_command(self, args):
        self.new_window = True
        log_dir = f'--logdir={settings.get_summaries_folder()}'
        return ['pipenv', 'run', 'tensorboard', log_dir]


class PerformTraining(Command):
    """Launches the training wrapper script with arguments loaded from a configuration file."""

    def execute(self, args: Namespace):
        self.command = self.create_command(args)

        command_util.save_to_history(self.command)
        command_util.execute_command(
            self.command, self.cwd, new_window=self.new_window, show_command=self.show_command
        )

    def create_command(self, args):

        self.show_command = False
        trainer_path = settings.get_training_wrapper_path()
        config_path = Path(args.configuration_file)
        config = config_util.load_config_file(config_path)
        config = self.override_configuration_values(config, args)

        if config_util.get_timestamp_enabled(config):
            run_id = config_util.get_run_id(config)
            # Note: If a log filename isn't configured, we explicitly set it
            # to the run_id before appending a timestamp to reduce the
            # number of log files being generated.
            if not config_util.get_log_filename(config):
                config = config_util.set_log_filename(run_id, config)

            timestamp = helpers.get_timestamp()
            run_id = f'{run_id}-{timestamp}'
            config = config_util.set_run_id(run_id, config)

        training_arguments = config_util.get_training_arguments(config)
        return (
            ['pipenv', 'run', 'python', str(trainer_path)]
            + training_arguments
            + args.args
            + ['--train']
        )

    def override_configuration_values(self, configuration: dict, args: Namespace):
        """Replaces values in the configuration dictionary with those stored in args."""

        if args.env is not None:
            configuration = config_util.set_env(args.env, configuration)
        if args.lesson is not None:
            configuration = config_util.set_lesson(str(args.lesson), configuration)
        if args.run_id is not None:
            configuration = config_util.set_run_id(args.run_id, configuration)
        if args.num_envs is not None:
            configuration = config_util.set_num_envs(str(args.num_envs), configuration)

        if args.graphics:
            # Note: As the argument is 'no-graphics', false in this case means
            # graphics are used.
            configuration = config_util.set_no_graphics_enabled(False, configuration)
        if args.no_graphics:
            configuration = config_util.set_no_graphics_enabled(True, configuration)

        if args.timestamp:
            configuration = config_util.set_timestamp_enabled(True, configuration)
        if args.no_timestamp:
            configuration = config_util.set_timestamp_enabled(False, configuration)

        return configuration


class ResumeTraining(Command):
    """Launches the training wrapper script with the arguments used
    by the last training command executed."""

    def create_command(self, args):

        self.show_command = False
        command = command_util.load_last_history()

        if '--timestamp' in command:
            command.remove('--timestamp')
        if '--load' not in command:
            command.append('--load')

        return command


def main():

    args = parse_args(sys.argv[1:])

    if args.list:
        ListTrainingOptions().execute(args)
    elif args.edit_config:
        EditGrimConfigFile().execute(args)
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
        '--list', action='store_true', help='List mlagents-learn training options'
    )
    options_parser.add_argument(
        '--edit-config',
        dest='edit_config',
        metavar='FILE',
        type=str,
        help='Open a grimagents configuration file for editing',
    )
    options_parser.add_argument(
        '--new-window', action='store_true', help='Run training process in a new console window'
    )
    options_parser.add_argument(
        '--tensorboard-start', action='store_true', help='Start tensorboard server'
    )
    options_parser.add_argument(
        '--resume', action='store_true', help='Resume the last training run'
    )

    # Parser for arguments that may override configuration values
    overrides_parser = argparse.ArgumentParser(add_help=False)
    overrides_parser.add_argument('--env', type=str)
    overrides_parser.add_argument('--lesson', type=int)
    overrides_parser.add_argument('--run-id', type=str)
    overrides_parser.add_argument('--num-envs', type=int)

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
        prog='grim-agents',
        description='CLI application that wraps Unity ML-Agents with some quality of life improvements.',
        parents=[options_parser, overrides_parser],
    )

    parser.add_argument(
        'configuration_file', type=str, help='Configuration file to load training arguments from'
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
    """Configures logging for the grim-agents CLI.
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
                "stream": "ext://sys.stdout",
                "formatter": "display",
            },
            "file": {"class": "logging.FileHandler", "filename": "", "formatter": "timestamp"},
        },
        "loggers": {
            "grimagents.config": {"handlers": ["console", "file"]},
            "grimagents.command_util": {"handlers": ["console", "file"]},
        },
        "root": {"level": "INFO"},
    }

    log_folder = settings.get_log_folder_absolute()
    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=True)

    log_path = log_folder / 'grim-agents.log'

    log_config['handlers']['file']['filename'] = log_path
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    configure_logging()
    main()
    logging.shutdown()
