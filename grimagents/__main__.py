"""CLI application that loads training arguments from a configuration file and sends
them to the mlagents-learn training wrapper. This script aims to automate several
repetitive training tasks.

- Load training arguments from a configuration file
- Optionally override loaded configuration arguments with command line arguments
- Optionally time-stamp the training run-id
- Optionally launch training in a new console window
- Resume the last training run

See training_wrapper for the features it provides.

Requirements:
- Windows
- Pipenv and a virtual environment setup for the MLAgents project
"""

import argparse
import logging
import logging.config
import sys

from argparse import Namespace
from pathlib import Path

from . import config as config_util
from . import command_util as command_util
from . import settings as settings


def list_training_options():
    """Outputs mlagents-learn usage options."""

    cwd = settings.get_project_folder_absolute()
    command = ['pipenv', 'run', 'mlagents-learn', '--help']
    command_util.execute_command(command, cwd)


def edit_config_file(args):
    """Opens a configuration file for editing.

    Args:
      args: Namespace: A Namespace object containing parsed arguments.
    """

    config_path = Path(args.edit_config)
    config_util.edit_config_file(config_path)


def start_tensorboard(args):
    """Starts a new instance of tensorboard server in a new terminal window."""

    cwd = settings.get_project_folder_absolute()
    log_dir = f'--logdir={settings.get_summaries_folder()}'
    command = ['pipenv', 'run', 'tensorboard', log_dir]

    command_util.execute_command(command, cwd, new_window=True)


def perform_training(args):
    """Launches the training wrapper script with arguments loaded from a configuration file.

    Args:
      args: Namespace: A Namespace object containing parsed arguments.
    """

    trainer_path = settings.get_training_wrapper_path()

    config_path = Path(args.configuration_file)
    config = config_util.load_config_file(config_path)
    config = override_configuration_values(config, args)

    training_arguments = config_util.get_training_arguments(config)
    command = (
        ['pipenv', 'run', 'python', str(trainer_path)]
        + training_arguments
        + args.args
        + ['--train']
    )

    cwd = settings.get_project_folder_absolute()
    command_util.execute_command(command, cwd, args.new_window, show_command=False)


def override_configuration_values(configuration: dict, args: Namespace):
    """Replaces values in the configuration dictionary with those stored in args.

    Args:
      configuration: dict: Configuration with values to override.
      args: Namespace: Values to insert into the configuration dict.

    Returns:
      A configuration dictionary.
    """

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


def resume_training(args):
    """Launches the training wrapper script with the arguments used by the
    last training command executed.

    Args:
      args: Namespace: A Namespace object containing parsed arguments.
    """

    command = command_util.load_last_history()
    command = prepare_resume_command()

    cwd = settings.get_project_folder_absolute()
    command_util.execute_command(command, cwd, args.new_window, show_command=False)


def prepare_resume_command(command: list):
    """Processes a command loaded from history and prepares it as a
    resume training command.

    Args:
      command: list: A command list loaded from history.

    Returns:
      A configured resume training command.
    """

    if '--timestamp' in command:
        command.remove('--timestamp')

    if '--load' not in command:
        command.append('--load')

    # Note: We slice off the first three elements as they represent a call
    # to mlagents-learn and construct a call to training_wrapper instead.
    command = command[3:]
    trainer_path = settings.get_training_wrapper_path()
    command = ['pipenv', 'run', 'python', str(trainer_path)] + command

    return command


def main():

    args = parse_args(sys.argv[1:])

    if args.list:
        list_training_options()
        return

    if args.edit_config:
        edit_config_file(args)
        return

    if args.tensorboard_start:
        start_tensorboard(args)
        return

    if args.resume:
        resume_training(args)
        return

    perform_training(args)


def parse_args(argv):
    """Builds a Namespace object with parsed arguments.

    Args:
      argv: List of arguments to parse.

    Returns:
      A Namespace object containing parsed arguments.
    """

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
        help='Open a configuration file for editing',
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


def configure_log():
    """Configures logging for the grim-agents CLI.
    """

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "display": {"style": "{", "format": "{levelname}: {message}"},
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
    configure_log()
    main()
    logging.shutdown()
