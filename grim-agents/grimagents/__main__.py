"""
CLI application that wraps 'mlagents-learn' with some quality of life improvements.
- Load arguments from a configuration file
- Log training output to file
- Optionally timestamp the training run-id
- Optionally override loaded configuration arguments
- Optionally launch training in a new console window
- Optionally import the trained model into Unity project
"""

import argparse
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
    if args.no_graphics is not None:
        configuration = config_util.set_no_graphics_enabled(str(args.no_graphics), configuration)
    if args.timestamp:
        configuration = config_util.set_timestamp_enabled(True, configuration)
    if args.no_timestamp:
        configuration = config_util.set_timestamp_enabled(False, configuration)

    return configuration


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
        '--new-window', action='store_true', help='Run training process in a new console window.'
    )
    options_parser.add_argument(
        '--tensorboard-start', action='store_true', help='Start tensorboard server')

    # Parser for arguments that may override configuration values
    overrides_parser = argparse.ArgumentParser(add_help=False)
    overrides_parser.add_argument('--env', type=str)
    overrides_parser.add_argument('--lesson', type=int)
    overrides_parser.add_argument('--run-id', type=str)
    overrides_parser.add_argument('--num-envs', type=int)
    overrides_parser.add_argument('--no-graphics', type=bool)

    group = overrides_parser.add_mutually_exclusive_group()
    group.add_argument('--timestamp', action="store_true", help='Append timestamp to run-id. Overrides configuration setting.')
    group.add_argument('--no-timestamp', action="store_true", help='Do not append timestamp to run-id. Overrides configuration setting.')

    # Parser for arguments that are passed on to the training wrapper
    parser = argparse.ArgumentParser(
        prog='grimagents',
        description='CLI application that wraps Unity ML-Agents with some quality of life improvements.',
        parents=[options_parser, overrides_parser],
    )

    parser.add_argument(
        'configuration_file', type=str, help='Configuration file to load training arguments from.'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments applied to training (ex. --slow, --debug, --load).',
    )

    args, unparsed_args = options_parser.parse_known_args()
    args, unparsed_args = overrides_parser.parse_known_args(unparsed_args, args)

    if len(argv) == 0:
        parser.print_help()
        sys.exit(0)

    if len(unparsed_args) > 0:
        args = parser.parse_args(unparsed_args, args)

    return args


if __name__ == '__main__':
    main()
