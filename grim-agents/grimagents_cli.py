"""
CLI application that wraps 'mlagents-learn' with some quality of life improvements.
- Load arguments from a configuration file
- Log training output to file
- Optionally timestamp the training run-id
- Optionally override loaded configuration arguments
- Optionally launch training in a new console window
"""

import argparse
import subprocess
import sys

from argparse import Namespace
from pathlib import Path
from subprocess import CREATE_NEW_CONSOLE

import grimagents.config as config_util
import grimagents.command_util as command_util
import grimagents.settings as settings


def list_training_options():
    """Outputs mlagents-learn usage options."""

    command = ['pipenv', 'run', 'mlagents-learn', '--help']
    command_util.execute_command(command)


def edit_config_file(args):
    """Opens a configuration file for editing.

    Args:
      args: Namespace: A Namespace object containing parsed arguments.
    """

    print('')
    config_path = Path(args.edit_config)
    config_util.edit_config_file(config_path)
    print('')


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

    if args.new_window:
        command = ['cmd', '/K'] + command
        subprocess.Popen(command, cwd=cwd, creationflags=CREATE_NEW_CONSOLE)
    else:
        subprocess.run(command, cwd=cwd)


def override_configuration_values(configuration: dict, args: Namespace):
    """Replaces values in the configuration dictionary with those stored in args.

    Args:
      configuration: dict: Configuration with values to override.
      args: Namespace: Values to insert into the configuration dict.

    Returns:
      A configuration dictionary.
    """

    if args.lesson:
        configuration = config_util.set_lesson(args.lesson, configuration)
    if args.run_id:
        configuration = config_util.set_run_id(args.run_id, configuration)
    if args.no_graphics:
        configuration = config_util.set_no_graphics_enabled(True, configuration)
    if args.timestamp:
        configuration = config_util.set_timestamp_enabled(True, configuration)

    return configuration


def main():

    args = parse_args(sys.argv[1:])

    if args.list:
        list_training_options()
        return

    if args.edit_config:
        edit_config_file(args)
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

    # Parser for arguments that may override configuration values
    overrides_parser = argparse.ArgumentParser(add_help=False)
    overrides_parser.add_argument('--lesson', type=int, default=0)
    overrides_parser.add_argument('--run-id', type=str, default=None)
    overrides_parser.add_argument('--no-graphics', action='store_true')
    overrides_parser.add_argument(
        '--timestamp', action='store_true', help='Append timestamp to run-id'
    )

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
