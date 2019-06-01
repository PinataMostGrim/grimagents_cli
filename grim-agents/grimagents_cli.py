"""
CLI application that wraps 'mlagents-learn' with some quality of life improvements.
- creating a timestamped run-id
- optionally load arguments from a configuration file
- optionally override loaded configuration
- logging training out put to file
"""

import argparse
import sys

from pathlib import Path

import grimagents.config as config_util
import grimagents.command_util as command_util
import grimagents.settings as settings


def list_training_options():
    command = ['pipenv', 'run', 'mlagents-learn', '--help']
    command_util.execute_command(command)


def edit_config_file(args):

    print('')
    config_path = Path(args.edit_config)
    config_util.edit_config_file(config_path)
    print('')


def perform_training(args):
    config_path = Path(args.configuration_file)
    config = config_util.load_config_file(config_path)

    trainer_path = settings.get_training_wrapper_path()

    command = ['pipenv', 'run', 'python', str(trainer_path)] + config_util.get_training_arguments(config) + ['--train']

    # TODO: Override configuration file commands with any arguments passed into grimagents_cli

    command_util.execute_command(command)
    # print(command)

    # TODO: Execute command from the project's root working directory
    # cwd = settings.get_project_folder_absolute()

    # TODO: Execute command in new cmd / shell


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

    parser = argparse.ArgumentParser(
        prog='grimagents',
        description='CLI application that wraps Unity ML-Agents with some quality of life improvements.',
        parents=[options_parser],
    )

    parser.add_argument('configuration_file', type=str, help='Training help')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Additional arguments')

    args, unparsed_args = options_parser.parse_known_args()

    if len(argv) == 0:
        parser.print_help()
        sys.exit(0)

    if len(unparsed_args) > 0:
        args = parser.parse_args(unparsed_args, args)

    return args


if __name__ == '__main__':
    main()
