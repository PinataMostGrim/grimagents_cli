"""
CLI application that wraps 'mlagents-learn' with some quality of life improvements.
- Creating a timestamped run-id
- Load arguments from a configuration file
- Optionally override loaded configuration arguments
- Log training output to file
"""

import argparse
import sys

from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE

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

    if args.timestamp:
        run_id = config_util.get_run_id(config)
        timestamp = get_timestamp()
        config = config_util.set_run_id(f'{run_id}-{timestamp}', config)

    training_arguments = config_util.get_training_arguments(config)

    command = (
        ['pipenv', 'run', 'python', str(trainer_path)]
        + training_arguments
        + args.args
        + ['--train']
    )

    # print(command)

    # TODO: Override configuration file commands with any arguments passed into cli
    # TODO: open new window

    cwd = settings.get_project_folder_absolute()
    with Popen(command, cwd=cwd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')


def get_timestamp():
    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


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

    options_parser.add_argument('--timestamp', '-t', action='store_true', help='Timestamp help')

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
