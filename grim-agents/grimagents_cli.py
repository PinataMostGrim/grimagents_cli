"""
CLI application that wraps 'mlagents-learn' with some quality of life improvements.
- creating a timestamped run-id
- optionally load arguments from a configuration file
- optionally override loaded configuration
- logging training out put to file
"""


import argparse
import sys

from argparse import Namespace
from pathlib import Path

import grimagents.config as config
import grimagents.command_util as command_util


class Command:
    def execute(self, args: Namespace):
        pass

    @staticmethod
    def configure(subparser):
        pass


class Config(Command):
    def execute(self, args):
        """Opens a configuration file for editing.

        Args:
          args: A Namespace object containing a parsed argument for the configuration
          file to edit.
        """

        config_path = Path(args.file)

        print('')
        config.edit_config_file(config_path)
        print('')

    @staticmethod
    def configure(subparser):
        subparser.add_argument('file', type=str, help='The configuration file to edit')


class List(Command):
    def execute(self, args):
        command = ['pipenv', 'run', 'mlagents-learn', '--help']
        command_util.execute_command(command)
        pass

    @staticmethod
    def configure(subparser):
        pass


class Train(Command):
    def execute(self, args):
        print("This is a test")

    @staticmethod
    def configure(subparser):
        pass


def main():
    args = parse_args(sys.argv[1:])
    command = args.command()
    command.execute(args)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='grimagents',
        description='CLI application that wraps Unity ML-Agents with some quality of life improvements.')

    subparsers = parser.add_subparsers(title='Commands')

    config_parser = subparsers.add_parser('config', help='Open a configuration file for editing')
    Config.configure(config_parser)
    config_parser.set_defaults(command=Config)

    list_parser = subparsers.add_parser('list', help='List mlagents-learn training options')
    List.configure(list_parser)
    list_parser.set_defaults(command=List)

    training_parser = subparsers.add_parser('train', help='Begins training using the specified configuration')
    Train.configure(training_parser)
    training_parser.set_defaults(command=Train)

    args = parser.parse_args(argv)
    if 'command' not in args:
        parser.print_help()
        sys.exit(2)

    return args


if __name__ == '__main__':
    main()
