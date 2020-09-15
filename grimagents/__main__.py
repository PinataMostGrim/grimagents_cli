"""
CLI application that loads training arguments from a configuration file and sends
them to the grimagents.training_wrapper. This script aims to automate several
repetitive training tasks.

Features:
- Load training arguments from a configuration file
- Override loaded configuration arguments with command line arguments
- Optionally time-stamp the training run-id

See readme.md for more information.
"""

import argparse
import logging
import logging.config
import sys

import grimagents.settings as settings
import grimagents.common as common

from grimagents.training_commands import (
    ListTrainingOptions,
    EditGrimConfigFile,
    EditTrainerConfigFile,
    StartTensorboard,
    PerformTraining,
)


main_log = logging.getLogger('grimagents.main')


def main():

    configure_logging()

    if not common.is_pipenv_present():
        main_log.error(
            'No virtual environment is accessible by Pipenv from this directory, unable to run mlagents-learn'
        )
        sys.exit(1)

    argv = get_argvs()
    args = parse_args(argv)

    if args.list:
        ListTrainingOptions(args).execute()
    elif args.edit_config:
        EditGrimConfigFile(args).execute()
    elif args.edit_trainer_config:
        EditTrainerConfigFile(args).execute()
    elif args.tensorboard_start:
        StartTensorboard(args).execute()
    else:
        PerformTraining(args).execute()

    logging.shutdown()


def get_argvs():

    return sys.argv[1:]


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
        '--tensorboard-start', '-s', action='store_true', help='Start tensorboard server'
    )
    options_parser.add_argument(
        '--resume', '-r', action='store_true', help='Resume the training run specified by --run-id'
    )
    options_parser.add_argument(
        '--dry-run', '-n', action='store_true', help='Print command without executing'
    )

    # Parser for arguments that may override configuration values
    overrides_parser = argparse.ArgumentParser(add_help=False)
    overrides_parser.add_argument(
        '--trainer-config', type=str, help='Overrides configuration setting'
    )
    overrides_parser.add_argument('--env', type=str, help='Overrides configuration setting')
    overrides_parser.add_argument('--run-id', type=str, help='Overrides configuration setting')
    overrides_parser.add_argument('--base-port', type=int, help='Overrides configuration setting')
    overrides_parser.add_argument('--num-envs', type=int, help='Overrides configuration setting')
    overrides_parser.add_argument(
        '--inference',
        action='store_true',
        help='Overrides configuration setting',
    )

    graphics_group = overrides_parser.add_mutually_exclusive_group()
    graphics_group.add_argument(
        '--graphics', action='store_true', help='Overrides configuration setting'
    )
    graphics_group.add_argument(
        '--no-graphics', action='store_true', help='Overrides configuration setting'
    )

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

    multi_gpu_group = overrides_parser.add_mutually_exclusive_group()
    multi_gpu_group.add_argument(
        '--multi-gpu',
        action='store_true',
        help='Use multi-gpu if supported. Overrides configuration setting.',
    )
    multi_gpu_group.add_argument(
        '--no-multi-gpu',
        action='store_true',
        help='Do not use multi-gpu. Overrides configuration setting.',
    )

    # Parser for arguments that are passed on to the training wrapper
    parser = argparse.ArgumentParser(
        prog='grimagents',
        description='CLI application that wraps Unity ML-Agents with more automation.',
        parents=[options_parser, overrides_parser],
    )

    parser.add_argument(
        'configuration_file', type=str, help='Configuration file to extract training arguments from'
    )
    parser.add_argument(
        'additional_args',
        metavar='args',
        nargs=argparse.REMAINDER,
        help='Additional arguments applied to training (ex. --debug, --load)',
    )

    args, unparsed_args = options_parser.parse_known_args(argv)
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
        'loggers': {
            'grimagents.main': {'handlers': ['console', 'file']},
            'grimagents.config': {'handlers': ['console', 'file']},
            'grimagents.command_util': {'handlers': ['console', 'file']},
        },
        'root': {'level': 'INFO'},
    }

    log_file = settings.get_log_file_path()

    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log_config['handlers']['file']['filename'] = log_file
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    main()
