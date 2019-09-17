"""CLI application that performs hyperparameter searches using a grimagents configuration file.

Features:
- Grid Search for hyperparameters
- Random Search for hyperparameters
- Bayesian Search for hyperparameters
- Resume Grid Search
- Export trainer configuration file for a given Grid Search index

See readme.md for more information.
"""

import argparse
import logging
import logging.config
import sys

import grimagents.common as common
import grimagents.settings as settings

from grimagents.search_commands import (
    EditGrimConfigFile,
    OutputGridSearchCount,
    PerformGridSearch,
    ExportGridSearchConfiguration,
    PerformRandomSearch,
    PerformBayesianSearch,
)


search_log = logging.getLogger('grimagents.search')


def main():

    configure_logging()

    if not common.is_pipenv_present():
        search_log.error(
            'No virtual environment is accessible by Pipenv from this directory, unable to run mlagents-learn'
        )
        return

    argv = get_argvs()
    args = parse_args(argv)

    if args.edit_config:
        EditGrimConfigFile(args).execute()
    elif args.search_count:
        OutputGridSearchCount(args).execute()
    elif args.export_index:
        ExportGridSearchConfiguration(args).execute()
    elif args.random:
        PerformRandomSearch(args).execute()
    elif args.bayesian:
        PerformBayesianSearch(args).execute()
    else:
        PerformGridSearch(args).execute()

    logging.shutdown()


def get_argvs():

    return sys.argv[1:]


def parse_args(argv):
    """Builds a Namespace object out of parsed arguments."""

    options_parser = argparse.ArgumentParser(add_help=False)
    options_parser.add_argument(
        '--edit-config',
        metavar='<file>',
        type=str,
        help='Open a grimagents configuration file for editing. Adds a default search entry if one is not present.',
    )
    options_parser.add_argument(
        '--search-count',
        action='store_true',
        help='Output the total number of searches a grimagents configuration file will attempt',
    )
    options_parser.add_argument(
        '--resume',
        metavar='<search index>',
        type=int,
        help='Resume grid search from <search index> (counting from zero)',
    )
    options_parser.add_argument(
        '--export-index',
        metavar='<search index>',
        type=int,
        help='Export trainer configuration for grid search <index>',
    )
    options_parser.add_argument(
        '--random',
        '-r',
        metavar='<n>',
        type=int,
        help='Execute <n> random searches instead of performing a grid search',
    )
    options_parser.add_argument(
        '--bayesian',
        '-b',
        metavar=('<exploration_steps>', '<optimization_steps>'),
        type=int,
        nargs=2,
        help='Execute Bayesian Search using a number of exploration steps and optimization steps',
    )
    options_parser.add_argument(
        '--bayes-save',
        '-s',
        action='store_true',
        help='Save Bayesian optimization progress log to folder',
    )
    options_parser.add_argument(
        '--bayes-load',
        '-l',
        action='store_true',
        help='Loads Bayesian optimization progress logs from folder',
    )

    parser = argparse.ArgumentParser(
        prog='grimsearch',
        description='CLI application that performs a hyperparameter search',
        parents=[options_parser],
    )
    parser.add_argument(
        'configuration_file', type=str, help='A grimagents configuration file containing search parameters'
    )

    args, unparsed_args = options_parser.parse_known_args(argv)

    if len(argv) == 0:
        parser.print_help()
        sys.exit(0)

    if len(unparsed_args) > 0:
        args = parser.parse_args(unparsed_args, args)

    return args


def configure_logging():
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
        'loggers': {'grimagents.search': {'handlers': ['console', 'file']}},
        'root': {'level': 'INFO'},
    }

    log_file = settings.get_log_file_path()

    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log_config['handlers']['file']['filename'] = log_file
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    main()
