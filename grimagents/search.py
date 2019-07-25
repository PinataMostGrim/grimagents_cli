"""
CLI application that performs hyperparameter searches using grimagents and a grimagents configuration file.

Features:
- Grid Search for hyperparameters
"""

import argparse
import logging
import logging.config
import subprocess
import sys

from argparse import Namespace
from pathlib import Path

import grimagents.command_util as command_util
import grimagents.config as config_util

from grimagents.grid_search import GridSearch


search_log = logging.getLogger('grimagents.search')


class Command:
    def execute(self, args: Namespace):
        pass


class EditGrimConfigFile(Command):
    """Opens a grimagents configuration file for editing or creates on if a file does not already exist. Appends a search entry to the configuration file if it does not already have one.
    """

    def execute(self, args):
        file_path = Path(args.edit_config)
        config_util.edit_grim_config_file(file_path, add_search=True)


class PerformGridSearch(Command):
    """Perform a hyperparameter grid search using values from a grimagents configuration file.
    """

    def execute(self, args):

        # Gather configurations
        grim_config_path = Path(args.configuration_file)
        grim_config = config_util.load_grim_config_file(grim_config_path)

        search_config = grim_config[config_util.SEARCH]

        trainer_config_path = Path(grim_config['trainer-config-path'])
        trainer_config = config_util.load_trainer_configuration(trainer_config_path)

        grid_search = GridSearch(search_config, trainer_config)

        # Perform search on permutations
        search_log.info('-' * 63)
        search_log.info('Performing grid search for hyperparameters:')
        for i in range(len(grid_search.hyperparameters)):
            search_log.info(f'    {grid_search.hyperparameters[i]}: {grid_search.hyperparameter_sets[i]}')
        search_log.info('-' * 63)

        grid_config_path = trainer_config_path.with_name('search_config.yaml')

        for i in range(grid_search.get_intersect_count()):

            intersect = grid_search.get_intersect(i)
            intersect_brain_config = grid_search.get_brain_config_for_intersect(intersect)

            # Write trainer configuration file for grid
            command_util.write_yaml_file(intersect_brain_config, grid_config_path)

            # Execute training with the new trainer-config and run_id
            run_id = grim_config[config_util.RUN_ID] + f'_{i:02d}'
            command = ['pipenv', 'run', 'python', '-m', 'grimagents', str(grim_config_path), '--trainer-config', str(grid_config_path), '--run-id', run_id]

            search_log.info('-' * 63)
            search_log.info(f'Training {run_id}:')
            for i in range(len(intersect)):
                search_log.info(f'    {intersect[i][0]}: {intersect[i][1]}')
            search_log.info('-' * 63)

            subprocess.run(command)

        grid_config_path.unlink()
        search_log.info('Grid search complete\n')


def main():

    args = parse_args(sys.argv[1:])

    if args.edit_config:
        EditGrimConfigFile().execute(args)
    else:
        if not pipenv_exists():
            search_log.error('A Pipenv virtual environment is not accessible from this directory')
            return
        PerformGridSearch().execute(args)


def pipenv_exists():
    """Returns True if a virtual environment can be accessed through Pipenv and False otherwise.
    """

    process = subprocess.run(['pipenv', '--venv'], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if 'No virtualenv has been created for this project yet!' in process.stderr:
        return False

    return True


def parse_args(argv):
    """Builds a Namespace object out of parsed arguments."""

    options_parser = argparse.ArgumentParser(add_help=False)
    options_parser.add_argument('--edit-config', metavar='<file>', type=str, help='Open a grimagents configuration file for editing. Adds a search entry if one is not present.')
    # options_parser.add_argument('--random', '-r', metavar='<n>', type=int, help='Execute <n> random searches instead of performing a grid search')
    # options_parser.add_argument('--in-parallel', action='store_true', help='Perform all searchs in parallel (Be careful with this!)')

    parser = argparse.ArgumentParser(prog='search',
                                     description='CLI application that performs a hyperparameter grid search',
                                     parents=[options_parser])
    parser.add_argument('configuration_file', type=str, help='grimagents configuration file with search parameters')

    args, unparsed_args = options_parser.parse_known_args()

    if len(argv) == 0:
        parser.print_help()
        sys.exit(0)

    if len(unparsed_args) > 0:
        args = parser.parse_args(unparsed_args, args)

    return args


def configure_logging():
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
            }
        },
        "loggers": {
            "grimagents.search": {"handlers": ["console"]},
        },
        "root": {"level": "INFO"},
    }

    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    configure_logging()
    main()
    logging.shutdown()
