"""
CLI application that performs hyperparameter searches using grimagents and a grimagents configuration file.

Features:
- Grid Search for hyperparameters

See readme.md for more documentation.
"""

import argparse
import logging
import logging.config
import subprocess
import sys

from pathlib import Path

import grimagents.command_util as command_util
import grimagents.config as config_util
import grimagents.common as common
import grimagents.settings as settings

from grimagents.grid_search import GridSearch


search_log = logging.getLogger('grimagents.search')


class Command:
    def __init__(self, args):
        self.args = args

    def execute(self):
        pass


class EditGrimConfigFile(Command):
    """Opens a grimagents configuration file for editing or creates on if a file does not already exist. Appends a search entry to the configuration file if it does not already have one.
    """

    def execute(self):
        file_path = Path(self.args.edit_config)
        config_util.edit_grim_config_file(file_path, add_search=True)


class GridSearchCommand(Command):
    def __init__(self, args):

        self.args = args

        # Gather configurations
        self.grim_config_path = Path(args.configuration_file)
        self.grim_config = config_util.load_grim_config_file(self.grim_config_path)

        self.search_config = self.grim_config[config_util.SEARCH]

        self.trainer_config_path = Path(self.grim_config['trainer-config-path'])
        self.trainer_config = config_util.load_trainer_configuration(self.trainer_config_path)

        self.grid_search = GridSearch(self.search_config, self.trainer_config)

        self.search_config_path = self.trainer_config_path.with_name('search_config.yaml')


class OutputGridSearchCount(GridSearchCommand):
    """Prints out the total number of training runs a grimagents configuration file will attempt.
    """

    def execute(self):

        search_log.info(
            f' \'{self.trainer_config_path}\' will perform {self.grid_search.get_intersect_count()} training runs'
        )


class PerformGridSearch(GridSearchCommand):
    """Perform a hyperparameter grid search using values from a grimagents configuration file.
    """

    def execute(self):

        if self.args.resume:
            count = self.grid_search.get_intersect_count()
            if self.args.resume > count:
                search_log.warning(
                    f'\'{self.trainer_config_path}\' will only perform {count} training runs, unable to resume at index {self.args.resume}'
                )
                sys.exit()
            start_index = self.args.resume
        else:
            start_index = 0

        search_log.info('-' * 63)
        search_log.info('Performing grid search for hyperparameters:')
        for i in range(len(self.grid_search.hyperparameters)):
            search_log.info(
                f'    {self.grid_search.hyperparameters[i]}: {self.grid_search.hyperparameter_sets[i]}'
            )
        search_log.info('-' * 63)

        for i in range(start_index, self.grid_search.get_intersect_count()):

            intersect = self.grid_search.get_intersect(i)
            intersect_brain_config = self.grid_search.get_brain_config_for_intersect(intersect)

            # Write trainer configuration file for current intersect
            if self.args.parallel:
                self.search_config_path = self.search_config_path.with_name(
                    f'search_config{i:02d}.yaml'
                )

            command_util.write_yaml_file(intersect_brain_config, self.search_config_path)

            # Execute training with the intersect config and run_id
            run_id = self.grim_config[config_util.RUN_ID] + f'_{i:02d}'
            command = [
                'pipenv',
                'run',
                'python',
                '-m',
                'grimagents',
                str(self.grim_config_path),
                '--trainer-config',
                str(self.search_config_path),
                '--run-id',
                run_id,
            ]

            search_log.info('-' * 63)
            search_log.info(f'Training {run_id}:')
            for j in range(len(intersect)):
                search_log.info(f'    {intersect[j][0]}: {intersect[j][1]}')
            search_log.info('-' * 63)

            if self.args.parallel:
                command = ['cmd', '/K'] + command
                command = command + ['--base-port', str(5005 + i)]
                subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.run(command)

        if not self.args.parallel and self.search_config_path.exists():
            self.search_config_path.unlink()

        search_log.info('Grid search complete\n')


class ExportGridSearchConfiguration(GridSearchCommand):
    """Exports a trainer config file for a given GridSearch intersect."""

    def execute(self):

        search_log.info(
            f'Exporting trainer configuration for GridSearch intersect \'{self.args.export_intersect}\' into \'{self.search_config_path}\''
        )

        intersect = self.grid_search.get_intersect(self.args.export_intersect)
        intersect_brain_config = self.grid_search.get_brain_config_for_intersect(intersect)
        command_util.write_yaml_file(intersect_brain_config, self.search_config_path)


def main():

    args = parse_args(sys.argv[1:])

    if not common.is_pipenv_present():
        search_log.error(
            'No virtual environment is accessible by Pipenv from this directory, unable to run mlagents-learn'
        )
        return

    if args.edit_config:
        EditGrimConfigFile(args).execute()
    elif args.search_count:
        OutputGridSearchCount(args).execute()
    elif args.export_intersect:
        ExportGridSearchConfiguration(args).execute()
    else:
        PerformGridSearch(args).execute()


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
        '--parallel',
        action='store_true',
        help='Perform all searchs in parallel (be careful with this one!)',
    )
    options_parser.add_argument(
        '--resume',
        metavar='<search index>',
        type=int,
        help='Resume grid search from <search index> (counting from zero)',
    )
    options_parser.add_argument(
        '--export-intersect',
        metavar='<search index>',
        type=int,
        help='Export trainer configuration for a GridSearch intersect',
    )
    # options_parser.add_argument('--random', '-r', metavar='<n>', type=int, help='Execute <n> random searches instead of performing a grid search')

    parser = argparse.ArgumentParser(
        prog='search',
        description='CLI application that performs a hyperparameter grid search',
        parents=[options_parser],
    )
    parser.add_argument(
        'configuration_file', type=str, help='grimagents configuration file with search parameters'
    )

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
            },
            "file": {"class": "logging.FileHandler", "filename": "", "formatter": "timestamp"},
        },
        "loggers": {"grimagents.search": {"handlers": ["console", "file"]}},
        "root": {"level": "INFO"},
    }

    log_file = settings.get_log_file_path()

    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log_config['handlers']['file']['filename'] = log_file
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    configure_logging()
    main()
    logging.shutdown()
