#TODO: logging config, convert print statements to logging


import argparse
import itertools
import subprocess
import sys

import grimagents.command_util as command_util
import grimagents.config as config_util
import grimagents.settings as settings

from argparse import Namespace
from pathlib import Path
from pprint import pprint


class Command:
    def execute(self, args: Namespace):
        pass


class EditGrimConfigFile(Command):
    """Opens a grimagents configuration file for editing or creates on if a file does not already exist. Appends a search entry to the configuration file if it does not already have one.
    """

    def execute(self, args):
        file_path = Path(args.edit_config)
        config_util.edit_grim_config_file(file_path, add_search=True)


class GridSearch(Command):
    """Perform a hyperparameter grid search using values from a grimagents configuration file.
    """

    def execute(self, args):

        # Load trainer configuration from file
        grim_config_path = Path(args.configuration_file)
        grim_config = config_util.load_grim_config_file(grim_config_path)
        trainer_config_path = Path(grim_config['trainer-config-path'])

        trainer_config = config_util.load_trainer_configuration(trainer_config_path)

        # Load search configuration
        search_config = grim_config[config_util.SEARCH]

        # Fetch brain configuration
        brain_name = search_config['brain']['name']
        brain_config = self.get_brain_configuration(trainer_config, brain_name)

        # Determine search permutations
        hyperparameters = self.get_search_hyperparameters(search_config)
        sets = self.get_search_sets(search_config)
        permutations = self.get_search_permutations(sets)

        # Perform search on permutations
        grid_config_path = trainer_config_path.with_name('search_config.yaml')
        for i in range(len(permutations)):

            grid = self.get_grid(hyperparameters, permutations, i)
            grid_brain_config = self.get_brain_config_for_grid(brain_config, grid)

            # Write trainer configuration file for grid
            command_util.write_yaml_file(grid_brain_config, grid_config_path)
            # TODO: Handle writing config files while parallel training

            # Execute training with the new trainer-config and run_id
            run_id = grim_config[config_util.RUN_ID] + f'_{i:02d}'
            command = ['pipenv', 'run', 'python', '-m', 'grimagents', str(grim_config_path), '--trainer-config', str(grid_config_path), '--run-id', run_id]

            print(f'Running grid {i:02d}')
            cwd = settings.get_project_folder_absolute()
            subprocess.run(command, cwd=cwd)
            print('')
        print('Search complete')
        # TODO: remove grid configuration file

    def get_brain_configuration(self, trainer_config, brain_name):
        """Returns a complete trainer configuration for a brain. If hyperparameter values
        are missing, they are isnerted with default values.
        """

        if ('default' in trainer_config):
            result = {'default': trainer_config['default']}
        else:
            result = {'default': config_util.get_default_trainer_config()['default']}

        if brain_name not in trainer_config:
            print(f'Unable to find configuration settings for brain \'{brain_name}\' in trainer_config')
            sys.exit()

        brain_data = trainer_config[brain_name]
        result[brain_name] = {}

        for key, value in brain_data.items():
            result[brain_name][key] = value

        return result

    def get_search_hyperparameters(self, search_config):
        return [name for name in search_config['brain']['hyperparameters']]

    def get_search_sets(self, search_config):

        sets = []
        for _, values in search_config['brain']['hyperparameters'].items():
            sets.append(values)

        return sets

    def get_search_permutations(self, hyperparameter_sets):
        return list(itertools.product(*hyperparameter_sets))

    def get_grid(self, hyperparameters, permutations, index):

        result = list(zip(hyperparameters, permutations[index]))
        return result

    def get_brain_config_for_grid(self, brain_config, grid):
        """Returns a brain configuration dictionary with values overriden by the grid settings.
        """

        result = brain_config.copy()
        brain_name = [*result.keys()][0]

        for hyperparameter in grid:
            result[brain_name][hyperparameter[0]] = hyperparameter[1]

        return result


# def random_search(args):
#     pass


def main():

    args = parse_args(sys.argv[1:])

    if args.edit_config:
        EditGrimConfigFile().execute(args)
    # elif args.random:
    #     random_search(args)
    else:
        GridSearch().execute(args)


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


if __name__ == '__main__':
    main()
