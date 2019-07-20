# write brain_config to file
# override trainer-config argument with configuration file
#   - add override argument to __main__.py
# sort out how to execute __main__ from search.py
# logging config


import argparse
import itertools
import numpy
import sys
import yaml

import grimagents.command_util as command_util
import grimagents.config as config_util

from argparse import Namespace
from pathlib import Path
from pprint import pprint


SEARCH_CONFIG = {
    "type": "grid",
    "sample_size": 3,
    "in_parallel": False,
    "brain": {
        "name": "PushBlockLearning",
        "hyperparameters": {
            "batch_size": {
                "min": 512,
                "max": 5120,
                "samples": 3},
            "beta": {
                "min": 1e-4,
                "max": 1e-2,
                "samples": 3},
            "num_epoch": {
                "min": 3,
                "max": 10,
                "samples": 3},
        }
    }
}


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
    """Performs a hyperparameter grid search using values from a grimagents configuration file.
    """

    def execute(self, args):

        # Load yaml and prepare trainer_configuration
        # run_id = 'PushBlock'
        path = Path(args.configuration_file)

        grim_config = command_util.load_json_file(path)
        trainer_config_path = Path(grim_config['trainer-config-path'])

        brain_name = SEARCH_CONFIG['brain']['name']
        brain_config = self.get_brain_configuration(trainer_config_path, brain_name)
        # pprint(brain_config)

        # Load hyperparameter names
        hyperparameters = self.get_hyperparameter_list(SEARCH_CONFIG)
        # print(hyperparameters)

        # Load hyperparameter variations
        sets = self.get_hyperparameter_sets(SEARCH_CONFIG)
        # pprint(sets)

        permutations = self.get_hyperparameter_permutations(sets)
        # pprint(permutations)

        grid = self.get_grid(hyperparameters, permutations, 10)
        new_brain_config = self.get_brain_config_for_grid(brain_config, grid)

        pprint(brain_config)
        print('-' * 70)
        pprint(new_brain_config)


    def get_brain_configuration(self, path: Path, brain):

        with path.open(mode='r') as file:
            loaded_data = yaml.load(file, Loader=yaml.BaseLoader)

        result = ""

        if ('default' in loaded_data):
            result = loaded_data['default']
        # else:
            # TODO: Get default configuration from config.py

        if brain not in loaded_data:
            print(f'Unable to find configuration settings for brain \'{brain}\' in loaded_data')
            sys.exit()

        brain_data = loaded_data[brain]

        for key, value in brain_data.items():
            result[key] = value

        return result


    def get_hyperparameter_list(self, search_config):
        return [name for name in search_config['brain']['hyperparameters']]


    def get_hyperparameter_sets(self, search_config):

        sets = []
        for _, values in search_config['brain']['hyperparameters'].items():
            minimum = values['min']
            maximum = values['max']
            sample_count = values['samples']

            samples = self.calculate_sample_set(minimum, maximum, sample_count).tolist()
            sets.append(samples)

        return sets


    def calculate_sample_set(self, minimum, maximum, sample_count):

        if (type(minimum) is float or type(maximum) is float):
            return numpy.linspace(minimum, maximum, sample_count)
        else:
            return numpy.linspace(minimum, maximum, sample_count).astype(int)


    def get_hyperparameter_permutations(self, hyperparameter_sets):
        return list(itertools.product(*hyperparameter_sets))


    def create_grid_generator(self, hyperparameters, permutations):

        for i in range(len(permutations)):
            yield self.get_grid(hyperparameters, permutations, i)


    def get_grid(self, hyperparameters, permutations, index):

        result = list(zip(hyperparameters, permutations[index]))
        return result


    def get_brain_config_for_grid(self, brain_config, grid):

        result = brain_config.copy()
        for hyperparameter in grid:
            result[hyperparameter[0]] = hyperparameter[1]
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
