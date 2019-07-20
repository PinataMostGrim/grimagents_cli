from pathlib import Path
from pprint import pprint
import itertools
import numpy
import sys
import yaml


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


def get_brain_configuration(path: Path, brain):

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


def get_hyperparameter_list(search_config):
    return [name for name in search_config['brain']['hyperparameters']]


def get_hyperparameter_sets(search_config):

    sets = []
    for _, values in search_config['brain']['hyperparameters'].items():
        minimum = values['min']
        maximum = values['max']
        sample_count = values['samples']

        samples = calculate_sample_set(minimum, maximum, sample_count).tolist()
        sets.append(samples)

    return sets


def calculate_sample_set(minimum, maximum, sample_count):

    if (type(minimum) is float or type(maximum) is float):
        return numpy.linspace(minimum, maximum, sample_count)
    else:
        return numpy.linspace(minimum, maximum, sample_count).astype(int)


def get_hyperparameter_permutations(hyperparameter_sets):
    return list(itertools.product(*hyperparameter_sets))


def create_grid_generator(hyperparameters, permutations):

    for i in range(len(permutations)):
        yield get_grid(hyperparameters, permutations, i)


def get_grid(hyperparameters, permutations, index):

    result = list(zip(hyperparameters, permutations[index]))
    return result


def get_brain_config_for_grid(brain_config, grid):

    result = brain_config.copy()
    for hyperparameter in grid:
        result[hyperparameter[0]] = hyperparameter[1]
    return result


# Load yaml and prepare trainer_configuration
run_id = 'PushBlock'
path = Path("../config/PushBlock.yaml")
brain_name = SEARCH_CONFIG['brain']['name']

brain_config = get_brain_configuration(path, brain_name)
# pprint(brain_config)


# Load hyperparameter names
hyperparameters = get_hyperparameter_list(SEARCH_CONFIG)
# print(hyperparameters)

# Load hyperparameter variations
sets = get_hyperparameter_sets(SEARCH_CONFIG)
# pprint(sets)

permutations = get_hyperparameter_permutations(sets)
# pprint(permutations)

grid = get_grid(hyperparameters, permutations, 10)
new_brain_config = get_brain_config_for_grid(brain_config, grid)

pprint(brain_config)
print('-' * 70)
pprint(new_brain_config)


# write brain_config to file
# override trainer-config argument with configuration file
#   - add override argument to __main__.py
# sort out how to execute __main__ from search.py
