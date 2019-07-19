from pathlib import Path
from pprint import pprint
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


def load_brain_configuration(path: Path, brain):

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


def get_hyperparameter_sets(search_config):

    sets = []
    for hyperparameter, values in SEARCH_CONFIG['brain']['hyperparameters'].items():
        minimum = values['min']
        maximum = values['max']
        sample_count = values['samples']

        samples = [hyperparameter]
        samples = samples + get_distributed_samples(minimum, maximum, sample_count).tolist()
        sets.append(samples)

    return sets


def get_distributed_samples(minimum, maximum, sample_count):

    if (type(minimum) is float or type(maximum) is float):
        return numpy.linspace(minimum, maximum, sample_count)
    else:
        return numpy.linspace(minimum, maximum, sample_count).astype(int)


# Load yaml and prepare trainer_configuration
run_id = 'PushBlock'
path = Path("../config/PushBlock.yaml")
brain_name = SEARCH_CONFIG['brain']['name']

brain_config = load_brain_configuration(path, brain_name)
pprint(brain_config)


# Load hyperparameter variations
sample_sets = get_hyperparameter_sets(SEARCH_CONFIG)
pprint(sample_sets)
