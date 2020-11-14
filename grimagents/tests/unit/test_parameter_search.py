import numpy
import pytest

from grimagents.parameter_search import (
    GridSearch,
    RandomSearch,
    BayesianSearch,
    InvalidGridSearchIndex,
)


@pytest.fixture
def search_config():
    return {
        'behavior_name': 'BEHAVIOR_NAME',
        'search_parameters': {
            'hyperparameters.beta': [1e-4, 1e-2],
            'hyperparameters.num_epoch': [3, 10],
            'hyperparameters.learning_rate': [1e-5, 1e-3],
            'network_settings.hidden_units': [32, 512],
            'network_settings.num_layers': [1, 3],
        },
    }


@pytest.fixture
def trainer_config():
    return {
        'behaviors': {
            'BEHAVIOR_NAME': {
                'trainer_type': 'ppo',
                'hyperparameters': {
                    'batch_size': 1024,
                    'beta': 5.0e-3,
                    'buffer_size': 10240,
                    'epsilon': 0.2,
                    'lambd': 0.95,
                },
                'network_settings': {
                    'hidden_units': 128,
                },
                'reward_signals': {
                    'extrinsic': {
                        'gamma': 0.99,
                        'strength': 1.0,
                    },
                },
            },
            'OTHER_BEHAVIOR_NAME': {
                'trainer_type': 'ppo',
                'hyperparameters': {
                    'batch_size': 1024,
                    'buffer_size': 10240,
                    'epsilon': 0.2,
                },
            },
        },
    }


def test_get_search_hyperparameters(search_config):
    """Test for the extraction of hyperparameter names from a search configuration dictionary."""

    assert GridSearch.get_search_hyperparameters(search_config) == [
        'hyperparameters.beta',
        'hyperparameters.num_epoch',
        'hyperparameters.learning_rate',
        'network_settings.hidden_units',
        'network_settings.num_layers',
    ]


def test_get_hyperparameter_sets(search_config):
    """Test for the correct construction of GridSearch hyperparameter sets."""

    assert GridSearch.get_hyperparameter_sets(search_config) == [
        [0.0001, 0.01],
        [3, 10],
        [1e-05, 0.001],
        [32, 512],
        [1, 3],
    ]


def test_get_search_permutations(search_config):
    """Test for the correct construction of GridSearch hyperparamter permutations."""

    sets = [[0.0001, 0.01], [32, 512], [1e-05, 0.001]]
    assert GridSearch.get_search_permutations(sets) == [
        (0.0001, 32, 1e-05),
        (0.0001, 32, 0.001),
        (0.0001, 512, 1e-05),
        (0.0001, 512, 0.001),
        (0.01, 32, 1e-05),
        (0.01, 32, 0.001),
        (0.01, 512, 1e-05),
        (0.01, 512, 0.001),
    ]


def test_get_grid_search_configuration(search_config, trainer_config):
    """Tests for the correct creation of GridSearch configurations."""

    search = GridSearch(search_config, trainer_config)

    assert search.get_search_configuration(0) == {
        'hyperparameters.beta': 0.0001,
        'hyperparameters.learning_rate': 1e-05,
        'hyperparameters.num_epoch': 3,
        'network_settings.hidden_units': 32,
        'network_settings.num_layers': 1,
    }

    assert search.get_search_configuration(15) == {
        'hyperparameters.beta': 0.0001,
        'hyperparameters.learning_rate': 0.001,
        'hyperparameters.num_epoch': 10,
        'network_settings.hidden_units': 512,
        'network_settings.num_layers': 3,
    }

    assert search.get_search_configuration(31) == {
        'hyperparameters.beta': 0.01,
        'hyperparameters.learning_rate': 0.001,
        'hyperparameters.num_epoch': 10,
        'network_settings.hidden_units': 512,
        'network_settings.num_layers': 3,
    }


def test_get_grid_search_count(search_config, trainer_config):
    """Tests for the correct calculation of permutation count."""

    search = GridSearch(search_config, trainer_config)
    assert search.get_grid_search_count() == 32


def test_invalid_grid_search_index(search_config, trainer_config):
    """Tests that InvalidGridSearchIndex exceptions are raised."""

    search = GridSearch(search_config, trainer_config)

    with pytest.raises(InvalidGridSearchIndex):
        search.get_search_configuration(32)


def test_get_trainer_config_with_overrides(search_config, trainer_config):
    """Tests that flattened period separated keys are correctly expanded for the search configuration. Additionally tests to ensure nested sibling keys do not get overwritten."""

    search = GridSearch(search_config, trainer_config)
    overrides = {
        'reward_signals.extrinsic.gamma': 0.91,
        'reward_signals.curiosity.encoding_size': 256,
        'reward_signals.curiosity.strength': 1.0,
        'reward_signals.curiosity.gamma': 0.99,
    }

    search_config = search.get_trainer_config_with_overrides(overrides)

    assert search_config == {
        'behaviors': {
            'BEHAVIOR_NAME': {
                'hyperparameters': {
                    'batch_size': 1024,
                    'beta': 0.005,
                    'buffer_size': 10240,
                    'epsilon': 0.2,
                    'lambd': 0.95,
                },
                'network_settings': {'hidden_units': 128},
                'reward_signals': {
                    'curiosity': {'encoding_size': 256, 'gamma': 0.99, 'strength': 1.0},
                    'extrinsic': {'gamma': 0.91, 'strength': 1.0},
                },
                'trainer_type': 'ppo',
            },
            'OTHER_BEHAVIOR_NAME': {
                'hyperparameters': {'batch_size': 1024, 'buffer_size': 10240, 'epsilon': 0.2},
                'trainer_type': 'ppo',
            },
        }
    }


def test_buffer_size_multiple(search_config, trainer_config):
    """Tests that 'buffer_size' is correctly calculated if 'buffer_size_multiple' is present and that 'buffer_size_multiple' is stripped from the brain_config."""

    search_config['search_parameters']['hyperparameters.buffer_size_multiple'] = [4]

    search = GridSearch(search_config, trainer_config)
    search_overrides = search.get_search_configuration(0)
    trainer_config = search.get_trainer_config_with_overrides(search_overrides)

    assert 'buffer_size_multiple' not in trainer_config['behaviors']['BEHAVIOR_NAME']

    assert trainer_config == {
        'behaviors': {
            'BEHAVIOR_NAME': {
                'hyperparameters': {
                    'batch_size': 1024,
                    'beta': 0.0001,
                    'buffer_size': 4096,
                    'epsilon': 0.2,
                    'lambd': 0.95,
                    'learning_rate': 1e-05,
                    'num_epoch': 3,
                },
                'network_settings': {'hidden_units': 32, 'num_layers': 1},
                'reward_signals': {'extrinsic': {'gamma': 0.99, 'strength': 1.0}},
                'trainer_type': 'ppo',
            },
            'OTHER_BEHAVIOR_NAME': {
                'hyperparameters': {'batch_size': 1024, 'buffer_size': 10240, 'epsilon': 0.2},
                'trainer_type': 'ppo',
            },
        }
    }


def test_get_random_value():
    """Test for the correct randomization of ints and floats."""

    assert RandomSearch.get_random_value([1, 5, 8], seed=10) == 1
    assert RandomSearch.get_random_value([0.01, 0.0001, 1], seed=5) == 0.6229394047202129


def test_get_random_search(search_config, trainer_config):
    """Tests for the correct generation of a randomized search configuration."""

    search = RandomSearch(search_config, trainer_config)
    random_search_config = search.get_randomized_search_configuration(seed=9871237)

    assert random_search_config == {
        'hyperparameters.beta': 0.008715030393329336,
        'hyperparameters.learning_rate': 0.0008715030393329336,
        'hyperparameters.num_epoch': 4,
        'network_settings.hidden_units': 477,
        'network_settings.num_layers': 1,
    }


def test_get_parameter_bounds():
    """Tests for the correct construction of a parameter bounds dictionary."""

    parameter_names = ['batch_size', 'buffer_size_multiple', 'beta']
    parameter_values = [[64, 128], [4], [0.001, 0.0001]]

    result = BayesianSearch.get_parameter_bounds(parameter_names, parameter_values)

    # Test for a dictionary return type
    assert type(result) is dict

    # Test for a second value inserted for any parameter that only contains one value
    assert result == {
        'batch_size': [64, 128],
        'buffer_size_multiple': [4, 4],
        'beta': [0.001, 0.0001],
    }


def test_get_search_config_from_bounds():
    """Tests that
    - Only standard Python value types are returned
    - Values that should be int are converted to int
    - The item() method is not called on non-numpy value types
    """

    bounds = {
        'hyperparameters.batch_size': numpy.float64(144.0682249028942),
        'hyperparameters.beta': numpy.float64(0.0028687875149226343),
        'hyperparameters.buffer_size_multiple': numpy.float64(50.017156222601734),
        'hyperparameters.num_epoch': numpy.float64(5.028942),
        'network_settings.hidden_units': numpy.float64(121.0682249028942),
        'network_settings.num_layers': numpy.float64(2.49028942),
        'network_settings.memory.memory_size': numpy.float64(154.9019),
        'network_settings.memory.sequence_length': numpy.float64(144.9028),
        'reward_signal.extrinsic.strength': numpy.float64(1.0),
        'reward_signal.strength.encoding_size': numpy.float64(184.6824928),
        'reward_signal.curiosity.encoding_size': numpy.float64(184.6824928),
        'reward_signal.gail.encoding_size': numpy.float64(184.6824928),
        'max_steps': numpy.float64(500000.1),
        'time_horizon': numpy.float64(64.049292),
    }

    assert BayesianSearch.get_search_config_from_bounds(bounds) == {
        'hyperparameters.batch_size': 144,
        'hyperparameters.beta': 0.0028687875149226343,
        'hyperparameters.buffer_size_multiple': 50,
        'hyperparameters.num_epoch': 5,
        'max_steps': 500000,
        'network_settings.hidden_units': 121,
        'network_settings.memory.memory_size': 152,
        'network_settings.memory.sequence_length': 145,
        'network_settings.num_layers': 2,
        'reward_signal.curiosity.encoding_size': 185,
        'reward_signal.extrinsic.strength': 1.0,
        'reward_signal.gail.encoding_size': 185,
        'reward_signal.strength.encoding_size': 185,
        'time_horizon': 64,
    }
