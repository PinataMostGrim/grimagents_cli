import numpy
import pytest

from grimagents.parameter_search import (
    GridSearch,
    RandomSearch,
    BayesianSearch,
    InvalidTrainerConfig,
    InvalidGridSearchIndex,
)


@pytest.fixture
def search_config():
    return {
        'brain': {
            'name': 'BRAIN_NAME',
            'hyperparameters': {
                'beta': [1e-4, 1e-2],
                'hidden_units': [32, 512],
                'learning_rate': [1e-5, 1e-3],
                'num_layers': [1, 3],
                'num_epoch': [3, 10],
            },
        }
    }


@pytest.fixture
def trainer_config():
    return {
        'default': {
            'trainer': 'ppo',
            'batch_size': 1024,
            'beta': 5.0e-3,
            'buffer_size': 10240,
            'epsilon': 0.2,
            'gamma': 0.99,
            'hidden_units': 128,
            'lambd': 0.95,
        },
        'BRAIN_NAME': {'beta': 5.0e-3, 'epsilon': 0.2},
        'OTHER_BRAIN_NAME': {'batch_size': 1024, 'buffer_size': 10240, 'epsilon': 0.2},
    }


def test_get_search_hyperparameters(search_config):
    """Test for the extraction of hyperparameter names from a search configuration dictionary."""

    assert GridSearch.get_search_hyperparameters(search_config) == [
        'beta',
        'hidden_units',
        'learning_rate',
        'num_layers',
        'num_epoch',
    ]


def test_get_hyperparameter_sets(search_config):
    """Test for the correct construction of GridSearch hyperparameter sets."""

    assert GridSearch.get_hyperparameter_sets(search_config) == [
        [0.0001, 0.01],
        [32, 512],
        [1e-05, 0.001],
        [1, 3],
        [3, 10],
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


def test_get_brain_configuration(trainer_config):
    """Tests for the correct creation of the GridSearch brain configuration from a trainer configuration."""

    assert GridSearch.get_brain_configuration(trainer_config, 'BRAIN_NAME') == {
        'default': {
            'trainer': 'ppo',
            'batch_size': 1024,
            'beta': 0.005,
            'buffer_size': 10240,
            'epsilon': 0.2,
            'gamma': 0.99,
            'hidden_units': 128,
            'lambd': 0.95,
        },
        'BRAIN_NAME': {'beta': 0.005, 'epsilon': 0.2},
    }


def test_get_search_configuration(search_config, trainer_config):
    """Tests for the correct creation of GridSearch intersect configurations."""

    search = GridSearch(search_config, trainer_config)

    assert search.get_search_configuration(0) == {
        'beta': 0.0001,
        'hidden_units': 32,
        'learning_rate': 1e-05,
        'num_layers': 1,
        'num_epoch': 3,
    }

    assert search.get_search_configuration(15) == {
        'beta': 0.0001,
        'hidden_units': 512,
        'learning_rate': 0.001,
        'num_layers': 3,
        'num_epoch': 10,
    }

    assert search.get_search_configuration(31) == {
        'beta': 0.01,
        'hidden_units': 512,
        'learning_rate': 0.001,
        'num_layers': 3,
        'num_epoch': 10,
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


def test_invalid_trainer_config(trainer_config):
    """Test for the correct raising of InvalidTrainerConfig exceptions."""

    # Test for configuration missing 'brain_name'
    with pytest.raises(InvalidTrainerConfig):
        assert GridSearch.get_brain_configuration(trainer_config, 'DOES_NOT_EXIST') == {
            'default': {
                'trainer': 'ppo',
                'batch_size': 1024,
                'beta': 0.005,
                'buffer_size': 10240,
                'epsilon': 0.2,
                'gamma': 0.99,
                'hidden_units': 128,
                'lambd': 0.95,
            },
            'BRAIN_NAME': {'beta': 0.005, 'epsilon': 0.2},
        }

    # Test for configuration missing 'default'
    del trainer_config['default']
    with pytest.raises(InvalidTrainerConfig):
        assert GridSearch.get_brain_configuration(trainer_config, 'BRAIN_NAME') == {
            'default': {
                'trainer': 'ppo',
                'batch_size': 1024,
                'beta': 0.005,
                'buffer_size': 10240,
                'epsilon': 0.2,
                'gamma': 0.99,
                'hidden_units': 128,
                'lambd': 0.95,
            },
            'BRAIN_NAME': {'beta': 0.005, 'epsilon': 0.2},
        }


def test_buffer_size_multiple(search_config, trainer_config):
    """Tests that 'buffer_size' is correctly calculated if 'buffer_size_multiple' is present and that 'buffer_size_multiple' is stripped from the brain_config.
    """

    search_config['brain']['hyperparameters']['buffer_size_multiple'] = [4]

    search = GridSearch(search_config, trainer_config)
    intersect = search.get_search_configuration(0)
    intersect_config = search.get_brain_config_with_overrides(intersect)

    assert 'buffer_size_multiple' not in intersect_config['BRAIN_NAME']

    assert intersect_config == {
        'default': {
            'trainer': 'ppo',
            'batch_size': 1024,
            'beta': 0.005,
            'buffer_size': 10240,
            'epsilon': 0.2,
            'gamma': 0.99,
            'hidden_units': 128,
            'lambd': 0.95,
        },
        'BRAIN_NAME': {
            'beta': 0.0001,
            'epsilon': 0.2,
            'hidden_units': 32,
            'learning_rate': 1e-05,
            'num_layers': 1,
            'num_epoch': 3,
            'buffer_size': 4096,
        },
    }


def test_get_random_value():
    """Test for the correct randomization of ints and floats."""

    assert RandomSearch.get_random_value([1, 5, 8], seed=10) == 1
    assert RandomSearch.get_random_value([0.01, 0.0001, 1], seed=5) == 0.6229394047202129


def test_get_random_intersect(search_config, trainer_config):
    """Tests for the correct generation of a randomized intersect."""

    search = RandomSearch(search_config, trainer_config)
    random_intersect = search.get_randomized_search_configuration(seed=9871237)

    assert random_intersect == {
        'beta': 0.008715030393329336,
        'hidden_units': 477,
        'learning_rate': 0.0008715030393329336,
        'num_layers': 1,
        'num_epoch': 4,
    }


def test_get_parameter_bounds():
    """Tests for the correct construction of a parameter bounds dictionary.
    """

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


def test_sanitize_parameter_values():
    """Tests that
        - Only standard Python value types are returned
        - Values that should be int are converted to int
        - The item() method is not called on non-numpy value types
    """

    bounds = {
        'batch_size': numpy.float64(144.0682249028942),
        'beta': numpy.float64(0.0028687875149226343),
        'buffer_size_multiple': numpy.float64(50.017156222601734),
        'hidden_units': numpy.float64(121.0682249028942),
        'num_epoch': numpy.float64(5.028942),
        'max_steps': numpy.float64(500000.1),
        'num_layers': numpy.float64(2.49028942),
        'time_horizon': numpy.float64(64.049292),
        'sequence_length': numpy.float64(144.9028),
        'curiosity_enc_size': numpy.float64(184.6824928),
        'curiosity_strength': 0.001,
    }

    assert BayesianSearch.sanitize_parameter_values(bounds) == {
        'batch_size': 144,
        'beta': 0.0028687875149226343,
        'buffer_size_multiple': 50,
        'hidden_units': 121,
        'num_epoch': 5,
        'max_steps': 500000,
        'num_layers': 2,
        'time_horizon': 64,
        'sequence_length': 144,
        'curiosity_enc_size': 184,
        'curiosity_strength': 0.001,
    }
