import itertools
import numpy
import random

import grimagents.common as common
import grimagents.constants as const


class InvalidTrainerConfig(Exception):
    """The trainer config yaml file is invalid."""

    pass


class InvalidGridSearchIndex(Exception):
    """An attempt to access an invalid GridSearch index was made."""

    pass


class ParameterSearch:
    """Object that facilitates performing hyperparameter searches."""

    def __init__(self, search_config, trainer_config):
        self.search_config = None

        # trainer_config potentially contains many embedded brain configurations and defaults.
        self.trainer_config = None

        # brain_name contains the specific brain this parameter search is concerned with
        self.brain_name = ''

        self.hyperparameters = []
        self.hyperparameter_sets = []

        # 'search_config' contains hyperparameters to search, as well as a range of values to search through for each parameter
        self.set_search_config(search_config)

        self.set_trainer_config(trainer_config)

    def set_search_config(self, search_config):

        self.search_config = search_config.copy()
        self.brain_name = self.search_config['behavior_name']
        self.hyperparameters = self.get_search_hyperparameters(self.search_config)
        self.hyperparameter_sets = self.get_hyperparameter_sets(self.search_config)

    @staticmethod
    def get_search_hyperparameters(search_config):
        """Returns the list of hyperparameter names defined in search configuration."""

        return [name for name in search_config['search_parameters']]

    @staticmethod
    def get_hyperparameter_sets(search_config):
        """Returns an array containing all sets of hyperparameter values to use in the search."""

        search_sets = []
        for _, values in search_config['search_parameters'].items():
            search_sets.append(values)

        return search_sets

    def set_trainer_config(self, trainer_config):
        """Assigns the trainer configuration for this parameter search and isolates
        a copy of the brain configuration we are concerned with, paired with default
        values.
        """

        self.trainer_config = trainer_config.copy()

    def get_trainer_config_with_overrides(self, overrides):
        """Returns a copy of the brain configuration with the specified values overriden.

        Parameters:
            overrides: dict: A dictionary containing hyperparameter names paired with override values
        """

        result = self.trainer_config.copy()
        for key, value in overrides.items():
            common.add_nested_dict_value(result['behaviors'][self.brain_name], key, value)

        # Set 'buffer_size' based on 'buffer_size_multiple', if present
        if const.GS_BUFFER_SIZE_MULTIPLE in result['behaviors'][self.brain_name]['hyperparameters']:
            batch_size = self.get_batch_size_value(result, self.brain_name)

            result['behaviors'][self.brain_name]['hyperparameters'][const.HP_BUFFER_SIZE] = (
                batch_size
                * result['behaviors'][self.brain_name]['hyperparameters'][
                    const.GS_BUFFER_SIZE_MULTIPLE
                ]
            )
            del result['behaviors'][self.brain_name]['hyperparameters'][
                const.GS_BUFFER_SIZE_MULTIPLE
            ]

        return result

    @staticmethod
    def get_batch_size_value(training_config, behavior_name):
        """Returns the 'batch_size' value from a trainer configuration for the given behavior name."""

        return training_config['behaviors'][behavior_name]['hyperparameters'][const.HP_BATCH_SIZE]


class GridSearch(ParameterSearch):
    """Object that facilitates performing hyperparameter grid searches."""

    def __init__(self, search_config, trainer_config):

        super().__init__(search_config, trainer_config)
        self.search_permutations = self.get_search_permutations(self.hyperparameter_sets)

    @staticmethod
    def get_search_permutations(hyperparameter_sets):
        """Returns a two dimensional list of grid search permutations."""

        return list(itertools.product(*hyperparameter_sets))

    def get_search_configuration(self, index):
        """Returns a dictionary containing the search parameters to use for a GridSearch by search index.

        Raises:
          InvalidGridSearchIndex: Raised if the 'index' parameter exceeds the number of search permutations the GridSearch contains.
        """

        try:
            result = dict(zip(self.hyperparameters, self.search_permutations[index]))
        except IndexError:
            raise InvalidGridSearchIndex(
                f'Unable to access GridSearch index \'{index}\', GridSearch only contains {self.get_grid_search_count()} elements.'
            )

        return result

    def get_grid_search_count(self):
        """Returns the total count of search permutations for this GridSearch."""

        return len(self.search_permutations)


class RandomSearch(ParameterSearch):
    """Object that facilitates performing hyperparameter random searches."""

    @staticmethod
    def get_random_value(values, seed=None):
        """Determines the minimum and maximum values in a range of values and picks a random value inside that range (inclusive). Returns a float if any of the values are floats and returns an int value otherwise."""

        if seed is not None:
            random.seed(seed)

        for element in values:
            # If values contain one or more floats, return a random float
            if isinstance(element, float):
                return random.uniform(min(values), max(values))

        return random.randint(min(values), max(values))

    def get_randomized_search_configuration(self, seed=None):
        """Returns a search configuration with randomized values. Values are chosen between the minimum and maximum values that exist for each hyperparameter."""

        randomized_hyperparameters = []
        for i in range(len(self.hyperparameters)):
            randomized_hyperparameters.append(
                self.get_random_value(self.hyperparameter_sets[i], seed=seed)
            )

        result = dict(zip(self.hyperparameters, randomized_hyperparameters))
        return result


class BayesianSearch(ParameterSearch):
    """Object that facilitates performing hyperparameter bayesian searches."""

    @staticmethod
    def get_parameter_bounds(parameter_names, parameter_values):
        """Returns a parameter bounds dictionary for consumption by a BayesianOptimization object.

        Parameters:
            parameter_names: list: A list of hyperparameter names
            parameter_values: list: A list of hyperparameter values
        """

        bounds = {}
        for i in range(len(parameter_names)):

            # The BayesianOptimization object requires two values for every parameter
            # so we duplicate an existing element if only one is present.
            if len(parameter_values[i]) < 2:
                parameter_values[i].append(parameter_values[i][0])

            bounds[parameter_names[i]] = parameter_values[i]

        return bounds

    @staticmethod
    def get_search_config_from_bounds(bounds: dict):
        """Enforces int type on parameters that should be int and ensures native value types are used for the rest.

        Converts values to standard Python value types. BayesianOptimization objects return numpy floats and numpy floats cause problems with yaml serialization.
        """

        for key, value in bounds.items():

            # Convert bound items that must be ints into int
            if (
                key == 'hyperparameters.batch_size'
                # 'buffer_size' should never be used for searches. Use 'buffer_size_multiple' instead.
                or key == 'hyperparameters.buffer_size_multiple'
                or key == 'network_settings.hidden_units'
                or key == 'max_steps'
                or key == 'hyperparameters.num_epoc'
                or key == 'network_settings.num_layers'
                or key == 'time_horizon'
                or key == 'network_settings.memory.sequence_length'
            ):
                bounds[key] = int(round(value))
                continue

            # Ensure 'memory_size' is a multiple of 4 and an int
            if key == 'network_settings.memory.memory_size':
                bounds[key] = int(round(bounds[key] - bounds[key] % 4))
                continue

            # Ensure all 'encoding_size' values are rounded to int
            splitKey = key.rsplit('.', maxsplit=1)
            if (len(splitKey) > 1) and (splitKey[-1] == 'encoding_size'):
                bounds[key] = int(round(value))
                continue

            if isinstance(value, numpy.generic):
                bounds[key] = value.item()

        return bounds
