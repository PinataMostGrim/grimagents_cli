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

        # brain_config contains only the brain configuration and defaults
        self.brain_config = {}

        # search_config contains hyperparameters to search, as well as a range of values to search through for each
        self.set_search_config(search_config)

        self.set_trainer_config(trainer_config)

    def set_search_config(self, search_config):

        self.search_config = search_config.copy()
        self.brain_name = self.search_config[const.GS_BRAIN][const.GS_NAME]
        self.hyperparameters = self.get_search_hyperparameters(self.search_config)
        self.hyperparameter_sets = self.get_hyperparameter_sets(self.search_config)

    @staticmethod
    def get_search_hyperparameters(search_config):
        """Returns the list of hyperparameter names defined in search configuration."""

        return [name for name in search_config[const.GS_BRAIN][const.GS_HYPERPARAMS]]

    @staticmethod
    def get_hyperparameter_sets(search_config):
        """Returns an array containing all sets of hyperparameter values to use in the search."""

        search_sets = []
        for _, values in search_config[const.GS_BRAIN][const.GS_HYPERPARAMS].items():
            search_sets.append(values)

        return search_sets

    def set_trainer_config(self, trainer_config):
        """Assigns the trainer configuration for this parameter search and isolates
        a copy of the brain configuration we are concerned with, paired with default
        values.
        """

        self.trainer_config = trainer_config.copy()
        self.brain_config = self.extract_brain_config(self.trainer_config, self.brain_name)

    @staticmethod
    def extract_brain_config(trainer_config, brain_name):
        """Returns a complete trainer configuration for a brain, including default parameters.

        This method strips all configuration values except 'default' and 'brain_name' from 'trainer_config'.

        Raises:
          InvalidTrainerConfig: Raised if 'default' or 'brain_name' parameters aren't found as keys in the trainer_config
        """

        try:
            result = {const.TC_DEFAULT: trainer_config[const.TC_DEFAULT]}
        except KeyError:
            raise InvalidTrainerConfig(
                f'Unable to find \'default\' configuration values in trainer configuration:\n{trainer_config}'
            )

        try:
            brain_data = trainer_config[brain_name]
        except KeyError:
            raise InvalidTrainerConfig(
                f'Unable to find configuration settings for brain \'{brain_name}\' in trainer configuration:{trainer_config}'
            )

        result[brain_name] = {}
        for key, value in brain_data.items():
            result[brain_name][key] = value

        return result

    def get_brain_config_with_overrides(self, overrides):
        """Returns a copy of the brain configuration with the specified values overriden.

        Parameters:
            overrides: dict: A dictionary containing hyperparameter names paired with override values
        """

        result = self.brain_config.copy()
        for key, value in overrides.items():
            common.add_nested_dict_value(result[self.brain_name], key, value)

        # Set 'buffer_size' based on 'buffer_size_multiple', if present
        if const.GS_BUFFER_SIZE_MULTIPLE in result[self.brain_name]:
            batch_size = self.get_batch_size_value(result, self.brain_name)
            result[self.brain_name][const.HP_BUFFER_SIZE] = (
                batch_size * result[self.brain_name][const.GS_BUFFER_SIZE_MULTIPLE]
            )
            del result[self.brain_name][const.GS_BUFFER_SIZE_MULTIPLE]

        return result

    @staticmethod
    def get_batch_size_value(brain_config, brain_name):
        """Returns the 'batch_size' value in a 'brain_config' dictionary. If the specified 'brain_name' does not contain an entry for 'batch_size', the 'default' value is returned instead.
        """

        if const.HP_BATCH_SIZE in brain_config[brain_name]:
            return brain_config[brain_name][const.HP_BATCH_SIZE]

        return brain_config[const.TC_DEFAULT][const.HP_BATCH_SIZE]


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
        """Determines the minimum and maximum values in a range of values and picks a random value inside that range (inclusive). Returns a float if any of the values are floats and returns an int value otherwise.
        """

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
    def enforce_parameter_value_types(bounds: dict):
        """Enforces int type on parameters that should be int and ensures native value types are used for the rest.

        Converts values to standard Python value types. BayesianOptimization objects return numpy floats and numpy floats cause problems with yaml serialization.
        """

        for key, value in bounds.items():
            if (
                key == const.HP_BATCH_SIZE
                # 'const.GS_BUFFER_SIZE' should never be used for searches. Use 'const.GS_BUFFER_SIZE_MULTIPLE' instead.
                or key == const.GS_BUFFER_SIZE_MULTIPLE
                or key == const.HP_HIDDEN_UNITS
                or key == const.HP_MAX_STEPS
                or key == const.HP_NUM_EPOCH
                or key == const.HP_NUM_LAYERS
                or key == const.HP_TIME_HORIZON
                or key == const.HP_SEQUENCE_LENGTH
            ):
                bounds[key] = int(value)
                continue

            if isinstance(value, numpy.generic):
                bounds[key] = value.item()

        return bounds
