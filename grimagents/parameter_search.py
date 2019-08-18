import itertools
import random


class GridSearchError(Exception):
    pass


class InvalidTrainerConfig(GridSearchError):
    """The trainer config yaml file is invalid."""

    pass


class InvalidIntersectIndex(GridSearchError):
    """An attempt to access an invalid intersection index was made."""

    pass


class ParameterSearch:
    """Object that facilitates performing hyperparameter searches."""

    def __init__(self, search_config, trainer_config):
        self.search_config = None
        self.trainer_config = None
        self.brain_name = ''
        self.hyperparameters = []
        self.hyperparameter_sets = []
        self.brain_config = {}

        self.set_search_config(search_config)
        self.set_trainer_config(trainer_config)

    def set_search_config(self, search_config):

        self.search_config = search_config.copy()
        self.brain_name = self.search_config['brain']['name']
        self.hyperparameters = self.get_search_hyperparameters(self.search_config)
        self.hyperparameter_sets = self.get_hyperparameter_sets(self.search_config)

    @staticmethod
    def get_search_hyperparameters(search_config):
        """Returns the list of hyperparameter names defined in search configuration."""

        return [name for name in search_config['brain']['hyperparameters']]

    @staticmethod
    def get_hyperparameter_sets(search_config):
        """Returns a two dimensional array containing all hyperparameter values to use in the grid search."""

        search_sets = []
        for _, values in search_config['brain']['hyperparameters'].items():
            search_sets.append(values)

        return search_sets

    def set_trainer_config(self, trainer_config):

        self.trainer_config = trainer_config.copy()
        self.brain_config = self.get_brain_configuration(self.trainer_config, self.brain_name)

    @staticmethod
    def get_brain_configuration(trainer_config, brain_name):
        """Returns a complete trainer configuration for a brain, including default parameters.

        This method essentially strips all configuration values except 'default' and 'brain_name' from 'trainer_config'.

        Raises:
          InvalidTrainerConfig: Raised if 'default' or 'brain_name' parameters aren't found as keys in the trainer_config
        """

        try:
            result = {'default': trainer_config['default']}
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

    def get_brain_config_for_intersect(self, intersect):
        """Returns a brain configuration dictionary with values overridden by the grid search intersect.
        """

        result = self.brain_config.copy()
        for key, value in intersect.items():
            result[self.brain_name][key] = value

        # Set 'buffer_size' based on 'buffer_size_multiple', if present
        if 'buffer_size_multiple' in result[self.brain_name]:
            batch_size = self.get_batch_size_value(result, self.brain_name)
            result[self.brain_name]['buffer_size'] = (
                batch_size * result[self.brain_name]['buffer_size_multiple']
            )
            del result[self.brain_name]['buffer_size_multiple']

        return result

    @staticmethod
    def get_batch_size_value(brain_config, brain_name):
        """Returns the 'batch_size' value in a 'brain_config' dictionary. If the specified 'brain_name' does not contain an entry for 'batch_size', the 'default' value is returned instead.
        """

        if 'batch_size' in brain_config[brain_name]:
            return brain_config[brain_name]['batch_size']

        return brain_config['default']['batch_size']


class GridSearch(ParameterSearch):
    """Object that facilitates performing hyperparameter grid searches."""

    def __init__(self, search_config, trainer_config):

        super().__init__(search_config, trainer_config)
        self.search_permutations = self.get_search_permutations(self.hyperparameter_sets)

    @staticmethod
    def get_search_permutations(hyperparameter_sets):
        """Returns a two dimensional list of grid search permutations."""

        return list(itertools.product(*hyperparameter_sets))

    def get_intersect(self, index):
        """Returns a dictionary containing the search parameters to use for a given intersect (index) in the grid search.

        Raises:
          InvalidIntersectIndex: Raised if the 'index' parameter exceeds the number of search permutations the GridSearch contains.
        """

        try:
            result = dict(zip(self.hyperparameters, self.search_permutations[index]))
        except IndexError:
            raise InvalidIntersectIndex(
                f'Unable to access intersection {index}, GridSearch only contains {self.get_intersect_count()} intersections.'
            )

        return result

    def get_intersect_count(self):
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

    def get_randomized_intersect(self, seed=None):
        """Returns an intersect with randomized values. Values are chosen between the minimum and maximum values that exist for each hyperparameter."""

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
    def sanitize_parameter_values(bounds: dict):
        """Enforces int type on parameters that should be int and ensures native value types are used.

        BayesianOptimization objects return numpy floats, which cause problems with yaml serialization.
        """

        for key, value in bounds.items():
            if (
                key == 'batch_size'
                or key == 'buffer_size_multiple'
                or key == 'hidden_units'
                or key == 'num_epoch'
                or key == 'num_layers'
                or key == 'time_horizon'
                or key == 'sequence_length'
                or key == 'curiosity_enc_size'
            ):
                bounds[key] = int(value)
                continue

            bounds[key] = value.item()

        return bounds
