import logging
import re
import subprocess
import sys

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from pathlib import Path

import grimagents.command_util as command_util
import grimagents.config as config_util
import grimagents.common as common
import grimagents.settings as settings

from grimagents.parameter_search import GridSearch, RandomSearch, BayesianSearch


search_log = logging.getLogger('grimagents.search')
reward_regex = re.compile(r'Final Mean Reward: (\d*[.,]?\d*)')


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


class SearchCommand(Command):
    def __init__(self, args):

        self.args = args

        # Gather configurations
        self.grim_config_path = Path(args.configuration_file)
        self.grim_config = config_util.load_grim_config_file(self.grim_config_path)

        self.search_config = self.grim_config[config_util.SEARCH]

        self.trainer_config_path = Path(self.grim_config['trainer-config-path'])
        self.trainer_config = config_util.load_trainer_configuration(self.trainer_config_path)

        self.search_config_path = self.trainer_config_path.with_name('search_config.yaml')

        self.search_counter = 0

    def perform_search_with_configuration(self, intersect, search_brain_config):
        """Executes a search using the provided intersect and matching brain_config."""

        # Write trainer configuration file for current intersect
        if self.args.parallel:
            self.search_config_path = self.search_config_path.with_name(
                f'search_config{self.search_counter:02d}.yaml'
            )

        command_util.write_yaml_file(search_brain_config, self.search_config_path)

        # Execute training with the intersect config and run_id
        run_id = self.grim_config[config_util.RUN_ID] + f'_{self.search_counter:02d}'
        command = [
            'pipenv',
            'run',
            'python',
            '-m',
            'grimagents',
            self.grim_config_path,
            '--trainer-config',
            self.search_config_path,
            '--run-id',
            run_id,
        ]

        search_log.info('-' * 63)
        search_log.info(f'Search: {run_id}')
        for key, value in intersect.items():
            search_log.info(f'    {key}: {value}')
        search_log.info('-' * 63)

        if self.args.parallel:
            command = ['cmd', '/K'] + command
            base_port = self.grim_config.get('--base-port', 5005)
            command = command + ['--base-port', base_port + self.search_counter]
            command = [str(element) for element in command]
            subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)

        else:
            command = [str(element) for element in command]
            subprocess.run(command)


class GridSearchCommand(SearchCommand):
    def __init__(self, args):

        super().__init__(args)
        self.grid_search = GridSearch(self.search_config, self.trainer_config)


class OutputGridSearchCount(GridSearchCommand):
    """Prints out the total number of training runs a grimagents configuration file will attempt.
    """

    def execute(self):

        search_log.info(
            f'\'{self.trainer_config_path}\' will perform {self.grid_search.get_intersect_count()} training runs'
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
            self.search_counter = i

            self.perform_search_with_configuration(intersect, intersect_brain_config)

        if not self.args.parallel and self.search_config_path.exists():
            self.search_config_path.unlink()

        if not self.args.parallel:
            search_log.info('Grid search complete\n')


class ExportGridSearchConfiguration(GridSearchCommand):
    """Exports a trainer config file for a given GridSearch intersect."""

    def execute(self):

        search_log.info(
            f'Exporting trainer configuration for GridSearch intersect \'{self.args.export_index}\' into \'{self.search_config_path}\''
        )

        intersect = self.grid_search.get_intersect(self.args.export_index)
        intersect_brain_config = self.grid_search.get_brain_config_for_intersect(intersect)
        command_util.write_yaml_file(intersect_brain_config, self.search_config_path)


class PerformRandomSearch(SearchCommand):
    def __init__(self, args):

        super().__init__(args)
        self.random_search = RandomSearch(self.search_config, self.trainer_config)

    def execute(self):

        search_log.info('-' * 63)
        search_log.info('Performing random search for hyperparameters:')
        for i in range(len(self.random_search.hyperparameters)):
            search_log.info(
                f'    {self.random_search.hyperparameters[i]}: {self.random_search.hyperparameter_sets[i]}'
            )
        search_log.info('-' * 63)

        for i in range(self.args.random):

            intersect = self.random_search.get_randomized_intersect()
            intersect_brain_config = self.random_search.get_brain_config_for_intersect(intersect)
            self.search_counter = i

            self.perform_search_with_configuration(intersect, intersect_brain_config)

        if not self.args.parallel and self.search_config_path.exists():
            self.search_config_path.unlink()

        if not self.args.parallel:
            search_log.info('Random search complete\n')


class PerformBayesianSearch(SearchCommand):
    def __init__(self, args):

        super().__init__(args)
        self.bayes_search = BayesianSearch(self.search_config, self.trainer_config)
        self.output_config_path = self.trainer_config_path.with_name('bayes_config.yaml')

    def execute(self):

        if self.args.parallel:
            search_log.warning(
                'The \'--parallel\' argument is not compatible with Bayesian Search and will be ignored.'
            )

        search_log.info('-' * 63)
        search_log.info('Performing bayesian search for hyperparameters:')
        for i in range(len(self.bayes_search.hyperparameters)):
            search_log.info(
                f'    {self.bayes_search.hyperparameters[i]}: {self.bayes_search.hyperparameter_sets[i]}'
            )
        search_log.info('-' * 63)

        # Load bounds from configuration and create an optimization object
        bounds = self.bayes_search.get_parameter_bounds(
            self.bayes_search.hyperparameters, self.bayes_search.hyperparameter_sets
        )

        optimizer = BayesianOptimization(
            f=self.perform_bayes_search, pbounds=bounds, random_state=1, verbose=0
        )

        # Save search observations to log file
        if self.args.bayes_save:
            bayes_log_path = self.get_bayes_log_path()
            search_log.info(f'Saving Bayesian optimization observations to \'{bayes_log_path}\'')
            bayes_logger = JSONLogger(path=str(bayes_log_path))
            optimizer.subscribe(Events.OPTMIZATION_STEP, bayes_logger)

        # Load search observations from log files
        if self.args.bayes_load:
            log_files_list = self.find_bayes_log_paths()
            search_log.info(f'Loading Bayesian optimization observations from:')
            for log in log_files_list:
                search_log.info(f'{str(log)}')

            load_logs(optimizer, logs=log_files_list)

        # Perform Bayesian searches
        optimizer.maximize(init_points=self.args.bayesian[0], n_iter=self.args.bayesian[1])

        search_log.info('-' * 63)
        search_log.info(f'Best Configuration ({optimizer.max["target"]}):')

        best_intersect = self.bayes_search.sanitize_parameter_values(optimizer.max["params"])
        for key, value in best_intersect.items():
            search_log.info(f'    {key}: {value}')

        search_log.info('-' * 63)
        self.save_max_to_file(optimizer.max)
        search_log.info('-' * 63)

    def perform_bayes_search(self, **kwargs):
        """Executes a training run using the provided intersect and returns the final mean reward."""

        intersect = self.bayes_search.sanitize_parameter_values(kwargs)
        bayes_brain_config = self.bayes_search.get_brain_config_for_intersect(intersect)
        command_util.write_yaml_file(bayes_brain_config, self.search_config_path)

        # # Execute training with the intersect config and run_id
        run_id = self.grim_config[config_util.RUN_ID] + f'_{self.search_counter:02d}'
        command = [
            'pipenv',
            'run',
            'python',
            '-m',
            'grimagents',
            self.grim_config_path,
            '--trainer-config',
            self.search_config_path,
            '--run-id',
            run_id,
        ]

        search_log.info('-' * 63)
        search_log.info(f'Search: {run_id}')
        for key, value in intersect.items():
            search_log.info(f'    {key}: {value}')
        search_log.info('-' * 63)

        command = [str(element) for element in command]
        subprocess.run(command)

        self.search_counter += 1
        return self.get_last_mean_reward_from_log()

    @staticmethod
    def get_last_mean_reward_from_log():
        """Returns the last Final Mean Reward value recorded in the grimagents log file,
        or 0 if no value is found.
        """

        log_file = settings.get_log_file_path()
        last_lines = command_util.load_last_lines_from_file(log_file, 50)

        reward = 0
        for line in last_lines:
            match = reward_regex.search(line)
            if match:
                reward = match.group(1)

        return float(reward)

    def save_max_to_file(self, max: dict):
        """Sanitizes a BayesianOptimization object's max parameter and writes it to a brain configuration file.
        """

        search_log.info(f'Saving best configuration to \'{self.output_config_path}\'')

        intersect = self.bayes_search.sanitize_parameter_values(max['params'])
        best_config = self.bayes_search.get_brain_config_for_intersect(intersect)
        command_util.write_yaml_file(best_config, self.output_config_path)

    def get_bayes_log_path(self):
        """Generates a timestamped log file name for Bayesian optimization observations."""

        log_folder_path = (
            self.trainer_config_path.parent / f'{self.grim_config[config_util.RUN_ID]}_bayes'
        )

        log_file_path = (
            log_folder_path
            / f'{self.grim_config[config_util.RUN_ID]}_{common.get_timestamp()}.json'
        )

        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True)

        return log_file_path

    def find_bayes_log_paths(self):
        """Returns a list of all json files in the Bayesian optimization observation logs folder.
        """

        log_folder_path = log_folder_path = (
            self.trainer_config_path.parent / f'{self.grim_config[config_util.RUN_ID]}_bayes'
        )
        log_file_list = log_folder_path.glob('*.json')

        return list(log_file_list)