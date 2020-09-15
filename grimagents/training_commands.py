from argparse import Namespace
from pathlib import Path

import grimagents.common as common
import grimagents.command_util as command_util
import grimagents.config as config_util
import grimagents.constants as const
import grimagents.settings as settings


class Command:
    def __init__(self, args: Namespace):
        self.args = args
        self.dry_run = args.dry_run
        self.show_command = True

    def execute(self):
        command = self.create_command()
        command_util.execute_command(command, show_command=self.show_command, dry_run=self.dry_run)

    def create_command(self):
        return ['cmd', '/K', 'echo', self.__class__.__name__, repr(self.args)]


class ListTrainingOptions(Command):
    """Outputs mlagents-learn usage options."""

    def create_command(self):
        return ['pipenv', 'run', 'mlagents-learn', '--help']


class EditGrimConfigFile(Command):
    """Opens a grimagents configuration file for editing or creates one if
    a file does not already exist."""

    def execute(self):
        file_path = Path(self.args.edit_config)
        config_util.edit_grim_config_file(file_path)


class EditTrainerConfigFile(Command):
    """Opens a trainer configuration file for editing or creates one if
    a file does not already exist.
    """

    def execute(self):
        file_path = Path(self.args.edit_trainer_config)
        config_util.edit_trainer_configuration_file(file_path)


class EditCurriculumFile(Command):
    """Opens a curriculum file for editing or creates one if a file does
    not already exist.
    """

    def execute(self):
        file_path = Path(self.args.edit_curriculum)
        config_util.edit_curriculum_file(file_path)


class StartTensorboard(Command):
    """Starts a new instance of tensorboard server."""

    def __init__(self, args):
        super().__init__(args)

    def create_command(self):
        log_dir = f'--logdir={settings.get_summaries_folder()}'
        return ['pipenv', 'run', 'tensorboard', log_dir]


class PerformTraining(Command):
    """Executes the training wrapper script with arguments loaded from a configuration file."""

    def __init__(self, args):
        super().__init__(args)

        self.show_command = False

    def execute(self):

        command = self.create_command()
        command_util.save_to_history(command)
        command_util.execute_command(command, show_command=self.show_command, dry_run=self.dry_run)

    def create_command(self):

        config_path = Path(self.args.configuration_file)
        config = config_util.load_grim_configuration_file(config_path)

        training_arguments = TrainingWrapperArguments(config)
        training_arguments.apply_argument_overrides(self.args)
        training_arguments.set_additional_arguments(self.args.additional_args)

        return training_arguments.get_arguments()


class TrainingWrapperArguments:
    """Faciliates converting grimagents configuration values into a list of
    training_wrapper command line arguments.
    """

    def __init__(self, arguments: dict):
        self.arguments = arguments.copy()

    def apply_argument_overrides(self, args):
        """Replaces values in the arguments dictionary with the overrides stored in args."""

        if args.trainer_config is not None:
            self.set_trainer_config(args.trainer_config)

        if args.env is not None:
            self.set_env(args.env)

        if args.resume:
            self.set_resume(args.resume)

        if args.run_id is not None:
            self.set_run_id(args.run_id)

        if args.base_port is not None:
            self.set_base_port(args.base_port)

        if args.num_envs is not None:
            self.set_num_envs(str(args.num_envs))

        if args.graphics:
            # As the argument is 'no-graphics', false in this case means
            # graphics are used.
            self.set_no_graphics_enabled(False)
        if args.no_graphics:
            self.set_no_graphics_enabled(True)

        if args.timestamp:
            self.set_timestamp_enabled(True)
        if args.no_timestamp:
            self.set_timestamp_enabled(False)

        if args.multi_gpu:
            self.set_multi_gpu_enabled(True)
        if args.no_multi_gpu:
            self.set_multi_gpu_enabled(False)

        if args.inference:
            self.set_inference(args.inference)

    def set_additional_arguments(self, args):
        self.arguments[const.GA_ADDITIONAL_ARGS] = args

    def get_arguments(self):
        """Converts a configuration dictionary into command line arguments
        for mlagents-learn and filters out values that should not be sent to
        the training process.
        """

        # We copy arguments in order to mutate them in the event '--time-stamp' is present.
        command_arguments = self.arguments.copy()

        self.process_timestamp_argument(command_arguments)
        self.process_inference_argument(command_arguments)

        result = []
        for key, value in command_arguments.items():
            # mlagents-learn requires trainer config path be the first argument.
            if key == const.ML_TRAINER_CONFIG_PATH and value:
                result.insert(0, value)
                continue

            # The --resume argument does not accept a value.
            if key == const.ML_RESUME:
                if value is True:
                    result += [key]
                continue

            # The --force argument does not accept a value.
            if key == const.ML_FORCE:
                if value is True:
                    result += [key]
                continue

            # The --no-graphics argument does not accept a value.
            if key == const.ML_NO_GRAPHICS:
                if value is True:
                    result += [key]
                continue

            # The --multi-gpu argument does not accept a value.
            if key == const.ML_MULTI_GPU:
                if value is True:
                    result += [key]
                continue

            # The --debug argument does not accept a value.
            if key == const.ML_DEBUG:
                if value is True:
                    result += [key]
                continue

            # The inference argument does not accept a value.
            if key == const.GA_INFERENCE:
                if value is True:
                    result += [key]
                continue

            # The --cpu argument does not accept a value.
            if key == const.ML_CPU:
                if value is True:
                    result += [key]
                continue

            # The timestamp argument is not sent to training_wrapper.
            if key == const.GA_TIMESTAMP:
                continue

            # The 'search' dictionary is not sent to training_wrapper.
            if key == const.GS_SEARCH:
                continue

            # # Env-args must be added to the end of the command arguments.
            if key == const.ML_ENV_ARGS:
                continue

            # Additional arguments are serialized as a list and the key should
            # not be included.
            if key == const.GA_ADDITIONAL_ARGS:
                for argument in value:
                    result.append(argument)
                continue

            if value:
                result += [key, value]

        # Env-args need to be appended to the end of the training arguments.
        if const.ML_ENV_ARGS in command_arguments and command_arguments[const.ML_ENV_ARGS]:
            result += [const.ML_ENV_ARGS] + command_arguments[const.ML_ENV_ARGS]

        trainer_path = settings.get_training_wrapper_path()
        result = ['pipenv', 'run', 'python', str(trainer_path)] + result

        return result

    @staticmethod
    def process_timestamp_argument(command_arguments):
        """Modifies command line arguments for mlagents-learn if the '--timestamp' argument is present."""

        # No processing is necessary if timestamp hasn't been requested
        if (
            const.GA_TIMESTAMP not in command_arguments
            or command_arguments[const.GA_TIMESTAMP] is False
        ):
            return

        # We do not want to apply a timestamp to the run ID if '--resume' has been requested.
        if const.ML_RESUME in command_arguments and command_arguments[const.ML_RESUME]:
            return

        timestamp = common.get_timestamp()
        command_arguments[const.ML_RUN_ID] = f'{command_arguments[const.ML_RUN_ID]}-{timestamp}'

    @staticmethod
    def process_inference_argument(command_arguments):
        """Modifies command line arguments for mlagents-learn if the '--inference' argument is present."""

        # No processing is necessary if the inference argument is False
        if (
            const.GA_INFERENCE not in command_arguments
            or command_arguments[const.GA_INFERENCE] is False
        ):
            return

        # We do not want to apply inference if '--resume' has been requested.
        if const.ML_RESUME in command_arguments and command_arguments[const.ML_RESUME]:
            command_arguments[const.GA_INFERENCE] = False
            return

        if const.GA_ADDITIONAL_ARGS not in command_arguments:
            command_arguments[const.GA_ADDITIONAL_ARGS] = []

        # Remove the export path, if it is present.
        if const.GA_EXPORT_PATH in command_arguments:
            del command_arguments[const.GA_EXPORT_PATH]

    def get_arguments_as_string(self):
        return ' '.join([str(element) for element in self.get_arguments()])

    def set_trainer_config(self, value):
        self.arguments[const.ML_TRAINER_CONFIG_PATH] = value

    def set_env(self, value):
        self.arguments[const.ML_ENV] = value

    def set_resume(self, value):
        self.arguments[const.ML_RESUME] = value

    def set_run_id(self, value):
        self.arguments[const.ML_RUN_ID] = value

    def get_run_id(self):
        return self.arguments[const.ML_RUN_ID]

    def set_num_envs(self, value):
        self.arguments[const.ML_NUM_ENVS] = value

    def set_inference(self, value):
        self.arguments[const.GA_INFERENCE] = value

    def set_no_graphics_enabled(self, value):
        self.arguments[const.ML_NO_GRAPHICS] = value

    def set_timestamp_enabled(self, value):
        self.arguments[const.GA_TIMESTAMP] = value

    def set_base_port(self, value):
        self.arguments[const.ML_BASE_PORT] = value

    def set_multi_gpu_enabled(self, value):
        self.arguments[const.ML_MULTI_GPU] = value

    def set_env_args(self, value: list):
        self.arguments[const.ML_ENV_ARGS] = value
