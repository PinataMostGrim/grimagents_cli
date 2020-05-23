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


class ResumeTraining(Command):
    """Launches the training wrapper script with the arguments used
    by the last training command executed."""

    def __init__(self, args):
        super().__init__(args)

        self.show_command = False

    def create_command(self):

        command = command_util.load_last_history()

        if const.ML_LOAD not in command:
            command.append(const.ML_LOAD)
        if self.args.lesson:
            command.append(const.ML_LESSON)
            command.append(str(self.args.lesson))

        return command


class TrainingWrapperArguments:
    """Faciliates converting grimagents configuration values into a list of training_wrapper command line arguments.
    """

    def __init__(self, arguments: dict):
        self.arguments = arguments.copy()

    def apply_argument_overrides(self, args):
        """Replaces values in the arguments dictionary with the overrides stored in args."""

        if args.trainer_config is not None:
            self.set_trainer_config(args.trainer_config)

        if args.env is not None:
            self.set_env(args.env)

        if args.sampler is not None:
            self.set_sampler(args.sampler)

        if args.lesson is not None:
            self.set_lesson(str(args.lesson))

        if args.run_id is not None:
            self.set_run_id(args.run_id)

        if args.base_port is not None:
            self.set_base_port(args.base_port)

        if args.num_envs is not None:
            self.set_num_envs(str(args.num_envs))

        if args.inference is not None:
            self.set_inference(args.inference)

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

    def set_additional_arguments(self, args):
        self.arguments[const.GA_ADDITIONAL_ARGS] = args

    def get_arguments(self):
        """Converts a configuration dictionary into command line arguments
        for mlagents-learn and filters out values that should not be sent to
        the training process.
        """

        # We copy arguments in order to mutate it in the event a time-stamp is present.
        command_arguments = self.arguments.copy()

        # Process --timestamp argument.
        if const.GA_TIMESTAMP in command_arguments and command_arguments[const.GA_TIMESTAMP]:

            timestamp = common.get_timestamp()
            command_arguments[const.ML_RUN_ID] = f'{command_arguments[const.ML_RUN_ID]}-{timestamp}'

        # Process --inference argument.
        use_inference = (
            const.GA_INFERENCE in command_arguments and command_arguments[const.GA_INFERENCE]
        )
        if use_inference:
            if const.GA_ADDITIONAL_ARGS not in command_arguments:
                command_arguments[const.GA_ADDITIONAL_ARGS] = []

            # Remove the export path, if it is present.
            if const.GA_EXPORT_PATH in command_arguments:
                del command_arguments[const.GA_EXPORT_PATH]

        result = []
        for key, value in command_arguments.items():
            # mlagents-learn requires trainer config path be the first argument.
            if key == const.ML_TRAINER_CONFIG_PATH and value:
                result.insert(0, value)
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

            # The --cpu argument does not accept a value.
            if key == const.ML_CPU:
                if value is True:
                    result += [key]
                continue

            # The timestamp argument is not sent to training_wrapper.
            if key == const.GA_TIMESTAMP:
                continue

            # The inference argument is not sent to training_wrapper.
            if key == const.GA_INFERENCE:
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

        # Exclude '--train' argument if inference was requested.
        if not use_inference:
            result += [const.ML_TRAIN]

        # Env-args need to be appended to the end of the training arguments.
        if const.ML_ENV_ARGS in command_arguments and command_arguments[const.ML_ENV_ARGS]:
            result += [const.ML_ENV_ARGS] + command_arguments[const.ML_ENV_ARGS]

        trainer_path = settings.get_training_wrapper_path()
        result = ['pipenv', 'run', 'python', str(trainer_path)] + result

        return result

    def get_arguments_as_string(self):
        return ' '.join([str(element) for element in self.get_arguments()])

    def set_trainer_config(self, value):
        self.arguments[const.ML_TRAINER_CONFIG_PATH] = value

    def set_env(self, value):
        self.arguments[const.ML_ENV] = value

    def set_sampler(self, value):
        self.arguments[const.ML_SAMPLER] = value

    def set_lesson(self, value):
        self.arguments[const.ML_LESSON] = value

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
