from . import common as common
from . import config as config_util
from . import settings as settings


ADDITIONAL_ARGS = 'additional-args'
SLOW = '--slow'


class Command():
    def __init__(self):
        self._arguments = {}

    def get_command(self):
        return ['echo', __class__.__name__, self.arguments]


class TrainingCommand(Command):
    """Training Wrapper command"""

    def __init__(self, arguments: dict):
        self.arguments = arguments.copy()

    def set_additional_arguments(self, args):
        self.arguments[ADDITIONAL_ARGS] = args

    def get_command(self):
        """Converts a configuration dictionary into command line arguments
        for mlagents-learn and filters out values that should not be sent to
        the training process.
        """

        # We copy arguments in order to mutate it in the event a time-stamp is present.
        command_arguments = self.arguments.copy()

        # Process --timestamp argument
        if config_util.TIMESTAMP in command_arguments and command_arguments[config_util.TIMESTAMP]:
            if config_util.LOG_FILE_NAME not in command_arguments or not command_arguments[config_util.LOG_FILE_NAME]:
                # Explicitly set a log-filename if it doesn't exist to prevent a million log files being generated.
                command_arguments[config_util.LOG_FILE_NAME] = command_arguments[config_util.RUN_ID]

            timestamp = common.get_timestamp()
            command_arguments[config_util.RUN_ID] = f'{command_arguments[config_util.RUN_ID]}-{timestamp}'

        # Process --inference argument
        use_inference = config_util.INFERENCE in command_arguments and command_arguments[config_util.INFERENCE]
        if use_inference:
            if ADDITIONAL_ARGS not in command_arguments:
                command_arguments[ADDITIONAL_ARGS] = []

            # Add the --slow flag if inference was requested, but it isn't present.
            if SLOW not in command_arguments[ADDITIONAL_ARGS]:
                command_arguments[ADDITIONAL_ARGS].append(SLOW)
            if config_util.EXPORT_PATH in command_arguments:
                del(command_arguments[config_util.EXPORT_PATH])

        result = list()
        for key, value in command_arguments.items():
            # mlagents-learn requires trainer config path be the first argument.
            if key == config_util.TRAINER_CONFIG_PATH and value:
                result.insert(0, value)
                continue

            # The --no-graphics argument does not accept a value.
            if key == config_util.NO_GRAPHICS:
                if value is True:
                    result = result + [key]
                continue

            # The --timestamp argument is not sent to training_wrapper.
            if key == config_util.TIMESTAMP:
                continue

            # The --inference argument is not sent to training_wrapper.
            if key == config_util.INFERENCE:
                continue

            # Additional arguments are serialized as a list and the key should
            # not be included.
            if key == ADDITIONAL_ARGS:
                for argument in value:
                    result.append(argument)
                continue

            if value:
                result = result + [key, value]

        trainer_path = settings.get_training_wrapper_path()
        result = ['pipenv', 'run', 'python', str(trainer_path)] + result

        # Exclude '--train' argument if inference was requested.
        if not use_inference:
            result = result + ['--train']

        return result

    def get_command_as_string(self):
        return ' '.join(self.get_command())

    def set_trainer_config(self, value):
        self.arguments[config_util.TRAINER_CONFIG_PATH] = value

    def set_env(self, value):
        self.arguments[config_util.ENV] = value

    def set_lesson(self, value):
        self.arguments[config_util.LESSON] = value

    def set_run_id(self, value):
        self.arguments[config_util.RUN_ID] = value

    def get_run_id(self):
        return self.arguments[config_util.RUN_ID]

    def set_num_envs(self, value):
        self.arguments[config_util.NUM_ENVS] = value

    def set_inference(self, value):
        self.arguments[config_util.INFERENCE] = value;

    def set_no_graphics_enabled(self, value):
        self.arguments[config_util.NO_GRAPHICS] = value

    def set_timestamp_enabled(self, value):
        self.arguments[config_util.TIMESTAMP] = value

    def set_log_filename(self, value):
        self.arguments[config_util.LOG_FILE_NAME] = value

    def set_base_port(self, value):
        self.arguments[config_util.BASE_PORT] = value


class MLAgentsLearnCommand(Command):
    pass
