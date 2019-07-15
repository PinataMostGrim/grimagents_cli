from . import common as common
from . import settings as settings


TRAINER_CONFIG_PATH = 'trainer-config-path'
ENV = '--env'
LESSON = '--lesson'
RUN_ID = '--run-id'
EXPORT_PATH = '--export-path'
BASE_PORT = '--base-port'
NUM_ENVS = '--num-envs'
INFERENCE = '--inference'
NO_GRAPHICS = '--no-graphics'
TIMESTAMP = '--timestamp'
LOG_FILE_NAME = '--log-filename'
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
        if TIMESTAMP in command_arguments and command_arguments[TIMESTAMP]:
            if LOG_FILE_NAME not in command_arguments or not command_arguments[LOG_FILE_NAME]:
                # Explicitly set a log-filename if it doesn't exist to prevent a million log files being generated.
                command_arguments[LOG_FILE_NAME] = command_arguments[RUN_ID]

            timestamp = common.get_timestamp()
            command_arguments[RUN_ID] = f'{command_arguments[RUN_ID]}-{timestamp}'

        # Process --inference argument
        use_inference = INFERENCE in command_arguments and command_arguments[INFERENCE]
        if use_inference:
            if ADDITIONAL_ARGS not in command_arguments:
                command_arguments[ADDITIONAL_ARGS] = []

            # Add the --slow flag if inference was requested, but it isn't present.
            if SLOW not in command_arguments[ADDITIONAL_ARGS]:
                command_arguments[ADDITIONAL_ARGS].append(SLOW)
            if EXPORT_PATH in command_arguments:
                del(command_arguments[EXPORT_PATH])

        result = list()
        for key, value in command_arguments.items():
            # mlagents-learn requires trainer config path be the first argument.
            if key == TRAINER_CONFIG_PATH and value:
                result.insert(0, value)
                continue

            # The --no-graphics argument does not accept a value.
            if key == NO_GRAPHICS:
                if value is True:
                    result = result + [key]
                continue

            # The --timestamp argument is not sent to training_wrapper.
            if key == TIMESTAMP:
                continue

            # The --inference argument is not sent to training_wrapper.
            if key == INFERENCE:
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
        self.arguments[TRAINER_CONFIG_PATH] = value

    def set_env(self, value):
        self.arguments[ENV] = value

    def set_lesson(self, value):
        self.arguments[LESSON] = value

    def set_run_id(self, value):
        self.arguments[RUN_ID] = value

    def get_run_id(self):
        return self.arguments[RUN_ID]

    def set_num_envs(self, value):
        self.arguments[NUM_ENVS] = value

    def set_inference(self, value):
        self.arguments[INFERENCE] = value;

    def set_no_graphics_enabled(self, value):
        self.arguments[NO_GRAPHICS] = value

    def set_timestamp_enabled(self, value):
        self.arguments[TIMESTAMP] = value

    def set_log_filename(self, value):
        self.arguments[LOG_FILE_NAME] = value

    def set_base_port(self, value):
        self.arguments[BASE_PORT] = value


class MLAgentsLearnCommand(Command):
    pass
