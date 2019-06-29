from . import settings as settings

TRAINER_CONFIG_PATH = 'trainer-config-path'
NO_GRAPHICS = '--no-graphics'
TIMESTAMP = '--timestamp'
ADDITIONAL_ARGS = 'additional-args'


class Command():
    def __init__(self):
        self._arguments = {}

    def get_command(self):
        return ['echo', __class__.__name__]


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

        result = list()
        for key, value in self.arguments.items():
            # Note: mlagents-learn requires trainer config path be the first argument.
            if key == TRAINER_CONFIG_PATH and value:
                result.insert(0, value)
                continue

            # Note: The --no-graphics argument does not accept a value.
            if key == NO_GRAPHICS:
                if value is True:
                    result = result + [key]
                continue

            # Note: The --timestamp argument does not get sent to training_wrapper.
            if key == TIMESTAMP:
                continue

            # Note: Additional arguments are serialized as a list and the key should
            # not be included.
            if key == ADDITIONAL_ARGS:
                for argument in value:
                    result.append(argument)
                continue

            if value:
                result = result + [key, value]

        trainer_path = settings.get_training_wrapper_path()
        result = ['pipenv', 'run', 'python', str(trainer_path)] + result + ['--train']
        return result


class MLAgentsLearnCommand(Command):
    pass
