"""Loads configuration files, fetches loaded configuration values and parses
training arguments from configuration values.

Notes:
- All path configuration values should be a relative path from MLAgents' project root folder to the target asset or folder
- `--export-path` and `--logname` configuration values apply to training_wrapper
"""

import logging

from pathlib import Path

from . import command_util as command_util


# Default configuration values
_TRAINER_CONFIG_PATH_KEY = 'trainer-config-path'
_ENV_KEY = '--env'
_LESSON_KEY = '--lesson'
_RUN_ID_KEY = '--run-id'
_NUM_ENVS_KEY = '--num-envs'
_NO_GRAPHICS_KEY = '--no-graphics'
_TIMESTAMP_KEY = '--timestamp'
_LOG_FILE_NAME = '--log-filename'

_DEFAULT_GRIM_CONFIG = {
    _TRAINER_CONFIG_PATH_KEY: '',
    _ENV_KEY: '',
    '--export-path': '',
    '--curriculum': '',
    '--keep-checkpoints': '',
    _LESSON_KEY: '',
    _RUN_ID_KEY: 'ppo',
    '--num-runs': '',
    '--save-freq': '',
    '--seed': '',
    '--base-port': '',
    _NUM_ENVS_KEY: '',
    _NO_GRAPHICS_KEY: False,
    _TIMESTAMP_KEY: False,
    _LOG_FILE_NAME: None,
}

_DEFAULT_TRAINER_CONFIG = """default:
    trainer: ppo
    batch_size: 1024
    beta: 5.0e-3
    buffer_size: 10240
    epsilon: 0.2
    gamma: 0.99
    hidden_units: 128
    lambd: 0.95
    learning_rate: 3.0e-4
    max_steps: 5.0e4
    memory_size: 256
    normalize: false
    num_epoch: 3
    num_layers: 2
    time_horizon: 64
    sequence_length: 64
    summary_freq: 1000
    use_recurrent: false
    use_curiosity: false
    curiosity_strength: 0.01
    curiosity_enc_size: 128
"""

_DEFAULT_CURRICULUM = {
    "measure": "progress",
    "thresholds": [0.1],
    "min_lesson_length": 100,
    "signal_smoothing": True,
    "parameters":
    {
        "example_reset_parameter": [1.5, 2.0]
    }
}


config_log = logging.getLogger('grimagents.config')


class ConfigurationError(Exception):
    """Base error for configuration module exceptions."""


class InvalidConfigurationError(ConfigurationError):
    """An error occurred while loading a configuration file."""


def edit_grim_config_file(config_path: Path):
    """Opens a grimagents configuration file with the system's default editor.
    Creates a configuration file with default values if file does not already exist.
    """

    if not config_path.suffix == '.json':
        config_path = config_path.with_suffix('.json')

    if not config_path.exists():
        create_grim_config_file(config_path)

    command_util.open_file(config_path)


def create_grim_config_file(config_path: Path):
    """Creates a configuration file with default values at the specified path."""

    command_util.write_json_file(get_default_grim_config(), config_path)


def get_default_grim_config():
    """Fetches a copy of the default configuration dictionary."""

    return _DEFAULT_GRIM_CONFIG.copy()


def load_grim_config_file(config_path: Path):
    """Loads a grimagents configuration dictionary from file.

    Raises:
      FileNotFoundError: An error occurred while attempting to load a configuration file.
      InvalidConfigurationError: The specified configuration file is not valid.
    """

    configuration = command_util.load_json_file(config_path)

    if not validate_grim_configuration(configuration):
        config_log.error(f'Configuration file \'{config_path}\' is invalid')
        raise InvalidConfigurationError

    return configuration


def validate_grim_configuration(configuration):
    """Checks the specified configuration dictionary for all required keys and conditions.

    Returns:
      True if the configuration is valid and False if it is not.
    """

    default_config = get_default_grim_config()
    is_valid_config = True

    # Check all keys in configuration against the default configuration.
    # It is valid for the configuration to have fewer keys than the full,
    # configuration, but it should not contain any keys that do not exist
    # in the full configuration.
    for key, value in configuration.items():
        try:
            default_config[key]
        except KeyError:
            config_log.error(f'Configuration contains invalid key \'{key}\'')
            is_valid_config = False

    # The only required keys are 'trainer-config-path' and '--run-id'
    for key in {_TRAINER_CONFIG_PATH_KEY, _RUN_ID_KEY}:
        try:
            if not configuration[key]:
                raise KeyError

        except KeyError:
            config_log.error(f'Configuration is missing required key \'{key}\'')
            is_valid_config = False

    return is_valid_config


def edit_trainer_configuration_file(config_path: Path):
    """Opens a trainer configuration file for editing. Creates a configuration
    file with default values if file does not already exit.
    """

    if not config_path.suffix == '.yaml':
        config_path = config_path.with_suffix('.yaml')

    if not config_path.exists():
        command_util.write_file(_DEFAULT_TRAINER_CONFIG, config_path)

    command_util.open_file(config_path)


def edit_curriculum_file(file_path: Path):
    """Opens a curriculum file for editing. Creates a curriculum file with
    default values if file does not already exit.
    """

    if not file_path.suffix == '.json':
        file_path = file_path.with_suffix('.json')

    if not file_path.exists():
        command_util.write_json_file(_DEFAULT_CURRICULUM, file_path)

    command_util.open_file(file_path)


def get_training_arguments(configuration):
    """Converts a configuration dictionary into command line arguments
    for mlagents-learn and filters out values that should not be sent to
    the training process.
    """

    command_args = list()

    for key, value in configuration.items():
        # Note: mlagents-learn requires trainer config path be the first argument.
        if key == _TRAINER_CONFIG_PATH_KEY and value:
            command_args.insert(0, value)
            continue

        # Note: The --no-graphics argument does not accept a value.
        if key == _NO_GRAPHICS_KEY:
            if value is True:
                command_args = command_args + [key]
            continue

        # Note: The --timestamp argument does not get sent to training_wrapper.
        if key == _TIMESTAMP_KEY:
            continue

        if value:
            command_args = command_args + [key, value]

    return command_args


def set_env(value: str, configuration):
    configuration[_ENV_KEY] = value
    return configuration


def set_lesson(value: int, configuration):

    configuration[_LESSON_KEY] = value
    return configuration


def get_run_id(configuration):

    return configuration[_RUN_ID_KEY]


def set_run_id(value: str, configuration):

    configuration[_RUN_ID_KEY] = value
    return configuration


def set_num_envs(value: int, configuration):
    configuration[_NUM_ENVS_KEY] = value
    return configuration


def set_no_graphics_enabled(value: bool, configuration):

    configuration[_NO_GRAPHICS_KEY] = value
    return configuration


def get_timestamp_enabled(configuration):

    try:
        return configuration['--timestamp']
    except KeyError:
        return False


def set_timestamp_enabled(value: bool, configuration):

    configuration[_TIMESTAMP_KEY] = value
    return configuration


def get_log_filename(configuration):
    try:
        return configuration[_LOG_FILE_NAME]
    except KeyError:
        return None


def set_log_filename(value: str, configuration):
    configuration[_LOG_FILE_NAME] = value
    return configuration
