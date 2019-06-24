"""Loads configuration files and fetches loaded configuration values.

# Relative path from MLAgents' project root folder to the summaries folder.

Notes:
- All path configuration values should be a relative path from MLAgents' project root folder to the target asset or folder
- The `--timestamp` and `--export-path` options apply to training_wrapper.
"""

import json
import logging

from pathlib import Path

from .command_util import open_file


# Default configuration values
_TRAINER_CONFIG_PATH_KEY = 'trainer-config-path'
_ENV_KEY = '--env'
_LESSON_KEY = '--lesson'
_RUN_ID_KEY = '--run-id'
_NUM_ENVS_KEY = '--num-envs'
_NO_GRAPHICS_KEY = '--no-graphics'
_TIMESTAMP_KEY = '--timestamp'

_DEFAULT_CONFIG = {
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
}


config_log = logging.getLogger('grimagents.config')


class ConfigurationError(Exception):
    """Base error for configuration module exceptions."""


class InvalidConfigurationError(ConfigurationError):
    """An error occurred while loading a configuration file."""


def edit_config_file(config_path: Path):
    """Opens the specified configuration file with the system's default editor.

    Args:
      config_path: Path: Path object for the configuration file to edit.
    """

    if not config_path.suffix == '.json':
        config_path = config_path.with_suffix('.json')

    if not config_path.exists():
        create_config_file(config_path)

    open_file(config_path)


def create_config_file(config_path: Path):
    """Creates a configuration file with default values at the specified path.

    Args:
      config_path: Path: Path object for the configuration file to create.
    """

    # Note: If directory doesn't exist, create it.
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True)

    config_log.info(f'Creating configuration file \'{config_path}\'')
    with config_path.open(mode='w') as f:
        json.dump(get_default_config(), f, indent=4)


def get_default_config():
    """Fetches a copy of the default configuration dictionary."""

    return _DEFAULT_CONFIG.copy()


def load_config_file(config_path: Path):
    """Loads a configuration file into the loaded configuration global dictionary.

    Args:
      config_path: Path: Path object for the configuration file to load into memory.

    Returns:
      Configuration dictionary loaded from file.

    Raises:
      FileNotFoundError: An error occurred while attempting to load a configuration file.
      InvalidConfigurationError: The specified configuration file is not valid.
    """

    try:
        with config_path.open('r') as f:
            configuration = json.load(f)
    except FileNotFoundError as exception:
        config_log.error(f'Configuration file \'{config_path}\' not found')
        raise exception

    if validate_configuration(configuration):
        loaded_config = configuration
    else:
        config_log.error(f'Configuration file \'{config_path}\' is invalid')
        raise InvalidConfigurationError

    return loaded_config


def validate_configuration(configuration):
    """Checks the specified configuration dictionary for all required keys and conditions.

    Args:
      configuration: The configuration dictionary to validate.

    Returns:
      True if the configuration is valid and False if it is not.
    """

    default_config = get_default_config()
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

    # The only currently required key is 'trainer-config-path.'
    try:
        if not configuration[_TRAINER_CONFIG_PATH_KEY]:
            raise KeyError

    except KeyError:
        config_log.error(f'Configuration is missing required key \'{_TRAINER_CONFIG_PATH_KEY}\'')
        is_valid_config = False

    return is_valid_config


def get_training_arguments(configuration):
    """Converts a configuration dictionary into command line arguments
    for mlagents-learn and filters out values that should not be sent to
    the training process.

    Args:
      configuration: A configuration dictionary

    Returns:
      A list of command line arguments.
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
