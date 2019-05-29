"""Loads configuration files and fetches loaded configuration values."""

import json

from pathlib import Path
from .command_util import open_file


# Default configuration values
_DEFAULT_CONFIG = {
      'env': '',
      'curriculum': '',
      'keep-checkpoints': '',
      'lesson': '',
      'run-id': '',
      'num-runs': '',
      'save-freq': '',
      'seed': '',
      'base-port': '',
      'num-envs': '',
      'no-graphics': '',
}


# Options that will not add to config,
# but will be supported on command line
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# load
# slow
# train
# debug


_loaded_config = None


class ConfigurationError(Exception):
    """Base error for configuration module exceptions."""


class InvalidConfigurationError(ConfigurationError):
    """An error occurred while loading a configuration file."""


class EmptyConfigurationError(ConfigurationError):
    """An error occurred because configuration values were accessed before a
    configuration was loaded."""


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

    print(f'Creating configuration file \'{config_path}\'')
    with config_path.open(mode='w') as f:
        json.dump(get_default_config(), f, indent=4)


def get_default_config():
    """Fetches a copy of the default configuration dictionary."""

    return _DEFAULT_CONFIG.copy()


def load_config_file(config_path: Path):
    """Loads a configuration file into the loaded configuration global dictionary.

    Args:
      config_path: Path: Path object for the configuration file to load into memory.

    Raises:
      FileNotFoundError: An error occurred while attempting to load a configuration file.
      InvalidConfigurationError: The specified configuration file is not valid.
    """

    global _loaded_config

    try:
        with config_path.open('r') as f:
            configuration = json.load(f)
    except FileNotFoundError as exception:
        print(f'Configuration file \'{config_path}\' not found')
        raise exception

    if validate_configuration(configuration):
        _loaded_config = configuration
    else:
        raise InvalidConfigurationError(f'Configuration file \'{config_path}\' is invalid')


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
            print(f'Configuration is missing key \'{key}\'')
            is_valid_config = False

    return is_valid_config


def get_config():
    """Fetches the configuration dictionary loaded into memory.

    Returns:
      The dictionary storing a mapping configuration keys and values.

    Raises:
      EmptyConfigurationError: A configuration value is accessed without a configuration
        file being loaded first.
    """

    global _loaded_config
    if _loaded_config is None:
        print(f'Unable to retrieve configuration values: A configuration file has not been loaded')
        raise EmptyConfigurationError

    return _loaded_config
