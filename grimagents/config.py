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
TRAINER_CONFIG_PATH = 'trainer-config-path'
RUN_ID = '--run-id'

_DEFAULT_GRIM_CONFIG = {
    TRAINER_CONFIG_PATH: '',
    '--env': '',
    '--export-path': '',
    '--curriculum': '',
    '--keep-checkpoints': '',
    '--lesson': '',
    RUN_ID: 'ppo',
    '--num-runs': '',
    '--save-freq': '',
    '--seed': '',
    '--base-port': '',
    '--num-envs': '',
    '--inference': False,
    '--no-graphics': False,
    '--timestamp': False,
    '--log-filename': None,
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
    "parameters": {"example_reset_parameter": [1.5, 2.0]},
}


config_log = logging.getLogger('grimagents.config')


class ConfigurationError(Exception):
    """Base error for configuration module exceptions."""


class InvalidConfigurationError(ConfigurationError):
    """An error occurred while loading a configuration file."""


def edit_grim_config_file(file_path: Path):
    """Opens a grimagents configuration file with the system's default editor.
    Creates a configuration file with default values if file does not already exist.
    """

    if not file_path.suffix == '.json':
        file_path = file_path.with_suffix('.json')

    if not file_path.exists():
        create_grim_config_file(file_path)

    command_util.open_file(file_path)


def create_grim_config_file(file_path: Path):
    """Creates a configuration file with default values at the specified path."""

    command_util.write_json_file(get_default_grim_config(), file_path)


def get_default_grim_config():
    """Fetches a copy of the default configuration dictionary."""

    return _DEFAULT_GRIM_CONFIG.copy()


def load_grim_config_file(file_path: Path):
    """Loads a grimagents configuration dictionary from file.

    Raises:
      FileNotFoundError: An error occurred while attempting to load a configuration file.
      InvalidConfigurationError: The specified configuration file is not valid.
    """

    configuration = command_util.load_json_file(file_path)

    if not validate_grim_configuration(configuration):
        config_log.error(f'Configuration file \'{file_path}\' is invalid')
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
    for key in {TRAINER_CONFIG_PATH, RUN_ID}:
        try:
            if not configuration[key]:
                raise KeyError

        except KeyError:
            config_log.error(f'Configuration is missing required key \'{key}\'')
            is_valid_config = False

    return is_valid_config


def edit_trainer_configuration_file(file_path: Path):
    """Opens a trainer configuration file for editing. Creates a configuration
    file with default values if file does not already exit.
    """

    if not file_path.suffix == '.yaml':
        file_path = file_path.with_suffix('.yaml')

    if not file_path.exists():
        create_trainer_configuration_file(file_path)

    command_util.open_file(file_path)


def create_trainer_configuration_file(file_path: Path):
    """Creates a trainer configuration file with default values at the specified path."""

    command_util.write_file(_DEFAULT_TRAINER_CONFIG, file_path)


def edit_curriculum_file(file_path: Path):
    """Opens a curriculum file for editing. Creates a curriculum file with
    default values if file does not already exit.
    """

    if not file_path.suffix == '.json':
        file_path = file_path.with_suffix('.json')

    if not file_path.exists():
        create_curriculum_file(file_path)

    command_util.open_file(file_path)


def create_curriculum_file(file_path: Path):
    """Creates a curriculum file with default values at the specified path."""

    command_util.write_json_file(_DEFAULT_CURRICULUM, file_path)
