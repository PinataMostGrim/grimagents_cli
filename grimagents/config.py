"""Creates new configuration files, loads configuration files for editing, and validates loaded configurations.

Notes:
- All path values should be a relative path from the MLAgents project root folder
- The `--export-path` configuration value is consumed by training_wrapper.py
"""

import logging
from pathlib import Path

import yaml

import grimagents.command_util as command_util


# Configuration keys
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
SEARCH = 'search'


_DEFAULT_GRIM_CONFIG = {
    TRAINER_CONFIG_PATH: '',
    ENV: '',
    EXPORT_PATH: '',
    '--curriculum': '',
    '--keep-checkpoints': '',
    LESSON: '',
    RUN_ID: 'ppo',
    '--num-runs': '',
    '--save-freq': '',
    '--seed': '',
    BASE_PORT: '',
    NUM_ENVS: '',
    INFERENCE: False,
    NO_GRAPHICS: False,
    TIMESTAMP: False,
}

_DEFAULT_TRAINER_CONFIG = """default:
    trainer: ppo
    batch_size: 1024
    beta: 5.0e-3
    buffer_size: 10240
    epsilon: 0.2
    hidden_units: 128
    lambd: 0.95
    learning_rate: 3.0e-4
    learning_rate_schedule: linear
    max_steps: 5.0e4
    memory_size: 256
    normalize: false
    num_epoch: 3
    num_layers: 2
    time_horizon: 64
    sequence_length: 64
    summary_freq: 1000
    use_recurrent: false
    vis_encode_type: simple
    reward_signals:
        extrinsic:
            strength: 1.0
            gamma: 0.99
"""

_DEFAULT_CURRICULUM = {
    'measure': 'progress',
    'thresholds': [0.1],
    'min_lesson_length': 100,
    'signal_smoothing': True,
    'parameters': {'example_reset_parameter': [1.5, 2.0]},
}


_DEFAULT_SEARCH_CONFIG = {
    'brain': {
        'name': 'BRAIN_NAME',
        'hyperparameters': {
            'batch_size': [512, 5120],
            'beta': [1e-4, 1e-2],
            'buffer_size_multiple': [4, 10],
            'epsilon': [0.1, 0.3],
            'hidden_units': [32, 512],
            'lambd': [0.9, 0.95],
            'learning_rate': [1e-5, 1e-3],
            'max_steps': [5e5, 1e7],
            'memory_size': [64, 512],
            'num_layers': [1, 3],
            'num_epoch': [3, 10],
            'time_horizon': [32, 2048],
            'sequence_length': [4, 128],
            'reward_signals.extrinsic.gamma': [0.98, 0.99],
        },
    }
}


config_log = logging.getLogger('grimagents.config')


class ConfigurationError(Exception):
    """Base error for configuration module exceptions."""


class InvalidConfigurationError(ConfigurationError):
    """An error occurred while loading a configuration file."""


def edit_grim_config_file(file_path: Path, add_search=False):
    """Opens a grimagents configuration file with the system's default editor.
    Creates a configuration file with default values if file does not already exist. If 'add_search' is True, will add default search configuration values.
    """

    if not file_path.suffix == '.json':
        file_path = file_path.with_suffix('.json')

    if not file_path.exists():
        create_grim_config_file(file_path)

    if add_search:
        config = command_util.load_json_file(file_path)
        if SEARCH not in config:
            config[SEARCH] = get_default_search_config()
            command_util.write_json_file(config, file_path)

    command_util.open_file(file_path)


def create_grim_config_file(file_path: Path):
    """Creates a configuration file with default values at the specified path."""

    command_util.write_json_file(get_default_grim_config(), file_path)


def get_default_grim_config():
    """Fetches a copy of the default configuration dictionary."""

    return _DEFAULT_GRIM_CONFIG.copy()


def get_default_trainer_config():
    """Fetches the default trainer configuration."""

    return yaml.safe_load(_DEFAULT_TRAINER_CONFIG)


def get_default_curriculum():
    """Fetches the default curriculum."""

    return _DEFAULT_CURRICULUM.copy()


def get_default_search_config():
    """Fetches a copy of the default search configuration dictionary."""

    return _DEFAULT_SEARCH_CONFIG.copy()


def load_grim_configuration_file(file_path: Path):
    """Loads a grimagents configuration dictionary from file.

    Raises:
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
        # 'search' is not defined in the default configuration but is still a valid key.
        if key == SEARCH:
            continue

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


def load_trainer_configuration_file(file_path: Path):
    """Loads a MLAgents trainer configuration from a yaml file.
    """

    configuration = command_util.load_yaml_file(file_path)
    return configuration


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

    command_util.write_yaml_file(get_default_trainer_config(), file_path)


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

    command_util.write_json_file(get_default_curriculum(), file_path)
