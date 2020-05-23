"""Creates new configuration files, loads configuration files for editing, and validates loaded configurations.

Notes:
- All path values should be a relative path from the MLAgents project root folder
- The `--export-path` configuration value is consumed by training_wrapper.py
"""

import logging
from pathlib import Path

import yaml

import grimagents.command_util as command_util
import grimagents.constants as const


_DEFAULT_GRIM_CONFIG = {
    const.ML_TRAINER_CONFIG_PATH: '',
    const.ML_ENV: '',
    const.GA_EXPORT_PATH: '',
    const.ML_CURRICULUM: '',
    const.ML_SAMPLER: '',
    const.ML_KEEP_CHECKPOINTS: '',
    const.ML_LESSON: '',
    const.ML_RUN_ID: 'TRAINING_RUN',
    const.ML_SAVE_FREQUENCY: '',
    const.ML_SEED: '',
    const.ML_BASE_PORT: '',
    const.ML_NUM_ENVS: '',
    const.ML_DOCKER_TARGET: '',
    const.ML_NO_GRAPHICS: False,
    const.ML_DEBUG: False,
    const.ML_MULTI_GPU: False,
    const.GA_INFERENCE: False,
    const.GA_TIMESTAMP: False,
    const.ML_ENV_ARGS: [],
    const.ML_CPU: False,
    const.ML_WIDTH: 320,
    const.ML_HEIGHT: 240,
    const.ML_TIMESCALE: "",
    const.ML_QUALITY_LEVEL: "",
    const.ML_TARGET_FRAME_RATE: "",
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
    const.CU_MEASURE: const.CU_PROGRESS,
    const.CU_THRESHOLDS: [0.1],
    const.CU_MIN_LESSON_LENGTH: 100,
    const.CU_SIGNAL_SMOOTHING: True,
    const.CU_PARAMETERS: {'example_reset_parameter': [1.5, 2.0]},
}


_DEFAULT_SEARCH_CONFIG = {
    const.GS_BRAIN: {
        const.GS_NAME: 'BRAIN_NAME',
        const.GS_HYPERPARAMS: {
            const.HP_BATCH_SIZE: [512, 5120],
            const.HP_BETA: [1e-4, 1e-2],
            const.GS_BUFFER_SIZE_MULTIPLE: [4, 10],
            const.HP_EPSILON: [0.1, 0.3],
            const.HP_HIDDEN_UNITS: [32, 512],
            const.HP_LAMBD: [0.9, 0.95],
            const.HP_LEARNING_RATE: [1e-5, 1e-3],
            const.HP_MAX_STEPS: [5e5, 1e7],
            const.HP_MEMORY_SIZE: [64, 512],
            const.HP_NUM_LAYERS: [1, 3],
            const.HP_NUM_EPOCH: [3, 10],
            const.HP_TIME_HORIZON: [32, 2048],
            const.HP_SEQUENCE_LENGTH: [4, 128],
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
        if const.GS_SEARCH not in config:
            config[const.GS_SEARCH] = get_default_search_config()
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
        if key == const.GS_SEARCH:
            continue

        try:
            default_config[key]
        except KeyError:
            config_log.error(f'Configuration contains invalid key \'{key}\'')
            is_valid_config = False

    # The only required keys are 'trainer-config-path' and '--run-id'
    for key in {const.ML_TRAINER_CONFIG_PATH, const.ML_RUN_ID}:
        try:
            if not configuration[key]:
                raise KeyError

        except KeyError:
            config_log.error(f'Configuration is missing required key \'{key}\'')
            is_valid_config = False

    # If --env-args entry is present, ensure it is a list.
    if (
        const.ML_ENV_ARGS in configuration
        and configuration[const.ML_ENV_ARGS]
        and not isinstance(configuration[const.ML_ENV_ARGS], list)
    ):
        config_log.error('Configuration value for \'--env-args\' must be a list.')
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
