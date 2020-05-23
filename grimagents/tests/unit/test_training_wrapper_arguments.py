import pytest

from argparse import Namespace

import grimagents.common

from grimagents.training_commands import TrainingWrapperArguments


@pytest.fixture
def grim_config():
    return {
        'trainer-config-path': 'config/3DBall.yaml',
        '--env': 'builds/3DBall/3DBall.exe',
        '--sampler': '',
        '--export-path': 'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--curriculum': '',
        '--keep-checkpoints': '',
        '--lesson': '',
        '--run-id': '3DBall',
        '--save-freq': '',
        '--seed': '',
        '--base-port': '',
        '--num-envs': '',
        '--no-graphics': False,
        '--timestamp': False,
        '--multi-gpu': False,
        '--env-args': '',
        '--cpu': '',
        '--width': '',
        '--height': '',
        '--time-scale': '',
        '--quality-level': '',
        '--target-frame-rate': '',
    }


def test_create_training_arguments(grim_config):
    """Test for creating TrainingWrapperArguments with default configuration."""

    arguments = TrainingWrapperArguments(grim_config)

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents/training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--env',
        'builds/3DBall/3DBall.exe',
        '--export-path',
        'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--run-id',
        '3DBall',
        '--train',
    ]


def test_training_arguments_handles_no_graphics(grim_config):
    """Test for correct handling of the '--no-graphics' argument."""

    # '--no-graphics' should be present
    grim_config['--no-graphics'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--no-graphics' in arguments.get_arguments()

    # '--no-graphics' should not be present
    grim_config['--no-graphics'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--no-graphics' not in arguments.get_arguments()


def test_training_arguments_handles_multi_gpu(grim_config):
    """Test for correct handling of the '--multi-gpu' argument."""

    # '--multi-gpu' should be present
    grim_config['--multi-gpu'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--multi-gpu' in arguments.get_arguments()

    # '--multi-gpu' should not be present
    grim_config['--multi-gpu'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--multi-gpu' not in arguments.get_arguments()


def test_training_arguments_handles_cpu(grim_config):
    """Test for correct handling of the '--cpu' argument."""

    # '--cpu' should be present
    grim_config['--cpu'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--cpu' in arguments.get_arguments()

    # '--cpu' should not be present
    grim_config['--cpu'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--cpu' not in arguments.get_arguments()


def test_training_arguments_excludes_timestamp(grim_config):
    """Test for ensuring TrainingWrapperArguments excludes the '--timestamp' argument."""

    # '--timestamp' should be present
    grim_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--timestamp' not in arguments.get_arguments()

    # '--timestamp' should not be present
    grim_config['--timestamp'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--timestamp' not in arguments.get_arguments()


def test_training_arguments_add_additional_args(grim_config):
    """Test for TrainingWrapperArguments correctly setting additional arguments."""

    arguments = TrainingWrapperArguments(grim_config)
    additional_args = ['--load']
    arguments.set_additional_arguments(additional_args)

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents/training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--env',
        'builds/3DBall/3DBall.exe',
        '--export-path',
        'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--run-id',
        '3DBall',
        '--load',
        '--train',
    ]


def test_override_configuration_values(grim_config):
    """Test for correct handling of TrainingWrapperArguments argument overrides.

    Ensures the following arguments are overridden:
        --trainer-config
        --env
        --sampler
        --lesson
        --run-id
        --num-envs
        --no-graphics
        --timestamp
        --multi-gpu
        '--cpu'
        '--width'
        '--height'
        '--time-scale'
        '--quality-leve'
        '--target-frame-rate'
    """

    grim_config['--sampler'] = 'config/3DBall_generalize.yaml'
    grim_config['--timestamp'] = True
    grim_config['--multi-gpu'] = True
    grim_config['--cpu'] = True
    grim_config['--width'] = 320
    grim_config['--height'] = 240
    grim_config['--time-scale'] = 20
    grim_config['--quality-leve'] = 1
    grim_config['--target-frame-rate'] = 60

    override_args = Namespace(
        configuration_file='config/3DBall_grimagents.json',
        trainer_config='config/PushBlock_grimagents.json',
        env='builds/PushBlock/PushBlock.exe',
        sampler='config/PushBlock_generalize.yaml',
        lesson=2,
        run_id='PushBlock',
        base_port=5010,
        num_envs=2,
        inference=False,
        graphics=None,
        no_graphics=True,
        timestamp=None,
        no_timestamp=True,
        multi_gpu=None,
        no_multi_gpu=True,
        args=['--load'],
    )

    arguments = TrainingWrapperArguments(grim_config)
    arguments.set_additional_arguments(override_args.args)
    arguments.apply_argument_overrides(override_args)

    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents/training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/PushBlock_grimagents.json',
        '--env',
        'builds/PushBlock/PushBlock.exe',
        '--sampler',
        'config/PushBlock_generalize.yaml',
        '--export-path',
        'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--lesson',
        '2',
        '--run-id',
        'PushBlock',
        '--base-port',
        5010,
        '--num-envs',
        '2',
        '--no-graphics',
        '--cpu',
        '--width',
        320,
        '--height',
        240,
        '--time-scale',
        20,
        '--target-frame-rate',
        60,
        '--quality-leve',
        1,
        '--load',
        '--train',
    ]


def test_training_arguments_set_methods(grim_config):
    """Tests that TrainingWrapperArguments correctly sets argument values."""

    arguments = TrainingWrapperArguments(grim_config)

    arguments.set_env('builds/3DBall/3DBallHard.exe')
    arguments.set_sampler('config/3DBall_generalize.yaml')
    arguments.set_lesson('3')
    arguments.set_run_id('ball')
    arguments.set_num_envs('4')
    arguments.set_no_graphics_enabled(True)
    arguments.set_timestamp_enabled(True)
    arguments.set_multi_gpu_enabled(True)
    arguments.set_env_args(["--num-orcs", 42])

    arguments_string = arguments.get_arguments_as_string()
    assert '--env builds/3DBall/3DBallHard.exe' in arguments_string
    assert '--sampler config/3DBall_generalize.yaml' in arguments_string
    assert '--lesson 3' in arguments_string
    assert '--run-id ball' in arguments_string
    assert '--num-envs 4' in arguments_string
    assert '--no-graphics' in arguments_string
    assert '--run-id ball-' in arguments_string
    assert '--multi-gpu' in arguments_string
    assert '--num-orcs 42' in arguments_string


def test_training_arguments_timestamp(monkeypatch, grim_config):
    """Test for TrainingWrapperArguments correctly applying a timestamp."""

    def mock_return():
        return '2019-06-29_17-13-41'

    monkeypatch.setattr(grimagents.common, "get_timestamp", mock_return)

    grim_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(grim_config)
    arguments_string = arguments.get_arguments_as_string()

    assert '--run-id 3DBall-2019-06-29_17-13-41' in arguments_string


def test_training_arguments_inference(grim_config):
    """Tests for correct processing of the '--inference' argument.

    - Ensures get_arguments() can handle no additional args being set
    - Ensures the '--train' argument is removed
    - Ensures the '--export-path' argument is removed
    """

    grim_config['--inference'] = True

    # --train is removed and no exceptions are caused by additional args not being set
    arguments = TrainingWrapperArguments(grim_config)
    arguments_list = [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--env',
        'builds/3DBall/3DBall.exe',
        '--run-id',
        '3DBall',
    ]

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents/training_wrapper.py'

    assert result == arguments_list


def test_env_args_placed_last():
    """Tests that '--env-args' are placed at the end of the training command arguments."""

    config = {
        'trainer-config-path': 'config/3DBall.yaml',
        '--env-args': ["--num-orcs", 42],
        '--env': 'builds/3DBall/3DBall.exe',
        '--export-path': 'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--run-id': '3DBall',
    }

    arguments = TrainingWrapperArguments(config)
    result = arguments.get_arguments_as_string()

    assert result.endswith('--env-args --num-orcs 42')
