import pytest

from argparse import Namespace

import grimagents.common

from grimagents.training_commands import TrainingWrapperArguments


@pytest.fixture
def grim_config():
    return {
        'trainer-config-path': 'config/3DBall.yaml',
        '--run-id': '3DBall',
        '--env': '',
        '--sampler': '',
        '--export-path': '',
        '--curriculum': '',
        '--keep-checkpoints': '',
        '--force': '',
        '--lesson': '',
        "--initialize-from": "",
        '--save-freq': '',
        '--seed': '',
        '--base-port': '',
        '--num-envs': '',
        '--no-graphics': False,
        '--inference': False,
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
    """Test for creating TrainingWrapperArguments with mandatory configuration values."""

    arguments = TrainingWrapperArguments(grim_config)

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result = arguments.get_arguments()
    result[3] = 'grimagents/training_wrapper.py'

    # 'trainer-config-path' and '--run-id' should be present.
    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--run-id',
        '3DBall',
    ]


def test_optional_configuration_entries(grim_config):
    """Tests the correct handling of optional configuration entries, including:
        --base-port
        --cpu
        --curriculum
        --debug
        --docker-target-name
        --env
        --env-args
        --export-path
        --height
        --initialize-from
        --keep-checkpoints
        --force
        --lesson
        --num-envs
        --quality-level
        --sampler
        --save-freq
        --seed
        --target-frame-rate
        --time-scale
        --width

    Arguments requiring special handling are tested elsewhere. Examples include:
        --multi-gpu
        --no-graphics
        --timestamp
        --inference
    """

    grim_config['--base-port'] = 5010
    grim_config['--cpu'] = True
    grim_config['--curriculum'] = 'curricula/3DBall.yaml'
    grim_config['--debug'] = True
    grim_config['--docker-target-name'] = 'unity-volume'
    grim_config['--env'] = 'builds/3DBall/3DBall.exe'
    grim_config['--env-args'] = ["--num-orcs", 42]
    grim_config['--export-path'] = 'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels'
    grim_config['--height'] = 240
    grim_config['--initialize-from'] = '3DBall'
    grim_config['--keep-checkpoints'] = 10
    grim_config['--force'] = True
    grim_config['--lesson'] = 2
    grim_config['--num-envs'] = 5
    grim_config['--quality-leve'] = 1
    grim_config['--sampler'] = 'config/3DBall_randomize.yaml'
    grim_config['--save-freq'] = 5000
    grim_config['--seed'] = 500
    grim_config['--run-id'] = '3DBall-2020-05-23_17-43-24'
    grim_config['--target-frame-rate'] = 60
    grim_config['--time-scale'] = 20
    grim_config['--width'] = 320

    arguments = TrainingWrapperArguments(grim_config)
    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents/training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--run-id',
        '3DBall-2020-05-23_17-43-24',
        '--env',
        'builds/3DBall/3DBall.exe',
        '--sampler',
        'config/3DBall_randomize.yaml',
        '--export-path',
        'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels',
        '--curriculum',
        'curricula/3DBall.yaml',
        '--keep-checkpoints',
        10,
        '--force',
        '--lesson',
        2,
        '--initialize-from',
        '3DBall',
        '--save-freq',
        5000,
        '--seed',
        500,
        '--base-port',
        5010,
        '--num-envs',
        5,
        '--cpu',
        '--width',
        320,
        '--height',
        240,
        '--time-scale',
        20,
        '--target-frame-rate',
        60,
        '--debug',
        '--docker-target-name',
        'unity-volume',
        '--quality-leve',
        1,
        '--env-args',
        '--num-orcs',
        42,
    ]


def test_training_arguments_timestamp(monkeypatch, grim_config):
    """Test TrainingWrapperArguments correctly applies a timestamp."""

    def mock_return():
        return '2019-06-29_17-13-41'

    monkeypatch.setattr(grimagents.common, "get_timestamp", mock_return)

    grim_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(grim_config)
    arguments_string = arguments.get_arguments_as_string()

    assert '--run-id 3DBall-2019-06-29_17-13-41' in arguments_string


def test_training_arguments_handles_timestamp(grim_config):
    """Test correct processing of the '--timestamp' argument."""

    # '--timestamp' should be present
    grim_config['--timestamp'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--timestamp' not in arguments.get_arguments()

    # '--timestamp' should not be present
    grim_config['--timestamp'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--timestamp' not in arguments.get_arguments()


def test_training_arguments_inference(grim_config):
    """Tests correct processing of the '--inference' argument.

    - Ensures '--inference' argument is added
    - Ensures get_arguments() can handle no additional args being set"
    - Ensures the '--export-path' argument is removed
    """

    grim_config['--export-path'] = 'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels'

    # '--inference' should not be present
    grim_config['--inference'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--inference' not in arguments.get_arguments()

    # '--inference' should be present and '--export-path' argument should be removed
    grim_config['--inference'] = True
    arguments = TrainingWrapperArguments(grim_config)
    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents/training_wrapper.py'

    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--run-id',
        '3DBall',
        '--inference',
    ]


def test_training_arguments_handles_no_graphics(grim_config):
    """Test for correct processing of the '--no-graphics' argument."""

    # '--no-graphics' should be present
    grim_config['--no-graphics'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--no-graphics' in arguments.get_arguments()

    # '--no-graphics' should not be present
    grim_config['--no-graphics'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--no-graphics' not in arguments.get_arguments()


def test_training_arguments_handles_multi_gpu(grim_config):
    """Test for correct processing of the '--multi-gpu' argument."""

    # '--multi-gpu' should be present
    grim_config['--multi-gpu'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--multi-gpu' in arguments.get_arguments()

    # '--multi-gpu' should not be present
    grim_config['--multi-gpu'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--multi-gpu' not in arguments.get_arguments()


def test_training_arguments_handles_cpu(grim_config):
    """Test for correct processing of the '--cpu' argument."""

    # '--cpu' should be present
    grim_config['--cpu'] = True
    arguments = TrainingWrapperArguments(grim_config)
    assert '--cpu' in arguments.get_arguments()

    # '--cpu' should not be present
    grim_config['--cpu'] = False
    arguments = TrainingWrapperArguments(grim_config)
    assert '--cpu' not in arguments.get_arguments()


def test_training_arguments_add_additional_args(grim_config):
    """Test that TrainingWrapperArguments correctly appends additional arguments."""

    arguments = TrainingWrapperArguments(grim_config)
    additional_args = ['--debug']
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
        '--run-id',
        '3DBall',
        '--debug',
    ]


def test_override_configuration_values(grim_config):
    """Test that TrainingWrapperArguments correctly applies argument overrides, including:
    --base-port
    --env
    --multi-gpu
    --no-graphics
    --num-envs
    --run-id
    --timestamp
    --trainer-config
    """

    grim_config['--base-port'] = 5010
    grim_config['--env'] = 'builds/3DBall/3DBall.exe'
    grim_config['--multi-gpu'] = True
    grim_config['--no-graphics'] = False
    grim_config['--num-envs'] = 2
    grim_config['--timestamp'] = True

    override_args = Namespace(
        args=[],
        base_port=6010,
        configuration_file='config/3DBall_grimagents.json',
        env='builds/PushBlock/PushBlock.exe',
        graphics=None,
        inference=False,
        multi_gpu=None,
        no_graphics=True,
        no_multi_gpu=True,
        no_timestamp=True,
        num_envs=4,
        resume=False,
        run_id='PushBlock',
        timestamp=None,
        trainer_config='config/PushBlock_grimagents.json',
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
        '--run-id',
        'PushBlock',
        '--env',
        'builds/PushBlock/PushBlock.exe',
        '--base-port',
        6010,
        '--num-envs',
        '4',
        '--no-graphics',
    ]


def test_inference_override_configuration_values(grim_config):
    """Test for correct handling of the '--inference' argument override."""

    # We test the override handling of the '--inference' argument by itself as it will affect the result of the '--export-path' argument.

    grim_config['--export-path'] = 'UnitySDK/Assets/ML-Agents/Examples/3DBall/ImportedModels'

    override_args = Namespace(
        configuration_file='config/3DBall_grimagents.json',
        trainer_config=None,
        resume=False,
        env=None,
        sampler=None,
        lesson=None,
        run_id=None,
        base_port=None,
        num_envs=None,
        inference=True,
        graphics=None,
        no_graphics=None,
        timestamp=None,
        no_timestamp=None,
        multi_gpu=None,
        no_multi_gpu=None,
        args=[],
    )

    arguments = TrainingWrapperArguments(grim_config)
    arguments.apply_argument_overrides(override_args)

    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents/training_wrapper.py'

    # The '--inference' argument should be present and '--export-path' should not.
    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--run-id',
        '3DBall',
        '--inference',
    ]


def test_resume_override_configuration_values(grim_config):
    """Test for correct handling of the '--resume' argument."""

    # We test handling of the '--resume' argument by itself as it will affect the result
    # of the timestamp and inference arguments.

    override_args = Namespace(
        configuration_file='config/3DBall_grimagents.json',
        trainer_config=None,
        resume=True,
        inference=True,
        timestamp=True,
        env=None,
        sampler=None,
        lesson=None,
        run_id=None,
        base_port=None,
        num_envs=None,
        graphics=None,
        no_graphics=None,
        no_timestamp=None,
        multi_gpu=None,
        no_multi_gpu=None,
        args=[],
    )

    arguments = TrainingWrapperArguments(grim_config)
    arguments.apply_argument_overrides(override_args)

    result = arguments.get_arguments()

    # The absolute path to training_wrapper.py will differ based on the system running this test.
    result[3] = 'grimagents/training_wrapper.py'

    # The inference argument should not be present and a timestamp should not be applied.
    assert result == [
        'pipenv',
        'run',
        'python',
        'grimagents/training_wrapper.py',
        'config/3DBall.yaml',
        '--run-id',
        '3DBall',
        '--resume',
    ]


def test_training_arguments_set_methods(grim_config):
    """Tests that TrainingWrapperArguments correctly sets argument values."""

    arguments = TrainingWrapperArguments(grim_config)

    arguments.set_env('builds/3DBall/3DBallHard.exe')
    arguments.set_run_id('ball')
    arguments.set_num_envs('4')
    arguments.set_no_graphics_enabled(True)
    arguments.set_timestamp_enabled(True)
    arguments.set_multi_gpu_enabled(True)
    arguments.set_env_args(["--num-orcs", 42])

    arguments_string = arguments.get_arguments_as_string()
    assert '--env builds/3DBall/3DBallHard.exe' in arguments_string
    assert '--run-id ball' in arguments_string
    assert '--num-envs 4' in arguments_string
    assert '--no-graphics' in arguments_string
    assert '--run-id ball-' in arguments_string
    assert '--multi-gpu' in arguments_string
    assert '--num-orcs 42' in arguments_string


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
