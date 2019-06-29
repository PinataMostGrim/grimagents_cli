from grimagents.commands import TrainingCommand


TEST_CONFIG = {
    "trainer-config-path": "config\\3DBall.yaml",
    "--env": "builds\\3DBall\\Unity Environment.exe",
    "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
    "--curriculum": "",
    "--keep-checkpoints": "",
    "--lesson": "",
    "--run-id": "3DBall",
    "--num-runs": "",
    "--save-freq": "",
    "--seed": "",
    "--base-port": "",
    "--num-envs": "",
    "--no-graphics": False,
    "--timestamp": True,
    "--log-filename": ""
}


def test_create_training_command():
    """Test for creating TrainingCommand with default configuration."""

    command = TrainingCommand(TEST_CONFIG)
    command_list = ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--env', 'builds\\3DBall\\Unity Environment.exe', '--export-path', 'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels', '--run-id', '3DBall', '--train']
    assert command.get_command() == command_list


def test_training_command_handles_no_graphics():
    """Test for correct handling of the --no-graphics argument."""

    # '--no-graphics' should be present
    test_config = {
        "--no-graphics": True,
    }
    command = TrainingCommand(test_config)
    assert '--no-graphics' in command.get_command()

    # '--no-graphics' should not be present
    test_config = {
        "--no-graphics": False,
    }
    command = TrainingCommand(test_config)
    assert '--no-graphics' not in command.get_command()


def test_training_command_excludes_timestamp():
    """Test for ensuring TrainingWrapperArguments excludes the '--timestamp' argument."""

    test_config = {
        "--timestamp": True,
    }
    command = TrainingCommand(test_config)
    assert '--timestamp' not in command.get_command()

    test_config = {
        "--timestamp": False,
    }
    command = TrainingCommand(test_config)
    assert '--timestamp' not in command.get_command()


def test_training_command_add_additional_args():
    """Test for TrainingCommand correctly setting additional arguments."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe"
    }
    command = TrainingCommand(test_config)
    additional_args = ['--slow', '--load']
    command.set_additional_arguments(additional_args)
    command_list = ['pipenv', 'run', 'python', 'grim-agents\\grimagents\\training_wrapper.py', 'config\\3DBall.yaml', '--env', 'builds\\3DBall\\Unity Environment.exe', '--slow', '--load', '--train']
    assert command.get_command() == command_list


def test_training_command_override_args():
    """Test for TrainingCommand correctly overriding argument values."""

    test_config = {
        "trainer-config-path": "config\\3DBall.yaml",
        "--env": "builds\\3DBall\\Unity Environment.exe",
        "--export-path": "UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels",
        "--curriculum": "",
        "--keep-checkpoints": "",
        "--lesson": "",
        "--run-id": "3DBall",
        "--num-runs": "",
        "--save-freq": "",
        "--seed": "",
        "--base-port": "",
        "--num-envs": "",
        "--no-graphics": False,
        "--timestamp": True,
        "--log-filename": ""
    }
    command = TrainingCommand(test_config)

    command.set_env('builds\\3DBall\\3DBall.exe')
    command.set_lesson('3')
    command.set_run_id('ball')
    command.set_num_envs('4')
    command.set_no_graphics_enabled(True)
    command.set_log_filename('3DBall.log')

    command_string = command.get_command_as_string()
    assert '--env builds\\3DBall\\3DBall.exe' in command_string
    assert '--lesson 3' in command_string
    assert '--run-id ball' in command_string
    assert '--num-envs 4' in command_string
    assert '--no-graphics' in command_string
    assert '--log-filename 3DBall.log' in command_string
