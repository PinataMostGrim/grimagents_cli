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
    command_list = ['config\\3DBall.yaml', '--env', 'builds\\3DBall\\Unity Environment.exe', '--export-path', 'UnitySDK\\Assets\\ML-Agents\\Examples\\3DBall\\ImportedModels', '--run-id', '3DBall']
    assert command.get_command() == command_list


def test_training_command_handles_no_graphics():
    """Test for correct handling of the --no-graphics argument."""

    # --no-graphics should be present
    test_config = {
        "--no-graphics": True,
    }
    command = TrainingCommand(test_config)
    assert '--no-graphics' in command.get_command()

    # --no-graphics should not be present
    test_config = {
        "--no-graphics": False,
    }
    command = TrainingCommand(test_config)
    assert '--no-graphics' not in command.get_command()


def test_training_command_excludes_timestamp():
    """Test for ensuring TrainingWrapperArguments excludes the --timestamp argument."""

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


