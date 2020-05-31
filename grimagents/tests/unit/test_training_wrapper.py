import pytest
import shutil

from argparse import Namespace
from pathlib import Path

import grimagents.training_wrapper

from grimagents.training_wrapper import TrainingRunInfo


@pytest.fixture
def arguments():
    return [
        'config/3DBall_config.yaml',
        '--env',
        'builds/3DBall/3DBall.exe',
        '--run-id',
        '3DBall',
    ]


@pytest.fixture
def namespace_args():
    return Namespace(
        args=['--env', 'builds/3DBall/3DBall.exe'],
        export_path=None,
        run_id='3DBall',
        trainer_config_path='config/3DBall_config.yaml',
    )


@pytest.fixture
def training_output():
    return [
        '\t\tmax_steps:\t3.0e3',
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.',
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 2000. Time Elapsed: 22.049 s Mean Reward: 1.454. Std of Reward: 0.878. Training.',
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 3000. Time Elapsed: 30.652 s Mean Reward: 1.763. Std of Reward: 1.054. Training.',
        'Final Mean Reward: 1.763',
        'INFO:mlagents.trainers:Exported ./models/3DBall_00/3DBallLearning.nn file',
    ]


@pytest.fixture
def export_brains():
    return [
        Path(__file__).parent / '3DBallLearning.nn',
        Path(__file__).parent / '3DBallHardLearning.nn',
    ]


@pytest.fixture
def export_brains_destination():
    return Path(__file__).parent / 'export_folder'


@pytest.fixture
def fixture_export_brains(export_brains, export_brains_destination):
    """Fixture that creates test brains to export then deletes them and the export folder upon test completion."""

    if export_brains_destination.exists():
        shutil.rmtree(export_brains_destination)

    for brain in export_brains:
        with brain.open('w') as f:
            f.write('Test brain')

    yield 'fixture_export_brains'

    for brain in export_brains:
        if brain.exists():
            brain.unlink()

    if export_brains_destination.exists():
        shutil.rmtree(export_brains_destination)


def test_parse_args(arguments, namespace_args):

    args = grimagents.training_wrapper.parse_args(arguments)
    assert args == namespace_args


def test_parse_max_steps(training_output):
    """Test that TrainingRunInfo can parse a max_steps value from console output."""

    info = TrainingRunInfo()

    info.update_from_training_output('Final Mean Reward: 1.763')
    assert info.max_steps == 0

    for line in training_output:
        info.update_from_training_output(line)

    assert info.max_steps == 3000


def test_parse_steps(training_output):
    """Test that TrainingRunInfo can parse a steps value from console output."""

    info = TrainingRunInfo()

    for line in training_output:
        info.update_from_training_output(line)

    assert info.step == 3000

    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.'
    )

    assert info.step == 1000


def test_parse_time_elapsed(training_output):
    """Test that TrainingRunInfo can parse a time elapsed value from console output."""

    info = TrainingRunInfo()

    for line in training_output:
        info.update_from_training_output(line)

    assert info.time_elapsed == 30.652

    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.'
    )

    assert info.time_elapsed == 12.398


def test_parse_mean_reward(training_output):
    """Test that TrainingRunInfo can parse a mean reward value from console output."""

    info = TrainingRunInfo()

    for line in training_output:
        info.update_from_training_output(line)

    assert info.mean_reward == 1.763

    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.'
    )

    assert info.mean_reward == 1.259


def test_parse_exported_brains(training_output):
    """Test that TrainingRunInfo can parse exported brain values from console output."""

    info = TrainingRunInfo()

    for line in training_output:
        info.update_from_training_output(line)

    assert info.exported_brains == [Path('./models/3DBall_00/3DBallLearning.nn')]

    info.update_from_training_output(
        'INFO:mlagents.trainers:Exported ./models/3DBall_00/3DBallHardLearning.nn file'
    )

    assert info.exported_brains == [
        Path('./models/3DBall_00/3DBallLearning.nn'),
        Path('./models/3DBall_00/3DBallHardLearning.nn'),
    ]


def test_time_remaining():
    """Test that TrainingRunInfo can calculate steps_remaining and time_remaining as well as handle edge cases.

    Edge cases include:
        - all values have yet to be found
        - max_steps has not been found
        - steps matches max_steps value
        - steps value exceeds max_steps value
    """

    info = TrainingRunInfo()

    assert info.time_remaining == 0

    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.'
    )
    assert info.time_remaining == 0

    info.max_steps = 3000
    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.'
    )

    assert info.time_remaining == 24.796

    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 3000. Time Elapsed: 30.652 s Mean Reward: 1.763. Std of Reward: 1.054. Training.'
    )

    assert info.time_remaining == 0.0

    info.update_from_training_output(
        'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 4000. Time Elapsed: 30.652 s Mean Reward: 1.763. Std of Reward: 1.054. Training.'
    )

    assert info.time_remaining == 0.0


def test_line_has_time_elapsed():
    """Test that TrainingRunInfo can parse time elapsed values from console output."""

    info = TrainingRunInfo()

    assert info.line_has_time_elapsed('\t\tmax_steps:\t3.0e3') is False

    assert (
        info.line_has_time_elapsed(
            'INFO:mlagents.trainers: 3DBall_00: 3DBallLearning: Step: 1000. Time Elapsed: 12.398 s Mean Reward: 1.259. Std of Reward: 0.799. Training.'
        )
        is True
    )


def test_export_brains(export_brains, export_brains_destination, fixture_export_brains):
    """Tests that training_wrapper correctly copies brain files into a destination folder. This also implicitly tests that the destination folder is created if it does not already exist.
    """

    grimagents.training_wrapper.export_brains(export_brains, export_brains_destination)

    for brain in export_brains:
        assert (export_brains_destination / brain.name).exists()
