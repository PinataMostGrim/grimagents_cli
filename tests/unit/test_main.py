import logging
import pytest

from argparse import Namespace

import grimagents.__main__
import grimagents.common

from grimagents.training_commands import (
    ListTrainingOptions,
    EditGrimConfigFile,
    EditTrainerConfigFile,
    EditCurriculumFile,
    StartTensorboard,
    PerformTraining,
    ResumeTraining,
)


@pytest.fixture
def arguments():
    return ['config/3DBall_grimagents.json']


@pytest.fixture
def namespace_args():
    return Namespace(
        additional_args=[],
        base_port=None,
        configuration_file='config/3DBall_grimagents.json',
        dry_run=False,
        edit_config=None,
        edit_curriculum=None,
        edit_trainer_config=None,
        env=None,
        graphics=False,
        inference=False,
        lesson=None,
        list=False,
        multi_gpu=False,
        no_graphics=False,
        no_multi_gpu=False,
        no_timestamp=False,
        num_envs=None,
        resume=False,
        run_id=None,
        sampler=None,
        tensorboard_start=False,
        timestamp=False,
        trainer_config=None,
    )


@pytest.fixture
def patch_main(monkeypatch, arguments, namespace_args):
    def mock_configure_logging():
        pass

    def mock_is_pipenv_present():
        return True

    def mock_get_argvs():
        return arguments

    def mock_logging_shutdown():
        pass

    monkeypatch.setattr(grimagents.__main__, 'configure_logging', mock_configure_logging)
    monkeypatch.setattr(grimagents.common, 'is_pipenv_present', mock_is_pipenv_present)
    monkeypatch.setattr(grimagents.__main__, 'get_argvs', mock_get_argvs)
    monkeypatch.setattr(logging, 'shutdown', mock_logging_shutdown)


@pytest.fixture
def patch_training_commands(monkeypatch):
    """Patches all training commands to assert False if their execute method is called. """

    def mock_init(self, args):
        pass

    def mock_execute_list_options(self):
        assert False

    def mock_execute_edit_grim_config(self):
        assert False

    def mock_execute_edit_trainer_config(self):
        assert False

    def mock_execute_edit_curriculum(self):
        assert False

    def mock_execute_start_tensorboard(self):
        assert False

    def mock_execute_perform_training(self):
        assert False

    def mock_execute_resume_training(self):
        assert False

    monkeypatch.setattr(ListTrainingOptions, '__init__', mock_init)
    monkeypatch.setattr(EditGrimConfigFile, '__init__', mock_init)
    monkeypatch.setattr(EditTrainerConfigFile, '__init__', mock_init)
    monkeypatch.setattr(EditCurriculumFile, '__init__', mock_init)
    monkeypatch.setattr(StartTensorboard, '__init__', mock_init)
    monkeypatch.setattr(PerformTraining, '__init__', mock_init)
    monkeypatch.setattr(ResumeTraining, '__init__', mock_init)

    monkeypatch.setattr(ListTrainingOptions, 'execute', mock_execute_list_options)
    monkeypatch.setattr(EditGrimConfigFile, 'execute', mock_execute_edit_grim_config)
    monkeypatch.setattr(EditTrainerConfigFile, 'execute', mock_execute_edit_trainer_config)
    monkeypatch.setattr(EditCurriculumFile, 'execute', mock_execute_edit_curriculum)
    monkeypatch.setattr(StartTensorboard, 'execute', mock_execute_start_tensorboard)
    monkeypatch.setattr(PerformTraining, 'execute', mock_execute_perform_training)
    monkeypatch.setattr(ResumeTraining, 'execute', mock_execute_resume_training)


def test_parse_args(arguments, namespace_args):
    """Tests that parse_arges() produces a Namespace object with the required attributes.
    """

    args = grimagents.__main__.parse_args(arguments)
    assert args == namespace_args


def test_list_training_options(monkeypatch, patch_main, namespace_args, patch_training_commands):
    """Tests that ListTrainingOptions is executed."""

    namespace_args.list = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.__main__, 'parse_args', mock_parse_args)
    monkeypatch.setattr(ListTrainingOptions, 'execute', mock_execute)

    grimagents.__main__.main()


def test_edit_grim_config_file(monkeypatch, patch_main, namespace_args, patch_training_commands):
    """Tests that EditGrimConfigFile is executed."""

    namespace_args.edit_config = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.__main__, 'parse_args', mock_parse_args)
    monkeypatch.setattr(EditGrimConfigFile, 'execute', mock_execute)

    grimagents.__main__.main()


def test_edit_trainer_config(monkeypatch, patch_main, namespace_args, patch_training_commands):
    """Tests that ListTrainingOptions is executed."""

    namespace_args.edit_trainer_config = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.__main__, 'parse_args', mock_parse_args)
    monkeypatch.setattr(EditTrainerConfigFile, 'execute', mock_execute)

    grimagents.__main__.main()


def test_edit_curriculum(monkeypatch, patch_main, namespace_args, patch_training_commands):
    """Tests that EditCurriculumFile is executed."""

    namespace_args.edit_curriculum = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.__main__, 'parse_args', mock_parse_args)
    monkeypatch.setattr(EditCurriculumFile, 'execute', mock_execute)

    grimagents.__main__.main()


def test_start_tensorboard(monkeypatch, patch_main, namespace_args, patch_training_commands):
    """Tests that StartTensorboard is executed."""

    namespace_args.tensorboard_start = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.__main__, 'parse_args', mock_parse_args)
    monkeypatch.setattr(StartTensorboard, 'execute', mock_execute)

    grimagents.__main__.main()


def test_perform_training(monkeypatch, patch_main, patch_training_commands):
    """Tests that PerformTraining is executed."""

    # PerformTraining is the default behaviour so namespace_args does not need to be modified.

    def mock_execute(self):
        assert True

    monkeypatch.setattr(PerformTraining, 'execute', mock_execute)

    grimagents.__main__.main()


def test_resume_training(monkeypatch, patch_main, namespace_args, patch_training_commands):
    """Tests that ResumeTraining is executed."""

    namespace_args.resume = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.__main__, 'parse_args', mock_parse_args)
    monkeypatch.setattr(ResumeTraining, 'execute', mock_execute)

    grimagents.__main__.main()
