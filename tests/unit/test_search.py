import logging
import pytest

from argparse import Namespace

import grimagents.search

from grimagents.search_commands import (
    EditGrimConfigFile,
    OutputGridSearchCount,
    PerformGridSearch,
    ExportGridSearchConfiguration,
    PerformRandomSearch,
    PerformBayesianSearch,
)


@pytest.fixture
def arguments():
    return ['config\\3DBall_grimagents.json']


@pytest.fixture
def namespace_args():
    return Namespace(
        bayes_load=False,
        bayes_save=False,
        bayesian=None,
        configuration_file='config\\3DBall_grimagents.json',
        edit_config=None,
        export_index=None,
        random=None,
        resume=None,
        search_count=False,
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

    monkeypatch.setattr(grimagents.search, 'configure_logging', mock_configure_logging)
    monkeypatch.setattr(grimagents.common, 'is_pipenv_present', mock_is_pipenv_present)
    monkeypatch.setattr(grimagents.search, 'get_argvs', mock_get_argvs)
    monkeypatch.setattr(logging, 'shutdown', mock_logging_shutdown)


@pytest.fixture
def patch_search_commands(monkeypatch):
    """Patches all training commands to assert False if their execute method is called. """

    def mock_init(self, args):
        pass

    def mock_execute_edit_grim_config(self):
        assert False

    def mock_execute_output_search_count(self):
        assert False

    def mock_execute_perform_grid_search(self):
        assert False

    def mock_execute_export_grid_search_config(self):
        assert False

    def mock_execute_perform_random_search(self):
        assert False

    def mock_execute_perform_bayesian_search(self):
        assert False

    monkeypatch.setattr(EditGrimConfigFile, '__init__', mock_init)
    monkeypatch.setattr(OutputGridSearchCount, '__init__', mock_init)
    monkeypatch.setattr(PerformGridSearch, '__init__', mock_init)
    monkeypatch.setattr(ExportGridSearchConfiguration, '__init__', mock_init)
    monkeypatch.setattr(PerformRandomSearch, '__init__', mock_init)
    monkeypatch.setattr(PerformBayesianSearch, '__init__', mock_init)

    monkeypatch.setattr(EditGrimConfigFile, 'execute', mock_execute_edit_grim_config)
    monkeypatch.setattr(OutputGridSearchCount, 'execute', mock_execute_output_search_count)
    monkeypatch.setattr(PerformGridSearch, 'execute', mock_execute_perform_grid_search)
    monkeypatch.setattr(
        ExportGridSearchConfiguration, 'execute', mock_execute_export_grid_search_config
    )
    monkeypatch.setattr(PerformRandomSearch, 'execute', mock_execute_perform_random_search)
    monkeypatch.setattr(PerformBayesianSearch, 'execute', mock_execute_perform_bayesian_search)


def test_parse_args(arguments, namespace_args):
    """Tests that parse_arges() produces a Namespace object with the required attributes.
    """

    args = grimagents.search.parse_args(arguments)
    assert args == namespace_args


def test_edit_grim_config(monkeypatch, patch_main, namespace_args, patch_search_commands):
    """Tests that EditGrimConfigFile is executed."""

    namespace_args.edit_config = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.search, 'parse_args', mock_parse_args)
    monkeypatch.setattr(EditGrimConfigFile, 'execute', mock_execute)

    grimagents.search.main()


def test_output_grid_search_count(monkeypatch, patch_main, namespace_args, patch_search_commands):
    """Tests that OutputGridSearchCount is executed."""

    namespace_args.search_count = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.search, 'parse_args', mock_parse_args)
    monkeypatch.setattr(OutputGridSearchCount, 'execute', mock_execute)

    grimagents.search.main()


def test_perform_grid_search(monkeypatch, patch_main, namespace_args, patch_search_commands):
    """Tests that PerformGridSearch is executed."""

    # PerformGridSearch is the default behaviour so namespace_args does not need to be modified.

    def mock_execute(self):
        assert True

    monkeypatch.setattr(PerformGridSearch, 'execute', mock_execute)

    grimagents.search.main()


def test_export_grid_search_config(monkeypatch, patch_main, namespace_args, patch_search_commands):
    """Tests that ExportGridSearchConfiguration is executed."""

    namespace_args.export_index = True

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.search, 'parse_args', mock_parse_args)
    monkeypatch.setattr(ExportGridSearchConfiguration, 'execute', mock_execute)

    grimagents.search.main()


def test_perform_random_search(monkeypatch, patch_main, namespace_args, patch_search_commands):
    """Tests that PerformRandomSearch is executed."""

    namespace_args.random = 1

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.search, 'parse_args', mock_parse_args)
    monkeypatch.setattr(PerformRandomSearch, 'execute', mock_execute)

    grimagents.search.main()


def test_perform_bayesian_search(monkeypatch, patch_main, namespace_args, patch_search_commands):
    """Tests that PerformBayesianSearch is executed."""

    namespace_args.bayesian = [1, 1]

    def mock_parse_args(argvs):
        return namespace_args

    def mock_execute(self):
        assert True

    monkeypatch.setattr(grimagents.search, 'parse_args', mock_parse_args)
    monkeypatch.setattr(PerformBayesianSearch, 'execute', mock_execute)

    grimagents.search.main()
