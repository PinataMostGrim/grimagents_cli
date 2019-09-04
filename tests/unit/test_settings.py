from pathlib import Path

import grimagents.settings as settings


def test_training_wrapper_path():
    """Tests for the correct path to training_wrapper.py"""

    assert settings.get_training_wrapper_path().parts[-1] == ('training_wrapper.py')
    assert settings.get_training_wrapper_path().exists()
