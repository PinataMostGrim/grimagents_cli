"""Methods useful across package modules."""

from datetime import datetime
import subprocess


def is_pipenv_present():
    """Returns True if a virtual environment can be accessed through Pipenv and False if it can't.
    """

    try:
        process = subprocess.run(
            ['pipenv', '--venv'],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        # Guard against Pipenv not being accessible via a subprocess call.
        return False

    return process.returncode == 0


def get_timestamp():
    """Fetch the current time as a string."""

    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def get_human_readable_duration(seconds):
    """Parses seconds into a human readable string."""

    if seconds < 0:
        return '0 seconds'

    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if seconds < 1:
        seconds = 1

    locals_ = locals()
    magnitudes_str = (
        f'{int(locals_[magnitude])} {magnitude}'
        for magnitude in ('days', 'hours', 'minutes', 'seconds')
        if locals_[magnitude]
    )

    result = ", ".join(magnitudes_str)

    return result
