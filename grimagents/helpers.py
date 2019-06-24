"""General package helper methods."""

from datetime import datetime


def get_timestamp():
    """Fetch the current time as a string.

    Returns:
      The current time as a string.
    """

    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def get_human_readable_duration(seconds):
    """Parses seconds into a human readable string.

    Returns:
      A human readable string.
    """

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
