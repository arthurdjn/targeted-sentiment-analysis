"""
This module defines basic function to render a simulation, like progress bar and statistics table.
"""

import numpy as np
import time


def get_time(start_time, end_time):
    """Get ellapsed time in minutes and seconds.

    Args:
        start_time (float): strarting time
        end_time (float): ending time

    Returns:
        elapsed_mins (float): elapsed time in minutes
        elapsed_secs (float): elapsed time in seconds.

    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def progress_bar(current_index, max_index, prefix=None, suffix=None, start_time=None):
    """Display a progress bar and duration.

    Args:
        current_index (int): current state index (or epoch number).
        max_index (int): maximal numbers of state.
        prefix (str, optional): prefix of the progress bar. The default is None.
        suffix (str, optional): suffix of the progress bar. The default is None.
        start_time (float, optional): starting time of the progress bar. If not None, it will display the time
            spent from the beginning to the current state. The default is None.

    Returns:
        None. Display the progress bar in the console.

    """
    # Add a prefix to the progress bar
    prefix = "" if prefix is None else str(prefix) + " "

    # Get the percentage
    percentage = current_index * 100 // max_index
    loading = "[" + "=" * (percentage // 2) + " " * (50 - percentage // 2) + "]"
    progress_display = "\r{0}{1:3d}% | {2}".format(prefix, percentage, loading)

    # Add a suffix to the progress bar
    progress_display += "" if suffix is None else sep + str(suffix)

    # Add a timer
    if start_time is not None:
        time_min, time_sec = get_time(start_time, time.time())
        time_display = f" | Time: {time_min:2d}m {time_sec:2d}s"
        progress_display += time_display

    # Print the progress bar
    # TODO: return a string instead
    print(progress_display, end="{}".format("" if current_index < max_index else " | Done !\n"))


def describe_dict(state_dict, key_length=50, show_iter=False, capitalize=False, pad=False, sep_key=', ', sep_val='='):
    """Describe and render a dictionary. Usually, this function is called on a ``Solver`` state dictionary,
    and merged with a progress bar.

    Args:
        state_dict (dict): the dictionary to showcase.
        key_length (int): number of letter from a string name to show.
        show_iter (bool): if ``True``, show iterable. Note that this may destroy the rendering.
        capitalize (bool): if ``True`` will capitalize the keys.
        pad (bool): if ``True``, will pad the displayed number up to 4 characters.
        sep_key (string): key separator.
        sep_val (string): value separator.

    Returns:
        string: the dictionary to render.

    """
    stats_display = ""
    use_sep = False
    for idx, (key, value) in enumerate(state_dict.items()):
        key = str(key).capitalize() if capitalize else str(key)

        if isinstance(value, float):
            if use_sep:
                stats_display += sep_key
            value_display = f"{key[:key_length]}{sep_val}{value:.4f}" if pad else f"{key[:key_length]}{sep_val}{value}"
            stats_display += f"{value_display}"
            use_sep = True
        elif isinstance(value, int):
            if use_sep:
                stats_display += sep_key
            value_display = f"{key[:key_length]}{sep_val}{value:4d}" if pad else f"{key[:key_length]}{sep_val}{value}"
            stats_display += f"{value_display}"
            use_sep = True
        elif isinstance(value, bool):
            if use_sep:
                stats_display += sep_key
            stats_display += f"{key[:key_length]}{sep_val}{value}"
            use_sep = True
        elif isinstance(value, str):
            if use_sep:
                stats_display += sep_key
            stats_display += f"{key[:key_length]}{sep_val}'{value}'"
            use_sep = True
        elif (isinstance(value, list) or isinstance(value, tuple)) and show_iter:
            if use_sep:
                stats_display += sep_key
            stats_display += f"{key[:key_length]}{sep_val}{value}"
            use_sep = True

    return stats_display


def stats_dict(state_dict):
    r"""Describe statistical information from a dictionary composed of lists.

    Args:
        state_dict (dict): dictionary were cumulative information are stored.

    Returns:
        dict

    """
    stats = {'mean': {},
             'std': {},
             'max': {}}
    for (key, value) in state_dict.items():
        if isinstance(value, list):
            if isinstance(value[0], int) or isinstance(value[0], float):
                stats['mean'].update({key: float(np.mean(value))})
                stats['std'].update({key: float(np.std(value))})
                stats['max'].update({key: float(np.max(value))})
                # stats['min'].update({key: float(np.min(value))})
                # stats['q1/4'].update({key: float(np.quantile(value, 0.25))})
                # stats['q2/4'].update({key: float(np.quantile(value, 0.5))})
                # stats['q3/4'].update({key: float(np.quantile(value, 0.75))})

    return stats
