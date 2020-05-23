"""
Utility functions.
"""

import functools
import itertools

import torch


def append2dict(main_dict, *dicts):
    """
    Append key values to another dict with the same keys.

    Args:
        main_dict (dict): dictionary where values will be added.
        *dicts (dict): dictionaries to extract values and append to another one.
            These dictionaries should have the same keys as dict.

    Examples::

        >>> dict1 = {"key1": [], "key2": []}
        >>> dict2 = {"key1": 0, "key2": 1}
        >>> append2dict(dict1, dict2)
        >>> dict1
            {"key1": [0], "key2": [1]}

        >>> dict3 = {"key1": 2, "key2": 3}
        >>> dict4 = {"key1": 4, "key2": 5}
        >>> append2dict(dict1, dict3, dict4)
        >>> dict1
            {"key1": [0, 2, 4], "key2": [1, 3, 5]}

    """
    # Multiples dictionaries to merge
    for d in dicts:
        for (key, value) in d.items():
            # Test if the dictionary to append have the key
            try:
                main_dict[key].append(value)
            # If not, create the key and merge the value
            except:
                main_dict[key] = [value]


def permutation_dict(params):
    r"""Generate a list of configuration files used to tune a model.

    Returns:
        list

    Examples::

        >>> hyper_params = {'dropout': [0, 0.1, 0.2, 0.3],
        ...                 'in_features': [10, 20, 30, 40],
        ...                 'out_features': [20, 30, 40, 50]}

        >>> permutation_dict(hyper_params)
            [{'dropout': 0, 'in_features': 10, 'out_features': 20},
             {'dropout': 0, 'in_features': 10, 'out_features': 30},
             {'dropout': 0, 'in_features': 10, 'out_features': 40},
             {'dropout': 0, 'in_features': 10, 'out_features': 50},
             {'dropout': 0, 'in_features': 20, 'out_features': 20},
             {'dropout': 0, 'in_features': 20, 'out_features': 30},
             ...
            ]

    """
    params_list = {key: value for (key, value) in params.items() if isinstance(value, list)}
    params_single = {key: value for (key, value) in params.items() if not isinstance(value, list)}
    keys, values = zip(*params_list.items())
    permutations = [dict(zip(keys, v), **params_single) for v in itertools.product(*values)]
    return permutations


def serialize_dict(data):
    r"""Serialize recursively a dict to another dict composed of basic python object (list, dict, int, float, str...)

    Args:
        data (dict): dict to serialize

    Returns:
        dict


    Examples::

        >>> data = {'tensor': torch.tensor([0, 1, 2, 3, 4]),
        ...         'sub_tensor': [torch.tensor([1, 2, 3, 4, 5])],
        ...         'data': [1, 2, 3, 4, 5],
        ...         'num': 1}
        >>> serialize_dict(data)
            {'tensor': None,
             'sub_tensor': [],
             'data': [1, 2, 3, 4, 5],
             'num': 1}

    """
    new_data = {}
    for (key, value) in data.items():
        if isinstance(value, dict):
            new_data[key] = serialize_dict(value)
        elif isinstance(value, list):
            new_data[key] = serialize_list(value)
        elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool):
            new_data[key] = value
        else:
            new_data[str(key)] = None

    return new_data


def serialize_list(data):
    """Serialize recursively a list to another list composed of basic python object (list, dict, int, float, str...)

    Args:
        data (list): list to serialize

    Returns:
        list


    Examples::

        >>> data = [1, 2, 3, 4]
        >>> serialize_list(data)
            [1, 2, 3, 4]
        >>> data = [torch.tensor([1, 2, 3, 4])]
        >>> serialize_list(data)
            []
        >>> data = [1, 2, 3, 4, torch.tensor([1, 2, 3, 4])]
        >>> serialize_list(data)
            [1, 2, 3, 4]

    """
    new_data = []
    for value in data:
        if isinstance(value, list):
            new_data.append(serialize_list(value))
        elif isinstance(value, dict):
            new_data.append(serialize_dict(value))
        elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool):
            new_data.append(value)
        else:
            return []

    return new_data


def rsetattr(obj, attr, val):
    r"""Set an attribute recursively.

    ..note ::

        Attributes should be split with a dot ``.``.

    Args:
        obj (object): object to set the attribute.
        attr (string): path to the attribute.
        val (value): value to set.

    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    r"""Get an attribute recursively.

    Args:
        obj (object): object to get the attribute.
        attr (string): path to the attribute.
        *args:

    Returns:
        attribute

    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
