"""
Optimization functions used for hyperparameters tuning.
"""

import inspect
from sentarget.utils import rgetattr, rsetattr


def tune(model, config):
    r"""

    .. note::

        If the key is separated with a '.', it means the first index is the module to change,
        then the attribute ``key = 'LSTM.dropout'`` will modify only the dropout corresponding to ``LSTM`` layers

        The double underscore ``__`` is used to modify a specific attribute by its name (and not its type),
        like ``key = 'linear__in_features'`` will modify only the ``in_features`` attribute from the
        ``Linear`` layer saved under the attribute ``linear`` of the custom model.


    .. warning::

        The operation modify the model inplace.


    Args:
        model (Model): the model to tune its hyperparameters.
        config (dict): dictionary of parameters to change.

    Returns:
        dict: the configuration to apply to a model.


    Examples::

        >>> from sentarget.nn.models.lstm import BiLSTM
        >>> # Defines the shape of the models
        >>> INPUT_DIM = len(TEXT.vocab)
        >>> EMBEDDING_DIM = 100
        >>> HIDDEN_DIM = 128
        >>> OUTPUT_DIM = len(LABEL.vocab)
        >>> N_LAYERS = 2
        >>> BIDIRECTIONAL = True
        >>> DROPOUT = 0.25
        >>> PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        >>> model = BiLSTM(INPUT_DIM,
        ...               EMBEDDING_DIM,
        ...                HIDDEN_DIM,
        ...                OUTPUT_DIM,
        ...                N_LAYERS,
        ...                BIDIRECTIONAL,
        ...                DROPOUT,
        ...                PAD_IDX)

        >>> config = {'LSTM.dropout': 0.2}
        >>> tune(model, config)

    """
    for (key, value) in config.items():
        attribute_list = key.split('__')
        attribute = attribute_list[0]
        module_path = key.split('__')[-1]

        # Change values from the attribute's key
        if len(attribute_list) == 2:
            attribute = getattr(model, attribute)
            try:
                rsetattr(attribute, module_path, value)
            except AttributeError:
                pass
        # Change values from module's type
        elif len(attribute_list) == 1:
            attribute = '.'.join(attribute.split('.')[1:])
            for module in model.modules():
                try:
                    rsetattr(module, attribute, value)
                except AttributeError:
                    pass
        else:
            raise KeyError(f'path to attribute {key} is ambiguous. Please separate objects with a `.` or `__`. \
                           More informations at https://pages.github.uio.no/arthurd/in5550-exam/source/package.html#sentarget-optim')


def init_cls(class_instance, config):
    r"""Initialize a class instance from a set of possible values.

    .. note::

        More parameters can be added than the object need. They will just not be used.


    Args:
        class_instance (class): class to initialize.
        config (dict): possible values of init parameters.

    Returns:
        initialized object

    """
    # Get the init parameters
    arguments = inspect.getargspec(class_instance.__init__).args
    # Remove the 'self' argument, which can't be changed.
    arguments.pop(0)
    init = {key: value for (key, value) in config.items() if key in arguments}
    return class_instance(**init)


def tune_optimizer(optimizer, config):
    r"""Tune te defaults parameters for an optimizer.

    .. warning::

        The operation modify directly the ``defaults`` optimizer's dictionary.


    Args:
        optimizer (Optimizer): optimizer to tune.
        config (dict): dictionary of new parameters to set.

    """
    for (key, value) in config.items():
        if key in optimizer.defaults:
            optimizer.defaults[key] = value
