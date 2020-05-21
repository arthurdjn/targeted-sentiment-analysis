r"""
Defines a model template.
A `Model` is really similar to the `Module` class, except that a `Model` has more inner methods,
used to train, evaluate and test a neural network.

The *API* is similar to sklearn or tensorflow.

.. code-block:: python

    class Net(Model):
        def __init__(self, *args):
            super(Model, self).__init__()
            # initialize your module as usual

        def forward(*args):
            # one forward step
            pass

        def run(train_iterator, criterion, optimizer):
            # train one single time the network
            pass

        def evaluate(eval_iterator, criterion):
            # evaluate one single time the network
            pass

        def predict(test_iterator):
            # predict one single time the network
            pass


    # Run and train the model
    model = Net()
    model.fit(epochs, train_iterator, eval_iterator, criterion, optimizer)

"""

from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.optim as optim

# Data science
import os
from pathlib import Path
import time
import copy
from sentarget.utils import append2dict, describe_dict, stats_dict


class Model(nn.Module, ABC):
    r"""
    A `Model` is used to define a neural network.
    This template is easier to handle for hyperparameters optimization, as the ``fit``, ``train``, ``evaluate``
    methods are part of the model.

    * :attr:`checkpoint` (dict): checkpoint of the best model tested.

    * :attr:`criterion` (Loss): loss function.

    * :attr:`optimizer` (Optimizer): optimizer for weights and biases.

    * :attr:`performance` (dict): dictionary where performances are stored.

        * ``'train'`` (dict): training dictionary.
        * ``'eval'`` (dict): testing dictionary.

    """

    def __init__(self):
        super().__init__()
        # Performances
        self.checkpoint = None
        self.performance = None
        self.reset()

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def reset(self):
        """Reset the performance and associated checkpoint dictionary."""
        self.checkpoint = {
            'epoch': None,
            'model_name': None,
            'model_state_dict': None,
            'optimizer_name': None,
            'criterion_name': None,
            'optimizer_state_dict': None,
            'train': None,
            'eval': None
        }
        self.performance = {
            "train": {},
            "eval": {}
        }

    @abstractmethod
    def run(self, iterator, criterion, optimizer, *args, **kwargs):
        r"""Train one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.
            criterion (Loss): loss function to measure scores.
            optimizer (Optimizer): optimizer used during training to update weights.

        Returns:
            dict: the performance and metrics of the training session.

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, iterator, criterion, *args, **kwargs):
        r"""Evaluate one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.
            criterion (Loss): loss function to measure scores.

        Returns:
            dict: the performance and metrics of the training session.

        """
        raise NotImplementedError

    def predict(self, iterator, *args, **kwargs):
        r"""Predict the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.

        Returns:
            dict: the performance and metrics of the training session.

        """
        raise NotImplementedError

    def _update_checkpoint(self, epoch, criterion, optimizer, results_train=None, results_eval=None):
        r"""Update the model's checkpoint. Keep track of its epoch, state, optimizer,
        and performances. In addition, it saves the current model in `best_model`.

        Args:
            epoch (int): epoch at the current training state.
            criterion (Loss): loss function to measure scores.
            optimizer (Optimizer): optimizer used during training to update weights.
            results_train (dict, optional): metrics for the training session at epoch. The default is ``None``.
            results_eval (dict, optional): metrics for the evaluation session at epoch. The default is ``None``.

        """
        self.checkpoint = {
            'epoch': epoch,
            'model_name': self.__class__.__name__,
            'model_state_dict': copy.deepcopy(self.state_dict()),
            'optimizer_name': optimizer.__class__.__name__,
            'criterion_name': criterion.__class__.__name__,
            'train': results_train,
            'eval': results_eval
        }

    def fit(self, train_iterator, eval_iterator,
            criterion=None, optimizer=None, epochs=10, verbose=True, compare_on='accuracy', **kwargs):
        r"""Train and evaluate a model X times. During the training, both training
        and evaluation results are saved under the `performance` attribute.

        Args:
            train_iterator (Iterator): iterator containing batch samples of data.
            eval_iterator (Iterator): iterator containing batch samples of data.
            epochs (int): number of times the model will be trained.
            criterion (Loss): loss function to measure scores.
            optimizer (Optimizer): optimizer used during training to update weights.
            verbose (bool, optional): if ``True`` display a progress bar and metrics at each epoch.
            compare_on (string): name of the score on which models are compared.

        Returns:
            Model: the best model evaluated.

        Examples::

            >>> model = MyModel()
            >>> # Train & eval EPOCHS times
            >>> criterion = nn.CrossEntropyLoss()
            >>> optimizer = metrics.Adam(model.parameters())
            >>> EPOCHS = 10
            >>> model.fit(train_iterator, eval_iterator, epochs=EPOCHS, criterion=criterion, optimizer=optimizer)
                Epoch:        1/10
                Training:     100% | [==================================================]
                Evaluation:   100% | [==================================================]
                Stats Training:    | Loss: 0.349 | Acc: 84.33% | Prec.: 84.26%
                Stats Evaluation:  | Loss: 0.627 | Acc: 72.04% | Prec.: 72.22%
            >>> # ...

        """
        self.reset()
        # Keep track of the best model
        best_model = None
        best_eval_score = 0
        start_time = time.time()

        # Default update rules
        criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        optimizer = optim.Adam(self.parameters()) if optimizer is None else optimizer

        # Train and evaluate the model epochs times
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch:\t{epoch + 1:3d}/{epochs}")

            # Train and evaluate the model
            results_train = self.run(train_iterator, criterion, optimizer, **{**kwargs, 'verbose': verbose})
            results_eval = self.evaluate(eval_iterator, criterion, **{**kwargs, 'verbose': verbose})
            # Update the eval dictionary by adding the results at the current epoch
            append2dict(self.performance["train"], results_train)
            append2dict(self.performance["eval"], results_eval)

            if verbose:
                print("\t Stats Train: | " + describe_dict(results_train, pad=True, capitalize=True, sep_val=': ', sep_key=' | '))
                print("\t  Stats Eval: | " + describe_dict(results_eval, pad=True, capitalize=True, sep_val=': ', sep_key=' | '))
                print()
            # We copy in memory the best model
            if best_eval_score < self.performance["eval"][compare_on][-1]:
                best_eval_score = self.performance["eval"][compare_on][-1]
                self._update_checkpoint(epoch + 1, criterion, optimizer, results_train=results_train, results_eval=results_eval)
                best_model = copy.deepcopy(self)

        self.performance['time'] = time.time() - start_time

        return best_model

    def describe_performance(self, *args, **kwargs):
        """Get a display of the last performance for both train and eval.

        Returns:
            tuple: two strings showing statistics for train and eval sessions.

        """
        dict_train = {key: performance[-1] for (key, performance) in self.performance['train'].items()}
        dict_eval = {key: performance[-1] for (key, performance) in self.performance['eval'].items()}
        return describe_dict(dict_train, *args, **kwargs), describe_dict(dict_eval, *args, **kwargs)

    def state_json(self):
        r"""Return a serialized ``state_dict``, so it can be saved as a ``json``.

        Returns:
            dict

        """
        state = {key: value.tolist() for (key, value) in self.state_dict().items()}
        return state

    def log_perf(self, **kwargs):
        """Get a log from the performances."""
        describe_train, describe_eval = self.describe_performance(pad=True, **kwargs)
        stats_train = stats_dict(self.performance['train'])
        stats_eval = stats_dict(self.performance['eval'])
        log = f"Performances(\n"
        log += f"  (train): Scores({describe_train})\n"
        log += f"  (eval): Scores({describe_eval})\n"
        for (key_train, stat_train), (key_eval, stat_eval) in zip(stats_train.items(), stats_eval.items()):
            log += f"  (train): {str(key_train).capitalize()}({describe_dict(stat_train, pad=True, **kwargs)})\n"
            log += f"  (eval) {str(key_eval).capitalize()}({describe_dict(stat_eval, pad=True, **kwargs)})\n"
        log += ')'
        return log

    def save(self, filename='model.pt', checkpoint=True):
        r"""Save the best torch model.

        Args:
            filename (string, optional): name of the file.
            checkpoint (bool, optional): True to save the model at the best checkpoint during training.

        """
        # Create the directory if it does not exists
        dirname = os.path.dirname(filename)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        torch.save(self, filename)
        # Save its checkpoint
        if checkpoint:
            epoch = self.checkpoint['epoch']
            basename = os.path.basename(filename)
            name = basename.split('.')[0]
            checkname = f"checkpoint_{name}_epoch{epoch}.pt"
            torch.save(self.checkpoint, os.path.join(dirname, checkname))