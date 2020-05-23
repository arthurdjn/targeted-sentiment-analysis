r"""
A ``Solver`` is an object used for training, evaluating and testing a model.
The performance is stored in a dictionary, both for training and testing.
In addition, the best model occurred during training is stored,
as well as it's checkpoint to re-load a model at a specific epoch.

Example:

.. code-block:: python

    import torch.nn as nn
    import torch.optim as optim

    model = nn.Sequential(nn.Linear(10, 100), nn.Sigmoid(), nn.Linear(100, 5), nn.ReLU())
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = LABEL_PAD_IDX)

    solver = BiLSTMSolver(model, optimizer=optimizer, criterion=criterion)

    # epochs = number of training loops
    # train_iterator = Iterator, DataLoader... Training data
    # eval_iterator = Iterator, DataLoader... Eval data
    solver.fit(train_iterator, eval_iterator, epochs=epochs)

"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim

# Data science
import os
from pathlib import Path
import time
import copy
from sentarget.utils import append2dict, describe_dict, deprecated


@deprecated("Solver instance is deprecated since v0.2. Please use the `Model` class to encapsulate your models instead.")
class Solver(ABC):
    r"""Train and evaluate model.

        * :attr:`model` (Module): model to optimize or test.

        * :attr:`checkpoint` (dict): checkpoint of the best model tested.

        * :attr:`criterion` (Loss): loss function.

        * :attr:`optimizer` (Optimizer): optimizer for weights and biases.

        * :attr:`performance` (dict): dictionary where performances are stored.

            * ``'train'`` (dict): training dictionary.
            * ``'eval'`` (dict): testing dictionary.


    Args:
        model (Module): model to optimize or test.
        criterion (Loss): loss function.
        optimizer (Optimizer): optimizer for weights and biases.

    """

    def __init__(self, model, criterion=None, optimizer=None):
        # Defaults attributes
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.optimizer = optim.Adam(model.parameters()) if optimizer is None else optimizer

        # Performances
        self.best_model = None
        self.performance = None
        self.checkpoint = None
        self.reset()

    def reset(self):
        """Reset the performance dictionary."""
        self.best_model = None
        self.checkpoint = {'epoch': None,
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
    def train(self, iterator, *args, **kwargs):
        r"""Train one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.

        Returns:
            dict: the performance and metrics of the training session.

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, iterator, *args, **kwargs):
        r"""Evaluate one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.

        Returns:
            dict: the performance and metrics of the training session.

        """
        raise NotImplementedError

    def _update_checkpoint(self, epoch, results_train=None, results_eval=None):
        r"""Update the model's checkpoint. Keep track of its epoch, state, optimizer,
        and performances. In addition, it saves the current model in `best_model`.

        Args:
            epoch (int): epoch at the current training state.
            results_train (dict, optional): metrics for the training session at epoch. The default is None.
            results_eval (dict, optional): metrics for the evaluation session at epoch. The default is None.

        """
        self.best_model = copy.deepcopy(self.model)
        self.checkpoint = {'epoch': epoch,
                           'model_name': self.best_model.__class__.__name__,
                           'model_state_dict': self.best_model.state_dict(),
                           'optimizer_name': self.optimizer.__class__.__name__,
                           'criterion_name': self.criterion.__class__.__name__,
                           'train': results_train,
                           'eval': results_eval
                           }

    def save(self, filename=None, dirpath=".", checkpoint=True):
        r"""Save the best torch model.

        Args:
            filename (str, optional): name of the model. The default is "model.pt".
            dirpath (str, optional): path to the desired foldre location. The default is ".".
            checkpoint (bool, optional): ``True`` to save the model at the best checkpoint during training.

        """
        if checkpoint:
            # Get the name and other relevant information
            model_name = self.checkpoint['model_name']
            epoch = self.checkpoint['epoch']
            filename = f"model_{model_name}_epoch{epoch}.pt" if filename is None else filename
            # Save in the appropriate directory, and create it if it doesn't exists
            Path(dirpath).mkdir(parents=True, exist_ok=True)
            # Save the best model
            path = os.path.join(dirpath, filename)
            torch.save(self.best_model, path)
            # Save its checkpoint
            checkname = f"checkpoint_{filename.split('.')[-2].split('_')[1]}_epoch{epoch}.pt"
            checkpath = os.path.join(dirpath, checkname)
            torch.save(self.checkpoint, checkpath)
        else:
            model_name = self.checkpoint['model_name']
            filename = f"model_{model_name}.pt" if filename is None else filename
            torch.save(self.model, filename)

    def get_accuracy(self, y_tilde, y):
        r"""Compute accuracy from predicted classes and gold labels.

        Args:
            y_tilde (Tensor): 1D tensor containing the predicted classes for each predictions
                in the batch. This tensor should be computed through `get_predicted_classes(y_hat)` method.
            y (Tensor): gold labels. Note that y_tilde an y must have the same shape.

        Returns:
            float: the mean of correct answers.


        Examples::

            >>> y       = torch.tensor([0, 1, 4, 2, 1, 3, 2, 1, 1, 3])
            >>> y_tilde = torch.tensor([0, 1, 2, 2, 1, 3, 2, 4, 4, 3])
            >>> solver.get_accuracy(y_tilde, y)
                0.7

        """
        assert y_tilde.shape == y.shape, "predicted classes and gold labels should have the same shape"
        correct = (y_tilde == y).astype(float)  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def fit(self, train_iterator, eval_iterator, *args, epochs=10, **kwargs):
        r"""Train and evaluate a model X times. During the training, both training
        and evaluation results are saved under the `performance` attribute.

        Args:
            train_iterator (Iterator): iterator containing batch samples of data.
            eval_iterator (Iterator): iterator containing batch samples of data.
            epochs (int): number of times the model will be trained.
            verbose (bool, optional): if ``True`` display a progress bar and metrics at each epoch.
                The default is ``True``.

        Examples::

            >>> solver = MySolver(model, criterion=criterion, optimizer=optimizer)
            >>> # Train & eval EPOCHS times
            >>> EPOCHS = 10
            >>> solver.fit(train_iterator, eval_iterator, epochs=EPOCHS, verbose=True)
                Epoch:        1/10
                Training:     100% | [==================================================]
                Evaluation:   100% | [==================================================]
                Stats Training:    | Loss: 0.349 | Acc: 84.33% | Prec.: 84.26%
                Stats Evaluation:  | Loss: 0.627 | Acc: 72.04% | Prec.: 72.22%
            >>> # ...

        """
        # By default, print a log each epoch
        verbose = True if 'verbose' not in {*kwargs} else kwargs['verbose']
        # Keep track of the best model
        best_eval_accuracy = 0
        start_time = time.time()

        # Train and evaluate the model epochs times
        for epoch in range(epochs):
            if verbose:
                print("Epoch:\t{0:3d}/{1}".format(epoch + 1, epochs))

            # Train and evaluate the model
            results_train = self.train(train_iterator, *args, **kwargs)
            results_eval = self.evaluate(eval_iterator, *args, **kwargs)
            # Update the eval dictionary by adding the results at the
            # current epoch
            append2dict(self.performance["train"],
                        results_train)
            append2dict(self.performance["eval"],
                        results_eval)

            if verbose:
                print("\t Stats Train: | " + describe_dict(results_train))
                print("\t  Stats Eval: | " + describe_dict(results_eval))
                print()
            # We copy in memory the best model
            if best_eval_accuracy < self.performance["eval"]["accuracy"][-1]:
                best_eval_accuracy = self.performance["eval"]["accuracy"][-1]
                self._update_checkpoint(epoch + 1, results_train, results_eval)

        self.performance['time'] = time.time() - start_time
