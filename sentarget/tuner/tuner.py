r"""
Hyperparameters optimization using a grid search algorithm.
Basically, you need to provide a set of parameters that will be modified.
The grid search will run on all permutations from the set of parameters provided.
Usually, you modify the hyperparameters and models' modules (ex, dropout etc.).
In addition, if you are using custom losses or optimizer that needs additional arguments / parameters,
you can provide them through the specific dictionaries (see the documentation of ``Tuner``).


Examples:

.. code-block:: python

    # Hyper parameters to tune
    params_hyper = {
                        'epochs': [150],
                        'lr': np.arange(0.001, 0.3, 0.01).tolist(),     # Make sure to convert it to a list (for saving after)
                    }

    # Parameters affecting the models
    params_model = {
                        'model': [BiLSTM]
                        'hidden_dim': [100, 150, 200, 250],      # Model attribute
                        'n_layers': [1, 2, 3],                   # Model attribute
                        'bidirectional': [False, True],          # Model attribute
                        'LSTM.dropout': [0.2, 0.3, 0.4, 0.6],    # Modify all LSTM dropout
                        # ...
                    }

    params_loss = {
                        'criterion': [CrossEntropyLoss]
                    }

    params_optim = {
                        'criterion': [Adam]
                    }

    tuner = Tuner(params_hyper, params_loss=params_loss, params_optim=params_optim)

    # Grid Search
    tuner.fit(train_iterator, eval_iterator, verbose=True)

"""

import copy
import json
import os
from pathlib import Path
import torch

from sentarget.nn.models import BiLSTM
from .functional import tune, tune_optimizer, init_cls
from sentarget.utils import describe_dict, serialize_dict, permutation_dict


class Tuner:
    r"""
    The ``Tuner`` class is used for hyper parameters tuning.
    From a set of models and parameters to tune, this class will look at the best model's performance.

    .. note::

        To facilitate the search and hyperameters tuning, it is recommended to use the
        ``sentarget.nn.models.Model`` abstract class as parent class for all of your models.


    * :attr:`hyper_params` (dict): dictionary of hyperparameters to tune.

    * :attr:`params_model` (dict): dictionary of model's parameters to tune.

    * :attr:`params_loss` (dict): dictionary of loss's parameters to tune.

    * :attr:`params_optim` (dict): dictionary of optimizer's parameters to tune.

    * :attr:`options` (dict): dictionary of general options.

    * :attr:`performance` (dict): dictionary of all models' performances.

    """

    def __init__(self, params_hyper=None, params_model=None, params_loss=None, params_optim=None,
                 options=None):
        # Hyper parameters with default values
        self.params_hyper = params_hyper if params_model is not None else {}
        self.params_model = params_model if params_model is not None else {}
        self.params_loss = params_loss if params_loss is not None else {}
        self.params_optim = params_optim if params_optim is not None else {}
        # General options
        self.options = {**self._init_options(), **options} if options is not None else self._init_options()
        # Keep track of all performances
        self.results = []
        self._log = None
        self._log_conf = None
        self._log_perf = None
        self.best_model = None

    def _init_options(self):
        options = {
            'saves': True,
            'dirsaves': '.saves',
            'compare_on': 'accuracy',
            'verbose': True,
        }
        return options

    def _init_hyper(self):
        params_hyper = {
            'batch_size': 64,
            'epochs': 100
        }
        return params_hyper

    def reset(self):
        r"""Reset all parameters to their default values."""
        self.results = []
        self._log = None
        self._log_conf = None
        self._log_perf = None
        self.best_model = None

    def fit(self, train_data, eval_data, **kwargs):
        r"""Run the hyper parameters tuning.

        Args:
            train_data (iterator): training dataset.
            eval_data (iterator): dev dataset.


        Examples::

            >>> from sentarget.tuner import Tuner
            >>> from sentarget.nn.models.lstm import BiLSTM
            >>> from sentarget.nn.models.gru import BiGRU

            >>> # Hyper parameters to tune
            >>> tuner = Tuner(
            ...                 params_hyper={
            ...                     'epochs': [2, 3],
            ...                     'lr': [0.01],
            ...                     'vectors': 'model.txt'
            ...                 }
            ...                 params_model={
            ...                     'model': [BiLSTM],
            ...                 }
            ...                 params_loss={
            ...                     'criterion': [torch.nn.CrossEntropyLoss],
            ...                     'ignore_index': 0
            ...                 }
            ...                 params_optim={
            ...                     'optimizer': [torch.optim.Adam]
            ...                 }
            ... )
            >>> # train_iterator = torchtext data iterato
            >>> tuner.fit(train_iterator, valid_iterator)

        """
        # Update the options dictionary
        self.options = {**self.options, **kwargs}
        dirsaves = self.options['dirsaves']
        saves = self.options['saves']
        compare_on = self.options['compare_on']
        verbose = self.options['verbose']
        # All combinations of parameters, for the grid search
        configs_hyper = permutation_dict(self.params_hyper)
        configs_model = permutation_dict(self.params_model)
        configs_loss = permutation_dict(self.params_loss)
        configs_optim = permutation_dict(self.params_optim)

        self._log = self.log_init(len(configs_hyper), len(configs_model), len(configs_loss), len(configs_optim))
        if verbose:
            print(self._log)

        num_search = 0
        for config_hyper in configs_hyper:
            for config_model in configs_model:
                for config_loss in configs_loss:
                    for config_optim in configs_optim:
                        num_search += 1
                        # Set a batch size to the data
                        train_data.batch_size = config_hyper['batch_size']
                        eval_data.batch_size = config_hyper['batch_size']
                        # Initialize the model from arguments that are in config_model, and tune it if necessary
                        model = init_cls(config_model['model'], config_model)
                        tune(model, config_model)
                        modelname = model.__class__.__name__
                        # Load the criterion and optimizer, with their parameters
                        criterion = init_cls(config_loss['criterion'], config_loss)
                        optimizer = init_cls(config_optim['optimizer'], {'params': model.parameters(), **config_optim})
                        tune_optimizer(optimizer, config_hyper)

                        # Update the configuration log
                        self._log_conf = f"Search n°{num_search}: {modelname}\n"
                        self._log_conf += self.log_conf(config_hyper=config_hyper,
                                                        config_model=config_model,
                                                        config_loss=config_loss,
                                                        config_optim=config_optim)
                        self._log_conf += f"\n{model.__repr__()}"
                        self._log += f"\n\n{self._log_conf}"
                        if verbose:
                            print(f"\n{self._log_conf}")

                        # Train the model
                        best_model = model.fit(train_data, eval_data,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               epochs=config_hyper['epochs'],
                                               verbose=False,
                                               compare_on=compare_on)
                        results = {
                            'performance': model.performance,
                            'hyper': config_hyper,
                            'model': config_model,
                            'optimizer': self.params_optim,
                            'criterion': self.params_loss
                        }
                        self.results.append(serialize_dict(results))

                        # Update the current best model
                        if (self.best_model is None or
                                best_model.performance['eval'][compare_on] > self.best_model.performance['eval'][compare_on]):
                            self.best_model = copy.deepcopy(best_model)

                        # Update the current performance log
                        self._log_perf = model.log_perf()
                        self._log += "\n" + self._log_perf
                        if verbose:
                            print(self._log_perf)

                        # Save the current checkpoint
                        if saves:
                            dirname = os.path.join(dirsaves, 'gridsearch', f"search_{num_search}")
                            filename = f"model_{modelname}.pt"
                            model.save(filename=os.path.join(dirname, filename), checkpoint=False)
                            filename = f"best_{modelname}.pt"
                            best_model.save(filename=os.path.join(dirname, filename), checkpoint=True)
                            # Save the associated log
                            self._save_current_results(os.path.join(dirname, 'results.json'))
                            self._save_current_log(os.path.join(dirname, 'log.txt'))

        if saves:
            self.save(dirsaves=dirsaves)

    def log_init(self, hyper, model, loss, optim):
        """Generate a general configuration log.

        Args:
            hyper (int): number of hyper parameters permutations.
            model (int): number of model parameters permutations.
            loss (int): number of loss parameters permutations.
            optim (int): number of optimizer parameters permutations.

        Returns:
            string: general log.

        """
        log = "GridSearch(\n"
        log += f"  (options): Parameters({describe_dict(self.options, )})\n"
        log += f"  (session): Permutations(hyper={hyper}, model={model}, loss={loss}, optim={optim}, total={hyper * model * loss * optim})\n"
        log += ")"
        return log

    def log_conf(self, config_hyper={}, config_model={}, config_loss={}, config_optim={}, **kwargs):
        """Generate a configuration log from the generated set of configurations files.

        Args:
            config_hyper (dict): hyper parameters configuration file.
            config_model (dict): model parameters configuration file.
            config_loss (dict): loss parameters configuration file.
            config_optim (dict): optimizer parameters configuration file.

        Returns:
            string: configuration file representation.

        """
        log = f"Configuration(\n"
        log += f"  (hyper): Variables({describe_dict(config_hyper, **kwargs)})\n"
        log += f"  (model): Parameters({describe_dict(config_model, **kwargs)})\n"
        log += f"  (criterion): {config_loss['criterion'].__name__}({describe_dict(config_loss, **kwargs)})\n"
        log += f"  (optimizer): {config_optim['optimizer'].__name__}({describe_dict(config_optim, **kwargs)})\n"
        log += ')'
        return log

    def _save_current_results(self, filename='results.json'):
        # Create the directory if it does not exists
        dirname = os.path.dirname(filename)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as outfile:
            json.dump(serialize_dict(self.results[-1]), outfile)

    def _save_current_log(self, filename='log.txt'):
        # Create the directory if it does not exists
        dirname = os.path.dirname(filename)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as outfile:
            outfile.write(self._log_conf + "\n" + self._log_perf)

    def save_log(self, filename='log.txt'):
        # Create the directory if it does not exists
        dirname = os.path.dirname(filename)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as outfile:
            outfile.write(self._log)

    def save_results(self, filename='results.json'):
        # Create the directory if it does not exists
        dirname = os.path.dirname(filename)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        data = {'results': self.results}
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def save(self, dirsaves=None, checkpoint=True):
        r"""Save the performances as a json file, by default.

        Args:
            dirsaves (string): name of saving directory.
            checkpoint (bool): if ``True``, saves the best model's checkpoint.

        """
        dirsaves = self.options['dirsaves'] if dirsaves is None else dirsaves
        self.save_log(os.path.join(dirsaves, 'log_gridsearch.txt'))
        self.save_results(os.path.join(dirsaves, 'results_gridsearch.json'))
        # Saving the best model
        filename = f"best_{self.best_model.__class__.__name__}.pt"
        self.best_model.save(filename=os.path.join(dirsaves, filename), checkpoint=checkpoint)
        # And its log / performances
        self._save_current_results(os.path.join(dirsaves, 'best_results.json'))
        self._save_current_log(os.path.join(dirsaves, 'best_log.txt'))
