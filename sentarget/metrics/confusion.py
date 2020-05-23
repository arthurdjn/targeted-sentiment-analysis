r"""
Defines a ```ConfusionMatrix```, used to compute scores (True Positive, False Negative etc.).

.. image:: images/confusion_matrix.png


Example:

.. code-block:: python

    # Create a confusion matrix
    confusion = ConfusionMatrix(num_classes=10)

    # Update the confusion matrix with a list of predictions and labels
    confusion.update(gold_labels, predictions)

    # Get the global accuracy, precision, scores from attributes or methods
    confusion.accuracy()

"""

import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from .functional import *

try:
    import seaborn as sns
except ModuleNotFoundError:
    print('WARNING: Seaborn is not installed. Plotting confusion matrices is unavailable.')


class ConfusionMatrix:
    r"""A ```ConfusionMatrix``` is a matrix of shape :math:`(C, C)`, used to index predictions :math:`p \in C`
    regarding their gold labels (or truth labels).

    """

    def __init__(self, labels=None, data=None, names=None, axis_label=0, axis_pred=1):
        assert labels is not None or data is not None, 'Failed to initialize a confusion matrix. Please provide ' \
                                                       'the number of classes `num_classes` or a starting ' \
                                                       'data `data`.'
        # General attributes
        self.num_classes = len(labels) if labels is not None else len(data)
        self.matrix = np.zeros((self.num_classes, self.num_classes)) if data is None else np.array(data)
        self.labels = list(range(self.num_classes)) if labels is None else labels
        self.names = names

        # map from labels indices to confusion matrix's indices
        self.label2idx = {label: i for (label, i) in zip(self.labels, np.arange(self.num_classes))}
        self.idx2label = {i: label for (label, i) in zip(self.labels, np.arange(self.num_classes))}

        self.predictions, self.gold_labels = ([], []) \
            if data is None else self.flatten(axis_label=axis_label, axis_pred=axis_pred, map=self.idx2label)

    def _init_labels(self, num_classes, ignore_index):
        labels = list(range((num_classes)))
        if isinstance(ignore_index, list):
            for idx in ignore_index:
                labels.pop(idx)
        return labels

    @property
    def tp(self):
        return true_positive(self.matrix)

    @property
    def tn(self):
        return true_negative(self.matrix)

    @property
    def fp(self):
        return false_positive(self.matrix)

    @property
    def fn(self):
        return false_negative(self.matrix)

    @property
    def tpr(self):
        return true_positive_rate(self.matrix)

    @property
    def tnr(self):
        return true_negative_rate(self.matrix)

    @property
    def ppv(self):
        return positive_predictive_value(self.matrix)

    @property
    def npv(self):
        return negative_predictive_value(self.matrix)

    @property
    def fpr(self):
        return false_positive_rate(self.matrix)
    @property
    def fnr(self):
        return false_negative_rate(self.matrix)

    @property
    def fdr(self):
        return false_discovery_rate(self.matrix)

    @property
    def acc(self):
        return np.diag(self.matrix) / self.matrix.sum()

    def precision_score(self, average='macro', zero_division=0, **kwargs):
        return precision_score(self.gold_labels, self.predictions, average=average, **kwargs)

    def recall_score(self, average='macro', zero_division=0, **kwargs):
        return recall_score(self.gold_labels, self.predictions, average=average, **kwargs)

    def f1_score(self, average='macro', zero_division=0, **kwargs):
        return f1_score(self.gold_labels, self.predictions, average=average, **kwargs)

    def accuracy_score(self, **kwargs):
        return accuracy_score(self.gold_labels, self.predictions, **kwargs)

    def update(self, gold_labels, predictions):
        r"""Update the confusion matrix from a list of predictions, with their respective gold labels.

        Args:
            gold_labels (list): a list of predictions.
            predictions (list): respective gold labels (or truth labels)

        """
        # Make sure the inputs are 1D arrays
        gold_labels = np.array(gold_labels).reshape(-1)
        predictions = np.array(predictions).reshape(-1)
        self.gold_labels.extend(gold_labels)
        self.predictions.extend(predictions)
        # Complete the confusion matrix
        for i, p in enumerate(predictions):
            # Ignore unknown predictions / labels / pad index etc.
            if gold_labels[i] in self.labels and predictions[i] in self.labels:
                actual = self.label2idx[gold_labels[i]]
                pred = self.label2idx[predictions[i]]
                self.matrix[actual, pred] += 1

    def to_dataframe(self, names=None, normalize=False):
        r"""Convert the ``ConfusionMatrix`` to a `DataFrame`.

        Args:
            names (list): list containing the ordered names of the indices used as gold labels.
            normalize (bool): if ``True``, normalize the ``matrix``.

        Returns:
            pandas.DataFrame

        """
        names = names or self.names
        matrix = self.normalize() if normalize else self.matrix
        return pd.DataFrame(matrix, index=names, columns=names)

    def to_dict(self):
        r"""Convert the ``ConfusionMatrix`` to a `dict`.

        * :attr:`global accuracy` (float): accuracy obtained on all classes.

        * :attr:`sensitivity` (float): sensitivity obtained on all classes.

        * :attr:`precision` (float): precision obtained on all classes.

        * :attr:`specificity` (float): specificity obtained on all classes.

        * :attr:`confusion` (list): confusion matrix obtained on all classes.

        Returns:
            dict

        """
        return {'score': float(self.accuracy_score()),
                'precision': float(self.precision_score()),
                'recall': float(self.recall_score()),
                'f1_score': float(self.f1_score()),
                'confusion': self.matrix.tolist()}

    def normalize(self):
        r"""Nomalize the confusion ``matrix``.

        .. math::

            \text{Norm}(Confusion) = \frac{Confusion}{sum(Confusion)}

        .. note::

            The operation is not inplace, and thus does not modify the attribute ```matrix```.


        Returns:
            numpy.ndarray: normalized confusion matrix.

        """
        top = self.matrix
        bottom = self.matrix.sum(axis=1)[:, np.newaxis]
        return np.divide(top, bottom, out=np.zeros_like(top), where=bottom != 0)

    def zeros(self):
        r"""Zeros the ```matrix```. Can be used to empty memory without removing the object.

        Returns:
            None. Inplace operation.

        """
        self.matrix = np.zeros_like(self.matrix)

    def flatten(self, *args, **kwargs):
        r"""Flatten a confusion matrix to retrieve its prediction and gold labels.

        """
        return flatten_matrix(self.matrix, *args, **kwargs)

    def plot(self, names=None, normalize=False, cmap='Blues', cbar=True, **kwargs):
        r"""Plot the ``matrix`` in a new figure.

        .. warning::

            `plot` is compatible with matplotlib 3.1.1.
            If you are using an older version, the display may change (version < 3.0).


        Args:
            names (list): list of ordered names corresponding to the indices used as gold labels.
            normalize (bool): if ``True`` normalize the ``matrix``.
            cmap (string or matplotlib.pyplot.cmap): heat map colors.
            cbar (bool): if ``True``, display the colorbar associated to the heat map plot.

        Returns:
            matplotlib.Axes: axes corresponding to the plot.

        """
        # Convert the matrix in dataframe to be compatible with Seaborn
        df = self.to_dataframe(names=names, normalize=normalize)
        # Plot a heat map
        ax = sns.heatmap(df, annot=True, cmap=cmap, cbar=cbar, **kwargs)
        # Correct some bugs in the latest matplotlib version (3.1.1)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        # Display correctly the labels
        ax.set_yticklabels(rotation=0, labels=names)
        ax.set_ylabel("Actual")
        ax.set_xticklabels(rotation=90, labels=names)
        ax.set_xlabel("Predicted")
        return ax
