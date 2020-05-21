r"""
Elementary functions used for statistical reports.
"""

import numpy as np


def true_positive(matrix):
    r"""True positive values from a confusion matrix.

    .. math::

        TP(M) = \text{Diag}(M)


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    return np.diag(matrix)


def true_negative(matrix):
    r"""True negatives values from a confusion matrix.

    .. math::

        TN(M) = \sum_{i=0}^{C-1}{\sum_{j=0}^{C-1}{M_{i, j}}} - FN(M) + FP(M) + TP(M)


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    return np.sum(matrix) - (false_positive(matrix) + false_negative(matrix) + true_positive(matrix))


def false_positive(matrix):
    r"""False positives values from a confusion matrix.

    .. math::

        FP(M) = \sum_{i=0}^{C-1}{M_i} - \text{Diag}(M)


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    return np.sum(matrix, axis=0) - np.diag(matrix)


def false_negative(matrix):
    r"""False negatives values from a confusion matrix.

    .. math::

        FN(M) = \sum_{j=0}^{C-1}{M_j} - \text{Diag}(M)


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    return np.sum(matrix, axis=1) - np.diag(matrix)


def true_positive_rate(matrix):
    r"""True positive rate from a confusion matrix.

    .. math::

        TPR(M) = \frac{TP(M)}{TP(M) + FN(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = true_positive(matrix)
    bottom = true_positive(matrix) + false_negative(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def true_negative_rate(matrix):
    r"""True negative rate from a confusion matrix.

    .. math::

        TNR(M) = \frac{TN(M)}{TN(M) + FP(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = true_negative(matrix)
    bottom = true_negative(matrix) + false_positive(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def positive_predictive_value(matrix):
    r"""Positive predictive value from a confusion matrix.

    .. math::

        PPV(M) = \frac{TP(M)}{TP(M) + FP(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = true_positive(matrix)
    bottom = true_positive(matrix) + false_positive(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def negative_predictive_value(matrix):
    r"""Negative predictive value from a confusion matrix.

    .. math::

        NPV(M) = \frac{TN(M)}{TN(M) + FN(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = true_negative(matrix)
    bottom = true_negative(matrix) + false_negative(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def false_positive_rate(matrix):
    r"""False positive rate from a confusion matrix.

    .. math::

        FPR(M) = \frac{FP(M)}{FP(M) + FN(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = false_positive(matrix)
    bottom = false_positive(matrix) + false_negative(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def false_negative_rate(matrix):
    r"""False negative rate from a confusion matrix.

    .. math::

        FNR(M) = \frac{FN(M)}{FN(M) + TP(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = false_negative(matrix)
    bottom = true_positive(matrix) + false_negative(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def false_discovery_rate(matrix):
    r"""False discovery rate from a confusion matrix.

    .. math::

        FDR(M) = \frac{FP(M)}{FP(M) + TP(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = false_positive(matrix)
    bottom = true_positive(matrix) + false_positive(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def accuracy(matrix):
    r"""Per class accuracy from a confusion matrix.

    .. math::

        ACC(M) = \frac{TP(M) + TN(M)}{TP(M) + TN(M) + FP(M) + FN(M)}


    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.

    Returns:
        numpy.ndarray

    """
    top = true_positive(matrix) + true_negative(matrix)
    bottom = true_positive(matrix) + true_negative(matrix) + false_positive(matrix) + false_negative(matrix)
    return np.where(bottom != 0, top / bottom, 0)


def flatten_matrix(matrix, axis_label=0, axis_pred=1, map=None):
    r"""Flatten a confusion matrix to retrieve its prediction and gold labels.

    Args:
        matrix (numpy.ndarray): confusion matrix of shape :math:`(C, C)`.
        axis_label (int): axis index corresponding to the gold labels.
        axis_pred (int): axis index corresponding to the predictions.
        map (dict): dictionary to map indices to label.

    Returns:
        gold labels and predictions.

    """
    gold_labels = []
    predictions = []
    # Change the index order ?
    matrix = np.array(matrix)
    if axis_label != 0 or axis_pred != 1:
        matrix = matrix.T
    # Make sure the matrix is a confusion matrix
    C = len(matrix)
    map = {idx: idx for idx in range(C)} if map is None else map
    assert matrix.shape == (C, C), 'the provided matrix is not square'
    for i in range(C):
        for j in range(C):
            gold_labels.extend([map[i]] * int(matrix[i, j]))
            predictions.extend([map[j]] * int(matrix[i, j]))

    return gold_labels, predictions
