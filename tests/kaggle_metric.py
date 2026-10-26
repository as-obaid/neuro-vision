# Vendored verbatim from the official Kaggle metric for rsna-intracranial-aneurysm-detection.
# Source: https://www.kaggle.com/code/metric/mean-weighted-columnwise-aucroc
# Kernel version: 5 of 5 (owner metric, "Kaggle Competition Metrics")
# Retrieved: 2026-10-26
# sha256 of the pulled notebook (mean-weighted-columnwise-aucroc.ipynb): 69896e53885f769cad084eeb4e028ca52d49f3cd5cedb822530573bdf3efd8c6
# Licence stated on the page: Apache 2.0
# Reference only: the metric logic below is unchanged; only these header lines were added.
# ruff: noqa
# fmt: off
"""Calculates the weighted macro-average AUCROC score.

This metric generalizes the `roc_auc_score` with `average='macro'` by allowing user-defined class weights.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    class_weights: Optional[List[float]] = None,
) -> float:
    """
    Calculates the weighted macro-average AUCROC score.

    This metric generalizes the `roc_auc_score` with `average='macro'` by allowing user-defined class weights.

    Parameters
    ----------
    solution : pd.DataFrame
        DataFrame containing the true labels.
    submission : pd.DataFrame
        DataFrame containing the predicted scores.
    row_id_column_name : str
        The name of the column containing the row IDs.
    class_weights : Optional[List[float]], default=None
        A list of weights for each class. If None, all classes are weighted equally.

    Returns
    -------
    float
        The weighted multi-label AUC score.

    Examples
    --------
    >>> # Test for perfect predictions.
    >>> solution = pd.DataFrame({'id': [1, 2, 3], 'cat': [1, 0, 1], 'dog': [0, 1, 1]})
    >>> submission = pd.DataFrame({'id': [1, 2, 3], 'cat': [0.9, 0.2, 0.8], 'dog': [0.1, 0.7, 0.6]})
    >>> score(solution.copy(), submission.copy(), 'id')
    1.0

    >>> score(solution.copy(), submission.copy(), 'id', class_weights=[0.25, 0.75])
    1.0

    >>> # Test weighting.
    >>> solution = pd.DataFrame({'id': [1, 2, 3, 4], 'A': [0, 0, 1, 1], 'B': [0, 0, 1, 1]})
    >>> submission = pd.DataFrame({'id': [1, 2, 3, 4], 'A': [0.1, 0.2, 0.8, 0.9], 'B': [0.8, 0.2, 0.1, 0.9]})
    >>> # Here, the AUC for class 'A' is 1.0 and for class 'B' is 0.5.
    >>> # The unweighted macro-average is (1.0 + 0.5) / 2 = 0.75.
    >>> score(solution.copy(), submission.copy(), 'id')
    0.75

    >>> # Using weights to prioritize the better-performing class 'A'.
    >>> # The weighted average is (1.0 * 0.75) + (0.5 * 0.25) = 0.875.
    >>> score(solution.copy(), submission.copy(), 'id', class_weights=[3, 1])
    0.875

    >>> # Using weights to prioritize the worse-performing class 'B'.
    >>> # The weighted average is (1.0 * 0.25) + (0.5 * 0.75) = 0.625.
    >>> score(solution.copy(), submission.copy(), 'id', class_weights=[1, 3])
    0.625
    """
    # Per competition requirements, we don't use the row_id_column
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Validate that all submission columns are numeric
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be numeric.')

    # Validate that the number of columns match
    if len(solution.columns) != len(submission.columns):
        raise ParticipantVisibleError(
            'Submission must have predictions for every class.'
        )

    return float(
        weighted_multilabel_auc(solution.values, submission.values, class_weights)
    )


def weighted_multilabel_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_weights: Optional[List[float]] = None,
) -> float:
    """Compute weighted AUC for multilabel classification.

    Parameters:
    -----------
    y_true : np.ndarray of shape (n_samples, n_classes)
        True binary labels (0 or 1) for each class
    y_scores : np.ndarray of shape (n_samples, n_classes)
        Target scores (probability estimates or decision values)
    class_weights : array-like of shape (n_classes,), optional
        Weights for each class. If None, uniform weights are used.
        Weights will be normalized to sum to 1.

    Returns:
    --------
    weighted_auc : float
        The weighted average AUC

    Raises:
    -------
    ValueError
        If any class does not have both positive and negative samples
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]

    # Get AUC for each class
    try:
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
    except ValueError:
        raise ParticipantVisibleError(
            'AUC could not be calculated from given predictions.'
        ) from None

    # Handle weights
    if class_weights is None:  # Uniform weights
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)

    # Check weight dimensions
    if len(weights_array) != n_classes:
        raise ValueError(
            f'Number of weights ({len(weights_array)}) must match '
            f'number of classes ({n_classes})'
        )

    # Check for non-negative weights
    if np.any(weights_array < 0):
        raise ValueError('All class weights must be non-negative')

    # Check that at least one weight is positive
    if np.sum(weights_array) == 0:
        raise ValueError('At least one class weight must be positive')

    # Normalize weights to sum to 1
    weights_array = weights_array / np.sum(weights_array)

    # Compute weighted average
    return np.sum(individual_aucs * weights_array)
