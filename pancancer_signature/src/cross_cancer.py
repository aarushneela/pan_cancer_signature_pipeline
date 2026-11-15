"""
Stage 5 — Cross-cancer generalization test.

Trains the final minimal-gene model on two cancer types combined and
evaluates it on the third, held-out cancer type it never saw during
training, repeating for all three leave-one-cancer-type-out combinations.
This is the key test of whether the signature captures generic
tumor-vs-normal biology (generalizes across tissue types) versus
cancer-type-specific patterns that happen to overlap in the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


def leave_one_cancer_out(
    X: pd.DataFrame,
    y: np.ndarray,
    cancer_types: pd.Series,
    estimator: Pipeline,
) -> pd.DataFrame:
    """Train on two cancer types, test on the third, for every combination.

    Args:
        X: feature matrix restricted to the final minimal gene signature.
        y: binary labels (1 = tumor, 0 = normal), aligned to X.
        cancer_types: per-sample cancer type label ('BRCA'/'LUAD'/'COAD'),
            aligned to X.index.
        estimator: an UNFIT sklearn Pipeline (e.g. the final chosen classifier
            from `modeling.build_classifiers`); refit fresh for each holdout
            combination so no information leaks between folds.

    Returns:
        DataFrame, one row per held-out cancer type, columns: 'held_out',
        'n_train', 'n_test', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1'.
    """
    from sklearn.base import clone

    results = []
    for held_out in cancer_types.unique():
        train_mask = cancer_types != held_out
        test_mask = cancer_types == held_out

        X_train, y_train = X.loc[train_mask], y[train_mask.to_numpy()]
        X_test, y_test = X.loc[test_mask], y[test_mask.to_numpy()]

        model = clone(estimator)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # predict_proba's second column is P(class=1=tumor); needed for AUC.
        y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        results.append(
            {
                "held_out": held_out,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "roc_auc": roc_auc_score(y_test, y_score) if len(set(y_test)) > 1 else np.nan,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    return pd.DataFrame(results)
