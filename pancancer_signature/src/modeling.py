"""
Stage 4a — Classifier definitions and cross-validated evaluation.

Defines the three classifiers to compare (regularized Logistic Regression,
Random Forest, linear SVM) and a shared evaluation routine that reports
AUC-ROC, accuracy, precision, recall, and F1 via stratified k-fold CV — the
appropriate CV scheme for small, potentially class-imbalanced sample sizes
(preserves the tumor:normal ratio in every fold, unlike plain KFold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_classifiers(random_state: int = 0) -> dict[str, Pipeline]:
    """Construct the three candidate classifiers, each wrapped with scaling.

    Every classifier is wrapped in a Pipeline with `StandardScaler` first:
    Logistic Regression and linear SVM are both sensitive to feature scale
    (their regularization penalty is applied to the raw coefficient
    magnitudes, which only makes sense if features are on comparable scales);
    Random Forest doesn't need scaling but including it doesn't hurt and
    keeps the interface uniform.

    Args:
        random_state: shared seed for reproducibility.

    Returns:
        Dict of name -> sklearn Pipeline (unfit).
    """
    return {
        "logreg_l1": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",  # supports l1 penalty
                        C=1.0,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logreg_elasticnet": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",  # only solver supporting elasticnet
                        l1_ratio=0.5,
                        C=1.0,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scale", StandardScaler()),  # no-op for trees, kept for interface consistency
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,  # guards against overfitting single-sample leaves
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        probability=True,  # needed for AUC-ROC scoring
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_model_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator: Pipeline,
    n_splits: int = 5,
    random_state: int = 0,
    groups: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Evaluate one estimator with stratified k-fold CV.

    With small sample sizes (common for matched tumor/normal cohorts), a
    single train/test split gives a noisy, easily-misleading performance
    estimate. k-fold CV averages over k different held-out folds, and
    stratification keeps the tumor:normal ratio consistent across folds so a
    lucky/unlucky fold split doesn't dominate the result. 5 folds is a
    reasonable default down to roughly n=30-40 per class; drop to 3 folds
    below that so each fold still has enough samples per class.

    IMPORTANT — patient leakage: if a dataset contains matched tumor/normal
    pairs (the same patient contributing both a tumor and a normal sample),
    plain StratifiedKFold can split one patient's two samples across
    different folds. Since a patient's tumor and normal tissue share that
    person's genetic background, the model can partly learn to recognize
    the *patient*, not just tumor-vs-normal biology in general — inflating
    reported performance. Passing `groups` (e.g. patient_id) switches to
    StratifiedGroupKFold, which keeps every sample from the same patient
    together in one fold, entirely in train or entirely in test, closing
    this leakage path. Always pass `groups` when your data has matched
    pairs or any other repeated-patient structure.

    Args:
        X: feature matrix (samples x genes), already restricted to the
            signature genes being evaluated.
        y: binary labels (1 = tumor, 0 = normal), same order as X.
        estimator: an unfit sklearn Pipeline/estimator (e.g. from
            `build_classifiers`).
        n_splits: number of CV folds.
        random_state: for reproducible fold assignment.
        groups: optional per-sample group label (typically patient_id) to
            prevent the same patient's samples from spanning both the train
            and test side of any fold. Strongly recommended whenever matched
            pairs exist in the data.

    Returns:
        Dict of mean metric across folds: 'roc_auc', 'accuracy', 'precision',
        'recall', 'f1'.
    """
    n_splits = min(n_splits, int(np.min(np.bincount(y))))  # can't have more folds than the smaller class
    n_splits = max(n_splits, 2)

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = cv.split(X, y)

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    scores = cross_validate(estimator, X, y, cv=list(cv_splits), scoring=scoring, n_jobs=-1)
    return {name: float(np.mean(scores[f"test_{name}"])) for name in scoring}


def compare_classifiers(
    X: pd.DataFrame,
    y: np.ndarray,
    classifiers: dict[str, Pipeline] | None = None,
    n_splits: int = 5,
    random_state: int = 0,
    groups: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Run `evaluate_model_cv` for each classifier and tabulate results.

    Args:
        X, y: see `evaluate_model_cv`.
        classifiers: dict of name -> unfit estimator; defaults to
            `build_classifiers()`.
        n_splits, random_state: passed through.
        groups: optional per-sample group (e.g. patient_id) to prevent
            patient-level leakage across folds — see `evaluate_model_cv`.

    Returns:
        DataFrame indexed by classifier name, columns = metrics, sorted by
        roc_auc descending.
    """
    classifiers = classifiers or build_classifiers(random_state=random_state)
    rows = {}
    for name, clf in classifiers.items():
        rows[name] = evaluate_model_cv(X, y, clf, n_splits=n_splits, random_state=random_state, groups=groups)
    return pd.DataFrame(rows).T.sort_values("roc_auc", ascending=False)
