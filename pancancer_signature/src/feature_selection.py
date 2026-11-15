"""
Stage 4b — Stepwise feature elimination to a minimal (<=5 gene) signature.

Starting from the shared pan-cancer signature (Stage 3, likely tens of
genes), shrink the feature set one gene at a time, re-evaluating
cross-validated performance at every step, so you can plot the
accuracy/gene-count tradeoff and pick the smallest set that keeps
most of the discriminative power (the central question of this project).

Two complementary approaches are provided:
    - `recursive_feature_elimination_cv`: uses sklearn's RFE machinery with a
      linear model's coefficients to rank and drop the least-informative
      gene at each step (fast, works well with `logreg_l1`/`linear_svm`).
    - `stepwise_backward_elimination_cv`: a more transparent, model-agnostic
      loop that at every step removes whichever single gene's removal hurts
      cross-validated AUC the least (works with Random Forest too, and is
      easier to explain gene-by-gene in a write-up, at the cost of being
      O(n_genes^2) in the number of CV fits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

from .modeling import evaluate_model_cv


def _pipeline_importance_getter(fitted_pipeline: Pipeline) -> np.ndarray:
    """Pull coef_ / feature_importances_ out of a fitted Pipeline's 'clf' step
    for sklearn's RFE, which by default only checks the top-level estimator.

    Handles both binary classifiers (coef_ shape: (1, n_features)) and
    multi-class classifiers (coef_ shape: (n_classes, n_features), one row
    per class in a one-vs-rest scheme) by averaging the absolute coefficient
    magnitude across classes — a gene important for distinguishing ANY one
    class from the rest gets a high aggregate score.
    """
    clf = fitted_pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        if coef.ndim > 1 and coef.shape[0] > 1:
            return np.abs(coef).mean(axis=0)  # multi-class: average |coef| across classes
        return np.abs(coef.ravel())
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    raise ValueError(f"Estimator {clf.__class__.__name__} exposes neither coef_ nor feature_importances_")


def recursive_feature_elimination_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator: Pipeline,
    min_features: int = 1,
    step: int = 1,
    n_splits: int = 5,
    random_state: int = 0,
    groups: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Shrink the gene set via sklearn RFE, tracking CV performance per step.

    RFE fits the estimator on the full feature set, ranks features by the
    fitted model's coefficient magnitude (for linear models) or importance
    (for tree ensembles), discards the least important, and repeats. This is
    efficient because it only requires ~n_genes/step model fits total, rather
    than re-evaluating every possible subset.

    Note: RFE requires an estimator exposing `coef_` or `feature_importances_`
    after fitting a Pipeline — sklearn's RFE supports Pipelines directly as
    long as the final step exposes one of those attributes (true for
    `logreg_l1`, `logreg_elasticnet`, `linear_svm`, and `random_forest` from
    `modeling.build_classifiers`).

    Args:
        X: feature matrix restricted to the candidate signature genes.
        y: binary labels.
        estimator: Pipeline whose final step supports RFE (see note above).
        min_features: smallest gene set size to evaluate down to (use 1 to see
            the full curve; you'll pick <=5 from the resulting table).
        step: number of genes to remove per RFE iteration.
        n_splits, random_state: passed to `evaluate_model_cv` at each step.
        groups: optional per-sample group (e.g. patient_id) to prevent
            patient-level leakage across CV folds at every step — see
            `modeling.evaluate_model_cv`. Strongly recommended whenever the
            data contains matched tumor/normal pairs.

    Returns:
        DataFrame with one row per gene-set size, columns:
            'n_genes', 'genes' (list), 'roc_auc', 'accuracy', 'precision',
            'recall', 'f1' — sorted by n_genes descending (full set first).
    """
    n_genes_total = X.shape[1]
    records = []

    current_genes = list(X.columns)
    while len(current_genes) >= min_features:
        X_sub = X[current_genes]
        metrics = evaluate_model_cv(X_sub, y, estimator, n_splits=n_splits, random_state=random_state, groups=groups)
        records.append({"n_genes": len(current_genes), "genes": list(current_genes), **metrics})

        if len(current_genes) == min_features:
            break

        # Fit RFE just to get a ranking on the *current* gene subset, then
        # drop the lowest-ranked one(s). We refit RFE at each shrinking step
        # (rather than doing one RFE pass over all genes at once) so the
        # importance ranking reflects the actual remaining feature set, not
        # stale rankings computed on a larger, already-pruned-from set.
        n_to_select = max(len(current_genes) - step, min_features)
        # `importance_getter` tells RFE how to pull coefficients/importances
        # out of a Pipeline (RFE's 'auto' mode only looks for coef_ /
        # feature_importances_ on the estimator itself, not inside a
        # Pipeline's named steps).
        rfe = RFE(
            estimator,
            n_features_to_select=n_to_select,
            step=step,
            importance_getter=_pipeline_importance_getter,
        )
        rfe.fit(X_sub, y)
        current_genes = [g for g, keep in zip(current_genes, rfe.support_) if keep]

    return pd.DataFrame(records).sort_values("n_genes", ascending=False).reset_index(drop=True)


def stepwise_backward_elimination_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator: Pipeline,
    min_features: int = 1,
    n_splits: int = 5,
    random_state: int = 0,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Greedy backward elimination: at each step, drop whichever gene's
    removal costs the least cross-validated performance.

    More computationally expensive than `recursive_feature_elimination_cv`
    (evaluates every remaining gene's removal at every step) but doesn't rely
    on a linear coefficient/importance ranking, so it works identically for
    any estimator and is easy to narrate in a methods section ("at each step
    we removed the gene whose absence least reduced cross-validated AUC").

    Args:
        X, y, estimator: see `recursive_feature_elimination_cv`.
        min_features: stop once this many genes remain.
        n_splits, random_state: passed to `evaluate_model_cv`.
        metric: which metric from `evaluate_model_cv`'s output to optimize for
            at each elimination step (default 'roc_auc').

    Returns:
        Same schema as `recursive_feature_elimination_cv`.
    """
    current_genes = list(X.columns)
    records = []

    metrics = evaluate_model_cv(X[current_genes], y, estimator, n_splits=n_splits, random_state=random_state)
    records.append({"n_genes": len(current_genes), "genes": list(current_genes), **metrics})

    while len(current_genes) > min_features:
        best_score = -np.inf
        best_genes = None
        best_metrics = None
        for gene_to_drop in current_genes:
            candidate = [g for g in current_genes if g != gene_to_drop]
            m = evaluate_model_cv(X[candidate], y, estimator, n_splits=n_splits, random_state=random_state)
            if m[metric] > best_score:
                best_score, best_genes, best_metrics = m[metric], candidate, m
        current_genes = best_genes
        records.append({"n_genes": len(current_genes), "genes": list(current_genes), **best_metrics})

    return pd.DataFrame(records).sort_values("n_genes", ascending=False).reset_index(drop=True)


def plot_performance_vs_genes(results_df: pd.DataFrame, metrics: list[str] | None = None, save_path: str | None = None):
    """Plot classification performance vs. number of genes retained.

    Args:
        results_df: output of `recursive_feature_elimination_cv` or
            `stepwise_backward_elimination_cv`.
        metrics: which metric columns to plot (default: roc_auc + accuracy).
        save_path: if given, save the figure here (e.g.
            'outputs/performance_vs_genes.png') instead of just returning the
            Figure.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    metrics = metrics or ["roc_auc", "accuracy", "f1"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in metrics:
        ax.plot(results_df["n_genes"], results_df[metric], marker="o", label=metric)

    ax.set_xlabel("Number of genes in signature")
    ax.set_ylabel("Cross-validated score")
    ax.set_title("Performance vs. signature size")
    ax.axvline(5, color="gray", linestyle="--", alpha=0.6, label="5-gene target")
    ax.invert_xaxis()  # read left-to-right as "shrinking" the signature
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig