"""
Stage 6 — Independent validation on a held-out external dataset.

Loads a separate study (different platform, different lab — e.g. one of the
GEO series listed in README.md) and evaluates the ALREADY-TRAINED final model
on it, with no retraining and no leakage from the external set back into
training. This is the strongest test of real-world generalization: TCGA
internal cross-validation can still share subtle processing-pipeline
artifacts across its own folds, but an independent GEO series on a different
array platform will not.

Key preprocessing subtlety: you cannot apply the *training* ComBat batch
correction to new external data (ComBat needs to see all batches together at
fit time, and refitting it on train+external together would leak the
external set's distribution into what should be a blind evaluation).
Instead, per-gene z-scoring of the external set against its own mean/SD
(implemented below) puts it on a comparable *relative* scale to the training
data's per-gene z-scores without ever looking at the training data's values
directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import load_expression_matrix, load_sample_metadata, log_transform


def load_independent_dataset(
    expr_path: str,
    meta_path: str,
    signature_genes: list[str],
    already_log_transformed: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load an external validation dataset and restrict it to the signature genes.

    Args:
        expr_path: # FILL IN: path to the external expression matrix, e.g.
            "data/raw/geo_validation/gse32863_luad/expression_matrix.tsv"
        meta_path: # FILL IN: path to the external sample metadata (needs
            'label' at minimum; 'cancer_type'/'study' if available).
        signature_genes: the final minimal gene list the model was trained on.
        already_log_transformed: set True if this platform's data is already
            log-scale (typical for GEO series matrix files) to avoid
            double-log-transforming.

    Returns:
        (X_external, y_external) — X restricted + ordered to `signature_genes`
        (missing genes raise a clear error rather than silently zero-filling),
        y as a 0/1 array (1 = tumor).
    """
    expr = load_expression_matrix(expr_path)
    meta = load_sample_metadata(meta_path)
    expr = expr.loc[expr.index.intersection(meta.index)]
    meta = meta.loc[expr.index]

    missing = [g for g in signature_genes if g not in expr.columns]
    if missing:
        raise ValueError(
            f"External dataset is missing signature gene(s) {missing}. "
            "Check gene symbol conventions match (e.g. HGNC symbol vs. probe "
            "ID vs. Ensembl ID) before proceeding — a silent zero-fill here "
            "would fabricate a result."
        )

    expr = expr if already_log_transformed else log_transform(expr)
    X_external = expr[signature_genes]
    y_external = (meta["label"] == "tumor").astype(int)
    return X_external, y_external


def rescale_external_to_training_distribution(
    X_external: pd.DataFrame,
    training_gene_means: pd.Series,
    training_gene_stds: pd.Series,
) -> pd.DataFrame:
    """Put an external dataset's signature genes on the training set's scale.

    Rather than re-running ComBat (which would require access to the raw
    training data at validation time and risks leakage), this z-scores the
    external data against ITS OWN mean/SD, then rescales to match the
    training set's mean/SD for each gene. This preserves each gene's
    *relative* ordering within the external cohort (who's high/low tumor
    signal) while putting the absolute values on a scale the trained
    classifier's learned coefficients/thresholds expect.

    Args:
        X_external: external expression matrix, restricted to signature genes.
        training_gene_means: per-gene mean from the TRAINING data (save these
            when you fit the final model — e.g.
            `X_train[signature_genes].mean()`).
        training_gene_stds: per-gene std from the TRAINING data, matching
            `training_gene_means`.

    Returns:
        Rescaled X_external, same shape.
    """
    external_means = X_external.mean(axis=0)
    external_stds = X_external.std(axis=0).replace(0, 1.0)  # avoid divide-by-zero on constant genes

    z = (X_external - external_means) / external_stds
    return z * training_gene_stds + training_gene_means


def evaluate_on_independent(model, X_external: pd.DataFrame, y_external: pd.Series) -> dict[str, float]:
    """Evaluate an already-trained model on external data. No fitting happens here.

    Args:
        model: a FITTED sklearn Pipeline/estimator (already trained on TCGA
            data with the final minimal signature).
        X_external, y_external: from `load_independent_dataset` (optionally
            passed through `rescale_external_to_training_distribution` first).

    Returns:
        Dict with 'roc_auc', 'accuracy', 'precision', 'recall', 'f1'.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = model.predict(X_external)
    y_score = model.predict_proba(X_external)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "roc_auc": roc_auc_score(y_external, y_score) if len(set(y_external)) > 1 else np.nan,
        "accuracy": accuracy_score(y_external, y_pred),
        "precision": precision_score(y_external, y_pred, zero_division=0),
        "recall": recall_score(y_external, y_pred, zero_division=0),
        "f1": f1_score(y_external, y_pred, zero_division=0),
    }
