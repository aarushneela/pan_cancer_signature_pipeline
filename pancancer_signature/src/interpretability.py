"""
Stage 7 — Interpretability of the final minimal-gene model.

For the final chosen model + gene set, extract per-gene contributions in a
form suitable for a research write-up:
    - Logistic Regression: coefficients and odds ratios (exp(coefficient)),
      i.e. "each 1-unit increase in log2 expression of gene X multiplies the
      odds of the sample being tumor by OR_X".
    - Linear SVM: coefficients (direction/magnitude of separation, no natural
      odds-ratio interpretation since SVM isn't a probabilistic model).
    - Random Forest: impurity-based feature importances (relative, not
      directional — doesn't tell you up- or down-regulation, only how useful
      the gene was for splitting).

Because every classifier in `modeling.build_classifiers` is wrapped in a
`StandardScaler` Pipeline, the extracted coefficients are in *standardized*
units (per 1-SD change in log2 expression), which is the correct and
comparable form for reporting across genes measured on different absolute
scales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract per-gene importance/coefficient/odds-ratio from a fitted Pipeline.

    Args:
        model: a FITTED Pipeline from `modeling.build_classifiers` (must have
            a 'clf' step, as constructed there).
        feature_names: gene names in the same order as the columns the model
            was fit on.

    Returns:
        DataFrame indexed by gene, with whichever of these columns apply to
        the underlying estimator type:
            'coefficient'   — standardized coefficient (linear models only)
            'odds_ratio'    — exp(coefficient) (Logistic Regression only)
            'importance'    — impurity-based importance (Random Forest only)
        Sorted by absolute effect size descending.
    """
    clf = model.named_steps.get("clf", model)  # supports a raw estimator too, not just a Pipeline

    if hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_).ravel()
        df = pd.DataFrame({"gene": feature_names, "coefficient": coefs}).set_index("gene")
        # Odds ratios are only meaningful for a model fit via logistic
        # regression's log-odds link function; SVM coefficients define a
        # separating hyperplane, not a probability model, so no OR column
        # for SVM.
        if hasattr(clf, "predict_proba") and clf.__class__.__name__.startswith("LogisticRegression"):
            df["odds_ratio"] = np.exp(df["coefficient"])
        df["abs_effect"] = df["coefficient"].abs()

    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        df = pd.DataFrame({"gene": feature_names, "importance": importances}).set_index("gene")
        df["abs_effect"] = df["importance"].abs()

    else:
        raise ValueError(f"Estimator {clf.__class__.__name__} exposes neither coef_ nor feature_importances_")

    return df.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")


def summarize_signature_direction(de_results_by_cancer: dict[str, pd.DataFrame], genes: list[str]) -> pd.DataFrame:
    """Report each final signature gene's log2FC/padj in every cancer type,
    for a compact "this gene is consistently up X-fold across all three
    cancers" table alongside the model's own coefficients/importances.

    Args:
        de_results_by_cancer: from Stage 2, cancer_type -> DE results.
        genes: final minimal signature gene list.

    Returns:
        DataFrame indexed by gene, columns are MultiIndex (cancer_type,
        'log2FC'/'padj').
    """
    frames = {}
    for cancer_type, de_results in de_results_by_cancer.items():
        frames[cancer_type] = de_results.loc[de_results.index.intersection(genes), ["log2FC", "padj"]]
    return pd.concat(frames, axis=1)
