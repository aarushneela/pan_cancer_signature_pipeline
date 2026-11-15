"""
Multi-class extension — instead of tumor vs. normal, predict WHICH cancer
type a tumor sample is (BRCA vs. LUAD vs. COAD), using only tumor samples,
shrunk down to <=5 genes the same way as the main pipeline.

Real-world framing: this is a small-scale version of a real clinical
problem called "Cancer of Unknown Primary" (CUP) — about 3-5% of cancer
diagnoses find cancer cells without knowing which organ they came from.
Gene-expression tissue-of-origin tests (e.g. the commercial "CancerTYPE ID"
test) already exist for exactly this. This script asks whether a small,
interpretable gene panel can do the same, on a 3-cancer-type scale.

Note: this uses RandomForestClassifier specifically for the RFE shrinking
step, since its feature_importances_ has a clean, unambiguous shape for
multi-class problems (unlike logistic regression's per-class coefficients).
"""

import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_expression_matrix, load_sample_metadata
from src.feature_selection import _pipeline_importance_getter
from sklearn.feature_selection import RFE

TCGA_COHORTS = {
    "BRCA": "data/raw/tcga_brca",
    "LUAD": "data/raw/tcga_luad",
    "COAD": "data/raw/tcga_coad",
}


def load_tumor_only_combined():
    """Load all 3 TCGA cohorts, keeping only tumor samples (mirrors the
    real clinical question: 'given this is cancer, which type is it?')."""
    exprs, metas = [], []
    for name, folder in TCGA_COHORTS.items():
        expr = load_expression_matrix(f"{folder}/expression_matrix.tsv")
        meta = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        tumor_ids = meta.index[meta["label"] == "tumor"]
        tumor_ids = expr.index.intersection(tumor_ids)
        exprs.append(expr.loc[tumor_ids])
        metas.append(meta.loc[tumor_ids])
    return pd.concat(exprs, axis=0), pd.concat(metas, axis=0)


def kruskal_wallis_filter(expr: pd.DataFrame, cancer_types: pd.Series, top_n: int = 300) -> list[str]:
    """Rank genes by how well they separate the 3 cancer-type groups
    (Kruskal-Wallis H-test, the multi-group analogue of Mann-Whitney U used
    elsewhere in this pipeline), and keep the top_n most significant —
    makes the downstream RFE step tractable by starting from a few hundred
    genes instead of all ~60,000.
    """
    groups = [expr.loc[cancer_types == ct] for ct in cancer_types.unique()]
    h_stats = {}
    for gene in expr.columns:
        try:
            h, p = kruskal(*(g[gene].dropna() for g in groups))
            h_stats[gene] = h
        except ValueError:
            h_stats[gene] = 0.0  # e.g. constant values in a group
    ranked = sorted(h_stats, key=h_stats.get, reverse=True)
    return ranked[:top_n]


def main():
    print("Loading tumor-only samples across BRCA, LUAD, COAD...")
    expr, meta = load_tumor_only_combined()
    print(f"{expr.shape[0]} tumor samples x {expr.shape[1]} genes")
    print(meta["cancer_type"].value_counts(), "\n")

    print("Ranking genes by Kruskal-Wallis test across the 3 cancer types (this takes a few minutes)...")
    top_genes = kruskal_wallis_filter(expr, meta["cancer_type"], top_n=300)
    print(f"Kept top {len(top_genes)} candidate genes.\n")

    X = expr[top_genes]
    y = meta["cancer_type"].to_numpy()
    groups = meta["patient_id"] if "patient_id" in meta.columns else None

    estimator = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)),
    ])

    print("Shrinking from 300 genes down to a handful via RFE (this also takes a few minutes)...")
    current_genes = list(X.columns)
    records = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    while len(current_genes) >= 1:
        X_sub = X[current_genes]
        preds = cross_val_predict(estimator, X_sub, y, cv=cv, groups=groups, n_jobs=-1)
        acc = (preds == y).mean()
        records.append({"n_genes": len(current_genes), "genes": list(current_genes), "accuracy": acc})

        if len(current_genes) <= 5:
            if len(current_genes) == 1:
                break
            n_to_select = len(current_genes) - 1
        else:
            n_to_select = max(len(current_genes) - 20, 5)  # coarse steps down to 5, then fine-grained

        rfe = RFE(estimator, n_features_to_select=n_to_select, importance_getter=_pipeline_importance_getter)
        rfe.fit(X_sub, y)
        current_genes = [g for g, keep in zip(current_genes, rfe.support_) if keep]

    results_df = pd.DataFrame(records).sort_values("n_genes", ascending=False).reset_index(drop=True)
    print("\nAccuracy vs. number of genes:")
    print(results_df[["n_genes", "accuracy"]].round(3))

    final_row = results_df[results_df["n_genes"] <= 5].iloc[0]
    final_genes = final_row["genes"]
    print(f"\nFinal minimal cancer-TYPE signature ({len(final_genes)} genes): {final_genes}")
    print(f"Cross-validated accuracy: {final_row['accuracy']:.3f}")

    print("\nDetailed classification report (5-fold CV predictions, final gene set):")
    final_preds = cross_val_predict(estimator, X[final_genes], y, cv=cv, groups=groups, n_jobs=-1)
    print(classification_report(y, final_preds))

    print("Confusion matrix (rows = true type, columns = predicted type):")
    labels = sorted(set(y))
    cm = confusion_matrix(y, final_preds, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))


if __name__ == "__main__":
    main()
