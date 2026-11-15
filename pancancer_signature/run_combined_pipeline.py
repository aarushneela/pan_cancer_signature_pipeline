"""
The combined, final pipeline: chain the tumor/normal detector and the
cancer-type classifier into one two-stage system that answers the full
real-world question — "is this cancer, and if so, which type?"

Stage 1 (detection): the 5-gene tumor/normal panel (COL10A1, NME1, IQANK1,
IQGAP3, HJURP) decides tumor vs. normal.
Stage 2 (typing): if Stage 1 says "tumor," the 5-gene type panel (HKDC1,
GCNT3, ACSL5, ROS1, NOX1) decides which of the three cancer types it is.

Final possible outcomes per sample: Normal, BRCA, LUAD, or COAD — a genuine
4-class real-world diagnostic outcome, evaluated with proper patient-grouped
cross-validation so both stages are trained and tested on consistent folds
(no leakage between the two stages).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data_loading import load_expression_matrix, load_sample_metadata

TCGA_COHORTS = {
    "BRCA": "data/raw/tcga_brca",
    "LUAD": "data/raw/tcga_luad",
    "COAD": "data/raw/tcga_coad",
}

STAGE1_GENES = ["ENSG00000123500", "ENSG00000239672", "ENSG00000203499", "ENSG00000183856", "ENSG00000123485"]
STAGE1_NAMES = ["COL10A1", "NME1", "IQANK1", "IQGAP3", "HJURP"]

STAGE2_GENES = ["ENSG00000156510", "ENSG00000140297", "ENSG00000197142", "ENSG00000047936", "ENSG00000007952"]
STAGE2_NAMES = ["HKDC1", "GCNT3", "ACSL5", "ROS1", "NOX1"]


def load_all_samples():
    """Load every TCGA sample (tumor + normal) with both gene panels."""
    exprs, metas = [], []
    for name, folder in TCGA_COHORTS.items():
        expr = load_expression_matrix(f"{folder}/expression_matrix.tsv")
        meta = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        exprs.append(expr[STAGE1_GENES + STAGE2_GENES])
        metas.append(meta)
    return pd.concat(exprs, axis=0), pd.concat(metas, axis=0)


def make_final_label(row) -> str:
    """The true, full real-world answer for each sample."""
    return "Normal" if row["label"] == "normal" else row["cancer_type"]


def build_stage1_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000, random_state=0)),
    ])


def build_stage2_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)),
    ])


def main():
    print("Loading all TCGA samples (tumor + normal) with both gene panels...")
    expr, meta = load_all_samples()
    meta = meta.copy()
    meta["final_label"] = meta.apply(make_final_label, axis=1)
    print(meta["final_label"].value_counts(), "\n")

    y_true_binary = (meta["label"] == "tumor").astype(int).to_numpy()
    y_final = meta["final_label"].to_numpy()
    groups = meta["patient_id"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    all_preds = np.empty(len(meta), dtype=object)

    print("Running 5-fold patient-grouped cross-validation of the full 2-stage pipeline...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(expr, y_final, groups=groups), 1):
        # --- Stage 1: train tumor/normal detector on this fold's training data ---
        X1_train = expr.iloc[train_idx][STAGE1_GENES]
        y1_train = y_true_binary[train_idx]
        stage1 = build_stage1_model()
        stage1.fit(X1_train, y1_train)

        # --- Stage 2: train cancer-type classifier on this fold's TUMOR-ONLY training data ---
        tumor_train_mask = y_true_binary[train_idx] == 1
        X2_train = expr.iloc[train_idx][tumor_train_mask][STAGE2_GENES]
        y2_train = meta["cancer_type"].iloc[train_idx][tumor_train_mask].to_numpy()
        stage2 = build_stage2_model()
        stage2.fit(X2_train, y2_train)

        # --- Apply both stages to this fold's held-out test data ---
        X1_test = expr.iloc[test_idx][STAGE1_GENES]
        stage1_pred = stage1.predict(X1_test)

        fold_preds = np.array(["Normal"] * len(test_idx), dtype=object)
        tumor_test_mask = stage1_pred == 1
        if tumor_test_mask.any():
            X2_test = expr.iloc[test_idx][tumor_test_mask][STAGE2_GENES]
            stage2_pred = stage2.predict(X2_test)
            fold_preds[tumor_test_mask] = stage2_pred

        all_preds[test_idx] = fold_preds
        print(f"  Fold {fold} done.")

    overall_accuracy = accuracy_score(y_final, all_preds)
    print(f"\nFull pipeline (both stages combined) accuracy: {overall_accuracy:.4f}")

    labels = ["Normal", "BRCA", "LUAD", "COAD"]
    cm = confusion_matrix(y_final, all_preds, labels=labels)
    print("\nConfusion matrix (rows = true answer, columns = predicted answer):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print("\nPer-class accuracy:")
    for i, label in enumerate(labels):
        row_total = cm[i].sum()
        correct = cm[i][i]
        print(f"  {label}: {correct}/{row_total} = {correct / row_total:.3f}")


if __name__ == "__main__":
    main()
