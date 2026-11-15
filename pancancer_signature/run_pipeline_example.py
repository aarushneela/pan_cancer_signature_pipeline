"""
End-to-end smoke test of the full pipeline using SYNTHETIC data.

Run this first (`python run_pipeline_example.py`) to confirm every stage
plugs together correctly before pointing the loaders at real TCGA/GEO files.
The synthetic data is constructed to have a known ground truth so the
pipeline's behavior is easy to sanity-check:
    - 5 "true signal" genes are up-regulated in tumor across ALL THREE
      simulated cancer types (the pan-cancer signature the pipeline should
      recover).
    - 3 "proliferation marker" genes are ALSO up-regulated in tumor across all
      three cancer types, so Stage 3's proliferation filter has something
      real to catch.
    - A handful of "cancer-type-specific" genes are only up in one cancer
      type, and should NOT survive the cross-cancer intersection.
    - Thousands of pure-noise genes are added so the DE step and multiple-
      testing correction have realistic work to do.
    - A synthetic "study"/batch effect is injected (an additive shift for a
      subset of samples per cancer type, mimicking TCGA vs. an independent
      cohort) so `batch_correct` has something to remove.

Everything below is FAKE data for testing the code's mechanics only — it says
nothing about real cancer biology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cross_cancer import leave_one_cancer_out
from src.data_loading import ExpressionDataset, batch_correct, find_matched_pairs
from src.differential_expression import run_differential_expression_per_cancer
from src.feature_selection import plot_performance_vs_genes, recursive_feature_elimination_cv
from src.interpretability import get_feature_importance, summarize_signature_direction
from src.modeling import build_classifiers, compare_classifiers
from src.signature_intersection import build_pancancer_signature
from src.validation import evaluate_on_independent

RNG = np.random.default_rng(42)

CANCER_TYPES = ["BRCA", "LUAD", "COAD"]
N_PATIENTS_PER_CANCER = 40  # -> 40 tumor + 40 matched-normal samples per cancer type
N_NOISE_GENES = 2000

TRUE_SIGNAL_GENES = [f"SIGNAL_{i}" for i in range(5)]  # what the pipeline should recover
PROLIFERATION_GENES = ["MKI67", "PCNA", "TOP2A"]  # matches the biological examples in the prompt
CANCER_SPECIFIC_GENES = {
    "BRCA": ["BRCA_ONLY_1", "BRCA_ONLY_2"],
    "LUAD": ["LUAD_ONLY_1", "LUAD_ONLY_2"],
    "COAD": ["COAD_ONLY_1", "COAD_ONLY_2"],
}


def make_synthetic_dataset() -> ExpressionDataset:
    """Simulate a multi-cancer-type, matched tumor/normal expression dataset."""
    all_special_genes = (
        TRUE_SIGNAL_GENES + PROLIFERATION_GENES + [g for genes in CANCER_SPECIFIC_GENES.values() for g in genes]
    )
    noise_genes = [f"NOISE_{i}" for i in range(N_NOISE_GENES)]
    all_genes = all_special_genes + noise_genes

    rows_expr, rows_meta = [], []

    for cancer_type in CANCER_TYPES:
        for patient_idx in range(N_PATIENTS_PER_CANCER):
            patient_id = f"{cancer_type}-PATIENT-{patient_idx}"
            # Simulate a mild batch effect: the first half of patients per
            # cancer type are "study A" (like TCGA), the second half "study B"
            # (like an independent GEO cohort), with a small additive shift.
            study = "studyA" if patient_idx < N_PATIENTS_PER_CANCER // 2 else "studyB"
            batch_shift = 0.0 if study == "studyA" else 0.8

            for label in ["tumor", "normal"]:
                baseline = RNG.normal(loc=6.0, scale=1.0, size=len(all_genes))  # baseline log2 expression
                expr_values = baseline.copy()

                if label == "tumor":
                    # True pan-cancer signal: up in tumor, in every cancer type.
                    for g in TRUE_SIGNAL_GENES:
                        expr_values[all_genes.index(g)] += RNG.normal(2.5, 0.4)
                    # Proliferation markers: also up in tumor, every cancer type
                    # (this is what Stage 3's filter should catch and remove).
                    for g in PROLIFERATION_GENES:
                        expr_values[all_genes.index(g)] += RNG.normal(2.0, 0.4)
                    # Cancer-type-specific signal: only up in its own cancer type.
                    for g in CANCER_SPECIFIC_GENES[cancer_type]:
                        expr_values[all_genes.index(g)] += RNG.normal(2.5, 0.4)

                expr_values += batch_shift
                expr_values += RNG.normal(0, 0.3, size=len(all_genes))  # measurement noise

                sample_id = f"{patient_id}-{label}"
                rows_expr.append(pd.Series(expr_values, index=all_genes, name=sample_id))
                rows_meta.append(
                    {
                        "sample_id": sample_id,
                        "label": label,
                        "cancer_type": cancer_type,
                        "study": study,
                        "patient_id": patient_id,
                    }
                )

    expr = pd.DataFrame(rows_expr)
    meta = pd.DataFrame(rows_meta).set_index("sample_id")
    # NOTE: this synthetic data is generated already on a log2-like scale, so
    # we skip data_loading.log_transform() in this demo — real raw
    # counts/intensities should be log-transformed first (see data_loading.py).
    return ExpressionDataset(expr=expr, meta=meta)


def main():
    print("=" * 70)
    print("STAGE 1: Data loading & preprocessing (synthetic data)")
    print("=" * 70)
    dataset = make_synthetic_dataset()
    print(f"Loaded {dataset.expr.shape[0]} samples x {dataset.expr.shape[1]} genes")
    print(dataset.meta.groupby(["cancer_type", "label"]).size(), "\n")

    matched_pairs = find_matched_pairs(dataset.meta)
    print(f"Found {len(matched_pairs)} matched tumor/normal pairs across all cancer types")

    expr_corrected = batch_correct(dataset.expr, dataset.meta, batch_col="study")
    dataset = ExpressionDataset(expr=expr_corrected, meta=dataset.meta)

    print("\n" + "=" * 70)
    print("STAGE 2: Differential expression per cancer type")
    print("=" * 70)
    de_results_by_cancer = run_differential_expression_per_cancer(dataset.expr, dataset.meta, method="mannwhitney")
    for cancer_type, de_df in de_results_by_cancer.items():
        n_sig = (de_df["padj"] < 0.05).sum()
        print(f"{cancer_type}: {n_sig} genes at FDR < 0.05 (top hit: {de_df.index[0]}, padj={de_df['padj'].iloc[0]:.2e})")

    print("\n" + "=" * 70)
    print("STAGE 3: Signature intersection + proliferation filtering")
    print("=" * 70)
    signature_result = build_pancancer_signature(
        de_results_by_cancer,
        proliferation_genes=set(PROLIFERATION_GENES),
        padj_threshold=0.05,
        log2fc_threshold=1.0,
        top_n=50,
    )
    print(f"Raw pan-cancer intersection ({len(signature_result['raw_intersection'])} genes): "
          f"{sorted(signature_result['raw_intersection'])}")
    print(f"Flagged as proliferation markers: {sorted(signature_result['proliferation_flagged'])}")
    print(f"Final candidate signature ({len(signature_result['final_signature'])} genes): "
          f"{sorted(signature_result['final_signature'])}")

    final_candidate_genes = sorted(signature_result["final_signature"])
    if len(final_candidate_genes) == 0:
        raise RuntimeError("No genes survived intersection + filtering — check thresholds.")

    print("\n" + "=" * 70)
    print("STAGE 4: Classifier comparison + recursive feature elimination")
    print("=" * 70)
    X = dataset.expr[final_candidate_genes]
    y = (dataset.meta["label"] == "tumor").astype(int).to_numpy()

    comparison = compare_classifiers(X, y)
    print("Classifier comparison on the full candidate signature:")
    print(comparison.round(3))

    best_classifier_name = comparison.index[0]
    print(f"\nUsing best classifier ('{best_classifier_name}') for RFE down to <=5 genes...")
    classifiers = build_classifiers()
    rfe_results = recursive_feature_elimination_cv(X, y, classifiers[best_classifier_name], min_features=1)
    print(rfe_results[["n_genes", "roc_auc", "accuracy", "f1"]].round(3))

    fig = plot_performance_vs_genes(rfe_results, save_path="outputs/performance_vs_genes.png")
    print("Saved performance-vs-gene-count plot to outputs/performance_vs_genes.png")

    final_row = rfe_results[rfe_results["n_genes"] <= 5].iloc[0]
    final_genes = final_row["genes"]
    print(f"\nFinal minimal signature ({len(final_genes)} genes): {final_genes}")
    print(f"Performance at this size: AUC={final_row['roc_auc']:.3f}, accuracy={final_row['accuracy']:.3f}")

    print("\n" + "=" * 70)
    print("STAGE 5: Cross-cancer generalization (leave-one-cancer-type-out)")
    print("=" * 70)
    X_final = dataset.expr[final_genes]
    loco_results = leave_one_cancer_out(X_final, y, dataset.meta["cancer_type"], classifiers[best_classifier_name])
    print(loco_results.round(3))

    print("\n" + "=" * 70)
    print("STAGE 6: Independent validation (simulated external cohort)")
    print("=" * 70)
    # In a real run, `validation.load_independent_dataset` would read a GEO
    # series from disk. Here we simulate that external cohort by re-using
    # "studyB" samples as a stand-in independent dataset and evaluating the
    # model fit on "studyA" only, to exercise the same code path.
    train_mask = dataset.meta["study"] == "studyA"
    test_mask = dataset.meta["study"] == "studyB"

    final_model = classifiers[best_classifier_name]
    final_model.fit(X_final.loc[train_mask], y[train_mask.to_numpy()])
    external_metrics = evaluate_on_independent(final_model, X_final.loc[test_mask], y[test_mask.to_numpy()])
    print(f"Performance on held-out 'studyB' (independent-cohort stand-in): {external_metrics}")

    print("\n" + "=" * 70)
    print("STAGE 7: Interpretability")
    print("=" * 70)
    importance_df = get_feature_importance(final_model, final_genes)
    print(importance_df)

    direction_summary = summarize_signature_direction(de_results_by_cancer, final_genes)
    print("\nPer-cancer-type log2FC / padj for the final signature genes:")
    print(direction_summary.round(3))

    print("\nPipeline ran end-to-end successfully on synthetic data.")
    print("Next step: point data_loading.py's loaders at your real TCGA/GEO files (see README.md).")


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)
    main()
