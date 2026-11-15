"""
Run Stages 1-5 of the pipeline on your REAL, combined TCGA data (BRCA + LUAD
+ COAD) — the payoff after all that downloading and reshaping.

Stage 6 (independent GEO validation) is deliberately left out of this script:
your three GEO files use platform-specific probe IDs (not the Ensembl gene
IDs TCGA uses), so they need a probe -> gene symbol mapping step before they
can be compared against these results. That's a separate follow-up.

A note on the proliferation gene list below: TCGA's genes are Ensembl IDs
(e.g. "ENSG00000148773"), not symbols, so the proliferation markers are given
here as their Ensembl IDs rather than names like "MKI67" directly. These are
well-established, widely-cited IDs, but if precision matters for your
write-up, double-check them against Ensembl/GeneCards before treating them as
final — worth doing before running your last, publication-facing pass.
"""

import pandas as pd

import os

from src.cross_cancer import leave_one_cancer_out
from src.data_loading import ExpressionDataset, load_expression_matrix, load_sample_metadata, find_matched_pairs
from src.differential_expression import run_differential_expression_per_cancer
from src.feature_selection import plot_performance_vs_genes, recursive_feature_elimination_cv
from src.interpretability import get_feature_importance, summarize_signature_direction
from src.modeling import build_classifiers, compare_classifiers
from src.signature_intersection import build_pancancer_signature

# Proliferation/cell-cycle reference gene list, as Ensembl IDs.
# Built by build_proliferation_list.py from a proper, broad gene symbol set
# (~70 genes spanning DNA replication, cyclins/CDKs, mitotic checkpoint, and
# cytokinesis) via a live mygene.info lookup — see that script for the full
# symbol list and how the mapping was done.
def load_proliferation_genes(path: str = "data/processed/proliferation_genes_ensembl.txt") -> set[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python build_proliferation_list.py` first to generate it "
            "(requires `pip install mygene`)."
        )
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}

TCGA_COHORTS = {
    "BRCA": "data/raw/tcga_brca",
    "LUAD": "data/raw/tcga_luad",
    "COAD": "data/raw/tcga_coad",
}


def load_tcga_cohort(folder: str) -> ExpressionDataset:
    """Load one already-reshaped TCGA cohort (from convert_tcga.py's output)."""
    expr = load_expression_matrix(f"{folder}/expression_matrix.tsv")
    meta = load_sample_metadata(f"{folder}/sample_metadata.tsv")
    # Xena's STAR-FPKM values are already log2(FPKM+1) — confirmed earlier by
    # the small (0-6ish) value range you saw when peeking at the raw file, so
    # we do NOT re-log-transform here.
    return ExpressionDataset(expr=expr, meta=meta)


def main():
    print("=" * 70)
    print("STAGE 1: Loading real TCGA data (BRCA + LUAD + COAD)")
    print("=" * 70)
    datasets = {name: load_tcga_cohort(folder) for name, folder in TCGA_COHORTS.items()}
    for name, ds in datasets.items():
        print(f"{name}: {ds.expr.shape[0]} samples x {ds.expr.shape[1]} genes")

    # Combine all three cohorts. inner join on columns keeps only genes
    # present in all three (should be nearly all of them, since all three
    # went through the same GDC/Xena pipeline).
    combined_expr = pd.concat([ds.expr for ds in datasets.values()], axis=0, join="inner")
    combined_meta = pd.concat([ds.meta for ds in datasets.values()], axis=0)
    print(f"\nCombined: {combined_expr.shape[0]} samples x {combined_expr.shape[1]} shared genes")
    print(combined_meta.groupby(["cancer_type", "label"]).size(), "\n")

    matched_pairs = find_matched_pairs(combined_meta)
    print(f"Found {len(matched_pairs)} matched tumor/normal pairs across all cancer types\n")

    print("=" * 70)
    print("STAGE 2: Differential expression per cancer type (this may take a few minutes)")
    print("=" * 70)
    de_results_by_cancer = run_differential_expression_per_cancer(combined_expr, combined_meta, method="mannwhitney")
    for cancer_type, de_df in de_results_by_cancer.items():
        n_sig = (de_df["padj"] < 0.05).sum()
        print(f"{cancer_type}: {n_sig} genes at FDR < 0.05 (top hit: {de_df.index[0]}, padj={de_df['padj'].iloc[0]:.2e})")

    print("\n" + "=" * 70)
    print("STAGE 3: Signature intersection + proliferation filtering")
    print("=" * 70)
    signature_result = build_pancancer_signature(
        de_results_by_cancer,
        proliferation_genes=load_proliferation_genes(),
        padj_threshold=0.05,
        log2fc_threshold=1.0,
        top_n=200,
    )
    print(f"Raw pan-cancer intersection: {len(signature_result['raw_intersection'])} genes")
    print(f"Flagged as proliferation markers: {signature_result['proliferation_flagged']}")
    print(f"Final candidate signature: {len(signature_result['final_signature'])} genes")

    final_candidate_genes = sorted(signature_result["final_signature"])
    if len(final_candidate_genes) == 0:
        raise RuntimeError(
            "No genes survived intersection + filtering. Try loosening padj_threshold/"
            "log2fc_threshold/top_n above, or double-check the DE results look reasonable."
        )
    if len(final_candidate_genes) > 500:
        print(f"({len(final_candidate_genes)} genes is a lot for RFE — this may take a while in Stage 4.)")

    print("\n" + "=" * 70)
    print("STAGE 4: Classifier comparison + recursive feature elimination")
    print("=" * 70)
    X = combined_expr[final_candidate_genes]
    y = (combined_meta["label"] == "tumor").astype(int).to_numpy()

    comparison = compare_classifiers(X, y, groups=combined_meta["patient_id"])
    print("Classifier comparison on the full candidate signature (patient-grouped CV):")
    print(comparison.round(3))

    best_classifier_name = comparison.index[0]
    print(f"\nUsing best classifier ('{best_classifier_name}') for RFE down to <=5 genes...")
    classifiers = build_classifiers()
    rfe_results = recursive_feature_elimination_cv(
        X, y, classifiers[best_classifier_name], min_features=1, groups=combined_meta["patient_id"]
    )
    print(rfe_results[["n_genes", "roc_auc", "accuracy", "f1"]].round(3))

    fig = plot_performance_vs_genes(rfe_results, save_path="outputs/performance_vs_genes_REAL_grouped.png")
    print("Saved plot to outputs/performance_vs_genes_REAL_grouped.png")

    final_row = rfe_results[rfe_results["n_genes"] <= 5].iloc[0]
    final_genes = final_row["genes"]
    print(f"\nFinal minimal signature ({len(final_genes)} genes): {final_genes}")
    print(f"Performance at this size: AUC={final_row['roc_auc']:.3f}, accuracy={final_row['accuracy']:.3f}")

    print("\n" + "=" * 70)
    print("STAGE 5: Cross-cancer generalization (leave-one-cancer-type-out)")
    print("=" * 70)
    X_final = combined_expr[final_genes]
    loco_results = leave_one_cancer_out(X_final, y, combined_meta["cancer_type"], classifiers[best_classifier_name])
    print(loco_results.round(3))

    print("\n" + "=" * 70)
    print("STAGE 7: Interpretability")
    print("=" * 70)
    final_model = classifiers[best_classifier_name]
    final_model.fit(X_final, y)
    importance_df = get_feature_importance(final_model, final_genes)
    print(importance_df)

    direction_summary = summarize_signature_direction(de_results_by_cancer, final_genes)
    print("\nPer-cancer-type log2FC / padj for the final signature genes:")
    print(direction_summary.round(3))

    print("\nDone. Stages 1-5 and 7 ran on REAL TCGA data.")
    print("Stage 6 (independent GEO validation) still needs the probe->gene-symbol mapping step.")


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)
    main()
