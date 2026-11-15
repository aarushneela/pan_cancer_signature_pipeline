"""
Stage 6 — Independent GEO validation, the real test of generalization.

Trains the final classifier on ALL of TCGA (no held-out TCGA split needed —
GEO data was never touched during training, so it's already a clean,
independent test) and evaluates it on all three GEO validation datasets.

Two evaluations, both legitimate and worth reporting:
    1. A clean 2-gene model (COL10A1 + NME1) — the only two of your five
       final genes present on ALL THREE GEO platforms, so this is your
       most directly comparable, apples-to-apples cross-cancer result.
    2. A "best available" model per cancer type, using whichever of your
       5 genes actually exist on that specific platform (see
       geo_probe_mapping.json from build_geo_probe_mapping.py) — exploratory,
       shows whether the extra genes help where they're measurable.

IQANK1 is excluded from all GEO evaluations below since it wasn't found on
any of the three platforms — a genuine, worth-reporting limitation of using
older microarray data, not a pipeline error.
"""

import json

import pandas as pd

from src.data_loading import load_expression_matrix, load_sample_metadata
from src.modeling import build_classifiers
from src.validation import evaluate_on_independent, rescale_external_to_training_distribution

TCGA_COHORTS = {
    "BRCA": "data/raw/tcga_brca",
    "LUAD": "data/raw/tcga_luad",
    "COAD": "data/raw/tcga_coad",
}

GEO_DATASETS = {
    "gse15852_brca": ("BRCA", "data/raw/geo_validation/gse15852_brca"),
    "gse32863_luad": ("LUAD", "data/raw/geo_validation/gse32863_luad"),
    "gse8671_coad": ("COAD", "data/raw/geo_validation/gse8671_coad"),
}

GENE_ENSEMBL = {
    "COL10A1": "ENSG00000123500",
    "NME1": "ENSG00000239672",
    "IQANK1": "ENSG00000203499",  # excluded everywhere — not on any GEO platform
    "IQGAP3": "ENSG00000183856",
    "HJURP": "ENSG00000123485",
}

CROSS_PLATFORM_GENES = ["COL10A1", "NME1"]  # present on all 3 platforms


def evaluate_on_independent_with_recalibration(model, X_external, y_external):
    """Evaluate a model on external data, reporting both the default-threshold
    metrics AND metrics using a threshold recalibrated for this dataset.

    Why this is needed: your TCGA training data is heavily class-imbalanced
    (e.g. BRCA is ~90% tumor), but your GEO validation sets are ~50/50 by
    design (matched pairs). A classifier's default decision threshold
    reflects the class balance it was trained on — applying it unchanged to
    a differently-balanced population is a well-documented way to get poor
    accuracy even when the model's underlying ranking (AUC) is genuinely
    good. Youden's J statistic picks the threshold that best balances true
    and false positive rates on THIS data, isolating "does the model rank
    samples correctly" (AUC, already trustworthy) from "was the default
    cutoff appropriate for this population's class balance" (a separate,
    fixable calibration question).

    Note: recalibrating the threshold using the same data you evaluate
    accuracy on is a supplementary diagnostic, not a strict held-out test —
    report AUC as your primary, fully independent metric, and present this
    recalibrated accuracy as "what performance looks like once threshold
    miscalibration is corrected for," not as an equally rigorous number.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

    y_score = model.predict_proba(X_external)[:, 1]
    default_pred = model.predict(X_external)

    fpr, tpr, thresholds = roc_curve(y_external, y_score)
    youden_j = tpr - fpr
    best_threshold = thresholds[youden_j.argmax()]
    recalibrated_pred = (y_score >= best_threshold).astype(int)

    return {
        "auc": roc_auc_score(y_external, y_score),
        "default_threshold_accuracy": accuracy_score(y_external, default_pred),
        "recalibrated_threshold": float(best_threshold),
        "recalibrated_accuracy": accuracy_score(y_external, recalibrated_pred),
        "recalibrated_precision": precision_score(y_external, recalibrated_pred, zero_division=0),
        "recalibrated_recall": recall_score(y_external, recalibrated_pred, zero_division=0),
        "recalibrated_f1": f1_score(y_external, recalibrated_pred, zero_division=0),
    }


def load_tcga_combined(genes_ensembl: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load just the needed gene columns across all 3 TCGA cohorts (fast — no DE re-run)."""
    exprs, metas = [], []
    for name, folder in TCGA_COHORTS.items():
        expr = load_expression_matrix(f"{folder}/expression_matrix.tsv")
        meta = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        exprs.append(expr[genes_ensembl])
        metas.append(meta)
    X = pd.concat(exprs, axis=0)
    meta_all = pd.concat(metas, axis=0)
    y = (meta_all["label"] == "tumor").astype(int).to_numpy()
    return X, y


def load_geo_gene_expression(dataset_folder: str, gene_to_probes: dict[str, list[str]]) -> pd.DataFrame:
    """Build a gene-level expression frame from a GEO dataset's probe-level matrix.

    If a gene has multiple probes on the platform, averages across them —
    a standard, defensible choice when multiple probes target the same gene
    (they usually correlate strongly; averaging reduces single-probe noise).
    """
    expr = load_expression_matrix(f"{dataset_folder}/expression_matrix.tsv")
    gene_cols = {}
    for gene_symbol, probes in gene_to_probes.items():
        available_probes = [p for p in probes if p in expr.columns]
        if not available_probes:
            continue  # gene not on this platform
        gene_cols[gene_symbol] = expr[available_probes].mean(axis=1)
    return pd.DataFrame(gene_cols)


def run_evaluation(gene_symbols: list[str], label: str, probe_mapping: dict):
    """Train on TCGA using these genes, evaluate on every GEO dataset that has them."""
    genes_ensembl = [GENE_ENSEMBL[g] for g in gene_symbols]
    print(f"\n{'=' * 70}\n{label}: {gene_symbols}\n{'=' * 70}")

    X_train, y_train = load_tcga_combined(genes_ensembl)
    # Rename TCGA's Ensembl-ID columns to plain gene symbols so they line up
    # with the GEO frame's symbol-named columns built by load_geo_gene_expression.
    X_train.columns = gene_symbols

    model = build_classifiers()["logreg_elasticnet"]  # last known winning classifier; adjust if yours differed
    model.fit(X_train, y_train)
    print(f"Trained on {X_train.shape[0]} TCGA samples using {len(gene_symbols)} gene(s).")

    # Save training-set per-gene mean/std — needed to rescale each external
    # GEO dataset onto TCGA's numeric scale before evaluating (RNA-seq vs.
    # microarray live on very different absolute value ranges; without this,
    # the model's learned decision threshold doesn't transfer, even though
    # its *ranking* of samples can still be good).
    train_means = X_train.mean(axis=0)
    train_stds = X_train.std(axis=0)

    for dataset_name, (cancer_type, folder) in GEO_DATASETS.items():
        gene_to_probes = probe_mapping[dataset_name]["gene_to_probes"]
        available_genes = [g for g in gene_symbols if gene_to_probes.get(g)]
        if len(available_genes) < len(gene_symbols):
            missing = set(gene_symbols) - set(available_genes)
            print(f"\n{dataset_name} ({cancer_type}): SKIPPING — missing {missing} on this platform.")
            continue

        X_external = load_geo_gene_expression(folder, gene_to_probes)[gene_symbols]
        meta_external = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        meta_external = meta_external.loc[meta_external.index.intersection(X_external.index)]
        X_external = X_external.loc[meta_external.index]
        y_external = (meta_external["label"] == "tumor").astype(int)

        X_external = rescale_external_to_training_distribution(X_external, train_means, train_stds)

        metrics = evaluate_on_independent_with_recalibration(model, X_external, y_external)
        print(f"\n{dataset_name} ({cancer_type}), n={len(y_external)}: {metrics}")


def main():
    with open("data/processed/geo_probe_mapping.json") as f:
        probe_mapping = json.load(f)

    # Evaluation 1: clean, directly comparable 2-gene model across all 3 cancer types.
    run_evaluation(CROSS_PLATFORM_GENES, "CROSS-PLATFORM MODEL (genes present on all 3 platforms)", probe_mapping)

    # Evaluation 2: best available subset per cancer type (exploratory).
    for dataset_name, (cancer_type, folder) in GEO_DATASETS.items():
        gene_to_probes = probe_mapping[dataset_name]["gene_to_probes"]
        available_genes = [g for g in GENE_ENSEMBL if gene_to_probes.get(g)]
        print(f"\n{'=' * 70}\nBEST-AVAILABLE MODEL for {dataset_name}: {available_genes}\n{'=' * 70}")

        genes_ensembl = [GENE_ENSEMBL[g] for g in available_genes]
        X_train, y_train = load_tcga_combined(genes_ensembl)
        X_train.columns = available_genes

        model = build_classifiers()["logreg_elasticnet"]
        model.fit(X_train, y_train)

        train_means = X_train.mean(axis=0)
        train_stds = X_train.std(axis=0)

        X_external = load_geo_gene_expression(folder, gene_to_probes)[available_genes]
        meta_external = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        meta_external = meta_external.loc[meta_external.index.intersection(X_external.index)]
        X_external = X_external.loc[meta_external.index]
        y_external = (meta_external["label"] == "tumor").astype(int)

        X_external = rescale_external_to_training_distribution(X_external, train_means, train_stds)

        metrics = evaluate_on_independent_with_recalibration(model, X_external, y_external)
        print(f"{dataset_name} ({cancer_type}), n={len(y_external)}: {metrics}")

    print("\nDone. This is the final, independent-data test of your pan-cancer signature.")


if __name__ == "__main__":
    main()