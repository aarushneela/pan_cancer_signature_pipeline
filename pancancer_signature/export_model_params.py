"""
Export the exact fitted parameters of your final 2-gene (COL10A1 + NME1)
cross-platform model, trained on all of TCGA — so the interactive demo tool
uses your REAL trained model's numbers, not placeholders.

Usage: python export_model_params.py
"""

from run_stage6_validation import CROSS_PLATFORM_GENES, GENE_ENSEMBL, load_tcga_combined
from src.modeling import build_classifiers


def main():
    genes_ensembl = [GENE_ENSEMBL[g] for g in CROSS_PLATFORM_GENES]
    X_train, y_train = load_tcga_combined(genes_ensembl)
    X_train.columns = CROSS_PLATFORM_GENES

    model = build_classifiers()["logreg_elasticnet"]
    model.fit(X_train, y_train)

    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]

    print("Gene order:", CROSS_PLATFORM_GENES)
    print("\n--- StandardScaler ---")
    print("mean_:", scaler.mean_.tolist())
    print("scale_:", scaler.scale_.tolist())
    print("\n--- LogisticRegression ---")
    print("coef_:", clf.coef_.tolist())
    print("intercept_:", clf.intercept_.tolist())

    print("\n--- Training data reference ranges (for the tool's sliders) ---")
    for gene in CROSS_PLATFORM_GENES:
        col = X_train[gene]
        print(f"{gene}: min={col.min():.3f}, max={col.max():.3f}, "
              f"tumor_mean={X_train.loc[y_train == 1, gene].mean():.3f}, "
              f"normal_mean={X_train.loc[y_train == 0, gene].mean():.3f}")


if __name__ == "__main__":
    main()
