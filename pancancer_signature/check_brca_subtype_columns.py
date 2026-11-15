"""Check whether histological type / receptor status columns already exist
in your already-downloaded BRCA clinical file, before deciding whether you
need to download anything new.
"""
import pandas as pd

clinical = pd.read_csv("data/raw/tcga_brca/TCGA-BRCA.clinical.tsv.gz", sep="\t")

candidates = [c for c in clinical.columns if any(
    kw in c.lower() for kw in ["diagnosis", "histolog", "er_status", "pr_status", "her2", "receptor"]
)]

print("Columns that might hold histological type or receptor status:")
for c in candidates:
    print(f"\n--- {c} ---")
    print(clinical[c].value_counts(dropna=False).head(10))
