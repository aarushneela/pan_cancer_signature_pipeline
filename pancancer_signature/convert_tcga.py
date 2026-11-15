"""
Convert raw Xena TCGA downloads (STAR-FPKM expression + clinical files) into
the expression_matrix.tsv / sample_metadata.tsv format data_loading.py expects.

Run this from inside the pancancer_signature folder:
    python convert_tcga.py

What it does, per cancer type:
    1. Reads the STAR-FPKM file (genes as rows, samples as columns — Xena's
       native layout) and TRANSPOSES it to samples x genes, which is what
       the pipeline needs.
    2. Strips the version suffix off Ensembl gene IDs (ENSG00000000003.15 ->
       ENSG00000000003) so they match cleanly if you cross-reference genes
       later (version numbers change between Ensembl releases and aren't
       needed for this project).
    3. Reads the clinical file and pulls out exactly the columns needed:
       'sample' (matches the expression file's sample IDs), 'id' (a UUID
       shared between a patient's tumor and normal sample — this becomes
       our patient_id for matched-pair detection), and 'tissue_type.samples'
       (already says "Tumor"/"Normal" directly).
    4. Aligns the two tables to the same set of sample IDs and writes both
       out as expression_matrix.tsv and sample_metadata.tsv, right alongside
       your original raw downloads.

Note: transposing a ~60,000-gene x ~1,200-sample matrix takes a minute or two
and a few hundred MB of RAM — that's normal, just let it run.
"""

import pandas as pd


def convert_tcga_cohort(expr_path: str, clinical_path: str, cancer_type: str, out_dir: str):
    """Reshape one TCGA cohort's raw Xena files into the pipeline's format.

    Args:
        expr_path: path to the raw STAR-FPKM .tsv.gz file (genes x samples).
        clinical_path: path to the raw clinical .tsv.gz file.
        cancer_type: short code to stamp on every sample, e.g. 'BRCA'.
        out_dir: folder to write expression_matrix.tsv / sample_metadata.tsv
            into (typically the same folder the raw files are in).
    """
    print(f"[{cancer_type}] Loading expression file: {expr_path}")
    expr = pd.read_csv(expr_path, sep="\t")
    expr = expr.set_index("Ensembl_ID")
    expr.index = expr.index.str.split(".").str[0]  # drop Ensembl version suffix
    expr = expr.T  # transpose: now samples are rows, genes are columns
    expr.index.name = "sample_id"

    print(f"[{cancer_type}] Loading clinical file: {clinical_path}")
    clinical = pd.read_csv(clinical_path, sep="\t")
    meta = clinical[["sample", "id", "tissue_type.samples"]].copy()
    meta.columns = ["sample_id", "patient_id", "label"]
    meta["label"] = meta["label"].str.lower().str.strip()
    meta["cancer_type"] = cancer_type
    meta["study"] = "TCGA"
    meta = meta.set_index("sample_id")
    meta = meta[~meta.index.duplicated(keep="first")]  # safety net against any dupes

    # Keep only samples present in BOTH files, in the same order.
    shared = expr.index.intersection(meta.index)
    dropped = len(expr.index) - len(shared)
    if dropped:
        print(f"[{cancer_type}] Note: {dropped} expression sample(s) had no matching clinical row; dropped.")
    expr = expr.loc[shared]
    meta = meta.loc[shared]

    expr_out = f"{out_dir}/expression_matrix.tsv"
    meta_out = f"{out_dir}/sample_metadata.tsv"
    expr.to_csv(expr_out, sep="\t")
    meta.to_csv(meta_out, sep="\t")

    n_tumor = (meta["label"] == "tumor").sum()
    n_normal = (meta["label"] == "normal").sum()
    print(f"[{cancer_type}] Saved {expr.shape[0]} samples x {expr.shape[1]} genes -> {expr_out}")
    print(f"[{cancer_type}] Saved metadata ({n_tumor} tumor, {n_normal} normal) -> {meta_out}\n")


if __name__ == "__main__":
    # FILL IN: adjust these paths/filenames if yours differ at all from what's below.
    convert_tcga_cohort(
        expr_path="data/raw/tcga_brca/TCGA-BRCA.star_fpkm.tsv.gz",
        clinical_path="data/raw/tcga_brca/TCGA-BRCA.clinical.tsv.gz",
        cancer_type="BRCA",
        out_dir="data/raw/tcga_brca",
    )
    convert_tcga_cohort(
        expr_path="data/raw/tcga_luad/TCGA-LUAD.star_fpkm.tsv.gz",
        clinical_path="data/raw/tcga_luad/TCGA-LUAD.clinical.tsv.gz",
        cancer_type="LUAD",
        out_dir="data/raw/tcga_luad",
    )
    convert_tcga_cohort(
        expr_path="data/raw/tcga_coad/TCGA-COAD.star_fpkm.tsv.gz",
        clinical_path="data/raw/tcga_coad/TCGA-COAD.clinical.tsv.gz",
        cancer_type="COAD",
        out_dir="data/raw/tcga_coad",
    )
    print("Done. Each tcga_* folder now has expression_matrix.tsv + sample_metadata.tsv.")
