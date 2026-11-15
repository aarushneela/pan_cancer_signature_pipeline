"""
Convert a raw GEO series matrix file (GSE15852-style) into the
expression_matrix.tsv / sample_metadata.tsv format data_loading.py expects.

GEO series matrix files mix metadata lines (starting with "!") with one big
data table. This script:
    1. Scans the "!Sample_*" metadata lines to build per-sample info:
       GEO accession (GSM ID), tumor/normal label (from
       Sample_source_name_ch1's free-text description), and a patient ID
       (extracted from Sample_title, which for this series looks like
       "Normal BC0043N" / "Cancer BC0043T" — same "BC0043" for a pair).
    2. Reads the actual data table (probe IDs x samples), transposes it to
       samples x probes.
    3. Writes both out as expression_matrix.tsv / sample_metadata.tsv.

IMPORTANT CAVEAT: the row index here is Affymetrix PROBE IDs (e.g.
"1007_s_at"), not gene symbols — this platform's genes need a probe->gene
symbol mapping (via the GPL96 platform annotation file) before this dataset's
columns will line up with your TCGA genes (which are Ensembl IDs) or your
proliferation gene list (which uses gene symbols like "MKI67"). That mapping
is a separate follow-up step — this script just gets the raw data extracted
and correctly labeled first.

NOTE: other GEO series (GSE32863, GSE8671) may format Sample_title / 
Sample_source_name_ch1 slightly differently. Peek at each one first (same way
we did for GSE15852) before assuming this exact parsing logic transfers
unchanged — the LABEL_KEYWORDS and patient-ID stripping logic below may need
small tweaks per series.
"""

import gzip
import re

import pandas as pd

# Keywords used to classify each sample as tumor or normal, based on the
# free-text Sample_source_name_ch1 field. Add more keywords here if a
# series' wording doesn't match (e.g. "carcinoma", "malignant").
TUMOR_KEYWORDS = ["tumor", "tumour", "cancer", "carcinoma", "malignant", "adenoma", "polyp"]
NORMAL_KEYWORDS = ["normal", "adjacent", "healthy"]


def _classify_label(text: str) -> str:
    text_lower = text.lower()
    # Check NORMAL keywords first: some series word normal samples as
    # "non-tumor" or "adjacent non-tumor" — which contains the substring
    # "tumor" and would be misclassified if TUMOR_KEYWORDS were checked
    # first. Checking NORMAL_KEYWORDS first avoids that trap.
    if any(k in text_lower for k in NORMAL_KEYWORDS):
        return "normal"
    if any(k in text_lower for k in TUMOR_KEYWORDS):
        return "tumor"
    return "unknown"


def _extract_patient_id(title: str) -> str:
    """Pull a shared patient identifier out of a sample title.

    Handles two observed formats:
      - Underscore-separated, embedded anywhere (e.g.
        "...lung tissue 05L12_N (expression)" / "...05L12_T (expression)")
        -> returns "05L12".
      - No separator, at the end of the title (e.g. "Normal BC0043N" /
        "Cancer BC0043T") -> returns "BC0043".
    Falls back to the full title if neither pattern matches (unpaired).
    """
    underscore_match = re.search(r"([A-Za-z0-9]+)_[NT]\b", title)
    if underscore_match:
        return underscore_match.group(1)

    last_token = title.strip().split()[-1]
    suffix_match = re.match(r"^([A-Za-z0-9]+?)([A-Za-z])$", last_token)
    if suffix_match:
        return suffix_match.group(1)

    return title.strip()


def _find_patient_id_characteristic(characteristics_rows: list[list[str]], prefix: str = "patient id:"):
    """Look through repeated Sample_characteristics_ch1 rows for one that
    holds a per-patient identifier (e.g. "Patient ID: GL"), and return the
    stripped values if found. Returns None if no matching row exists — the
    caller should fall back to extracting an ID from Sample_title instead.
    """
    for row in characteristics_rows:
        if row and row[0].strip().lower().startswith(prefix):
            return [v.split(":", 1)[1].strip() if ":" in v else v.strip() for v in row]
    return None


def parse_geo_series_matrix(path: str, cancer_type: str, study: str, out_dir: str):
    """Parse one GEO series matrix file into expression_matrix.tsv / sample_metadata.tsv.

    Args:
        path: path to the raw .txt.gz series matrix file.
        cancer_type: short code, e.g. 'BRCA'.
        study: label for this dataset's source, e.g. 'GSE15852'.
        out_dir: folder to write the two output files into.
    """
    print(f"[{study}] Reading metadata lines from: {path}")
    sample_meta_fields: dict[str, list[list[str]]] = {}
    data_start_line = None

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if line.startswith("!Sample_"):
                parts = line.split("\t")
                field = parts[0].lstrip("!")
                values = [v.strip('"') for v in parts[1:]]
                # Some fields (notably Sample_characteristics_ch1) repeat
                # across multiple lines with different sub-attributes (e.g.
                # one line for tissue type, another for patient ID) — keep
                # ALL occurrences as a list of rows, not just the first.
                sample_meta_fields.setdefault(field, []).append(values)
            elif line.startswith("!series_matrix_table_begin"):
                data_start_line = i
                break

    if data_start_line is None:
        raise ValueError(f"Could not find '!series_matrix_table_begin' in {path}")

    required = ["Sample_geo_accession", "Sample_source_name_ch1", "Sample_title"]
    missing = [f for f in required if f not in sample_meta_fields]
    if missing:
        raise ValueError(
            f"[{study}] Missing expected metadata field(s) {missing}. "
            "This series may be formatted differently — peek at its raw text "
            "and adjust the field names used above."
        )

    gsm_ids = sample_meta_fields["Sample_geo_accession"][0]
    labels = [_classify_label(s) for s in sample_meta_fields["Sample_source_name_ch1"][0]]
    titles = sample_meta_fields["Sample_title"][0]

    # Prefer a dedicated "Patient ID: XX" characteristics row if this series
    # has one (more reliable than guessing from title text/position); fall
    # back to title-based extraction otherwise.
    patient_id_row = _find_patient_id_characteristic(sample_meta_fields.get("Sample_characteristics_ch1", []))
    if patient_id_row is not None:
        patient_ids = patient_id_row
        print(f"[{study}] Using a dedicated 'Patient ID' characteristics field for pairing.")
    else:
        patient_ids = [_extract_patient_id(t) for t in titles]
        print(f"[{study}] No dedicated patient ID field found; extracting patient ID from Sample_title instead.")

    n_unknown = labels.count("unknown")
    if n_unknown:
        print(
            f"[{study}] WARNING: {n_unknown} sample(s) had a source description that didn't "
            "match tumor/normal keywords — check TUMOR_KEYWORDS/NORMAL_KEYWORDS above."
        )

    meta = pd.DataFrame(
        {
            "sample_id": gsm_ids,
            "label": labels,
            "patient_id": patient_ids,
            "cancer_type": cancer_type,
            "study": study,
        }
    ).set_index("sample_id")

    print(f"[{study}] Reading expression data table (this can take a moment)...")
    # skiprows accounts for the metadata lines already scanned above; the
    # table's own header row (ID_REF, GSM1, GSM2, ...) becomes the DataFrame header.
    expr = pd.read_csv(path, sep="\t", skiprows=data_start_line + 1, quotechar='"')
    expr = expr[expr["ID_REF"] != "!series_matrix_table_end"]  # drop the closing marker row if present
    expr = expr.set_index("ID_REF")
    expr = expr.T  # transpose: samples become rows, probes become columns
    expr.index.name = "sample_id"

    shared = expr.index.intersection(meta.index)
    dropped = len(expr.index) - len(shared)
    if dropped:
        print(f"[{study}] Note: {dropped} sample(s) in the data table had no matching metadata; dropped.")
    expr = expr.loc[shared]
    meta = meta.loc[shared]

    expr_out = f"{out_dir}/expression_matrix.tsv"
    meta_out = f"{out_dir}/sample_metadata.tsv"
    expr.to_csv(expr_out, sep="\t")
    meta.to_csv(meta_out, sep="\t")

    n_tumor = (meta["label"] == "tumor").sum()
    n_normal = (meta["label"] == "normal").sum()
    print(f"[{study}] Saved {expr.shape[0]} samples x {expr.shape[1]} probes -> {expr_out}")
    print(f"[{study}] Saved metadata ({n_tumor} tumor, {n_normal} normal) -> {meta_out}\n")


if __name__ == "__main__":
    # FILL IN: adjust filenames if yours differ.
    parse_geo_series_matrix(
        path="data/raw/geo_validation/gse15852_brca/GSE15852_series_matrix.txt.gz",
        cancer_type="BRCA",
        study="GSE15852",
        out_dir="data/raw/geo_validation/gse15852_brca",
    )
    parse_geo_series_matrix(
        path="data/raw/geo_validation/gse32863_luad/GSE32863_series_matrix.txt.gz",
        cancer_type="LUAD",
        study="GSE32863",
        out_dir="data/raw/geo_validation/gse32863_luad",
    )
    # NOTE: GSE8671's "tumor" class is colorectal ADENOMA (a precancerous
    # polyp), not invasive carcinoma. It's still a real, biologically
    # relevant tumor-vs-normal comparison, but expect this dataset to be a
    # somewhat easier or different comparison than TCGA-COAD's malignant
    # carcinoma — worth calling out explicitly as a limitation in your
    # write-up rather than treating it as an equivalent severity match to
    # the breast/lung validation sets.
    parse_geo_series_matrix(
        path="data/raw/geo_validation/gse8671_coad/GSE8671_series_matrix.txt.gz",
        cancer_type="COAD",
        study="GSE8671",
        out_dir="data/raw/geo_validation/gse8671_coad",
    )
    print("Done with all three GEO validation datasets.")