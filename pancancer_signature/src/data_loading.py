"""
Stage 1 — Data loading & preprocessing.

Responsibilities of this module:
    1. Read an expression matrix (samples x genes) and a matching sample-metadata
       table from CSV/TSV.
    2. Log-transform expression values (RNA-seq counts/FPKM and microarray
       intensities both benefit from variance stabilization before parametric
       tests / ML).
    3. Identify matched tumor/normal pairs (same patient, two sample types) —
       useful for paired statistical tests in Stage 2.
    4. Correct for batch effects between studies/platforms (e.g. TCGA vs. a GEO
       microarray study) using ComBat, with a documented fallback if the
       `combat` package isn't installed.

Biological/statistical notes are inline as comments.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExpressionDataset:
    """Container bundling an expression matrix with its aligned metadata.

    Attributes:
        expr: DataFrame, index = sample_id, columns = gene symbols. Values are
            expression levels (raw counts, FPKM/TPM, or microarray intensity —
            whatever your source provides; normalize/log-transform before
            downstream use, see `log_transform`).
        meta: DataFrame, index = sample_id, with at least the columns
            'label' ('tumor'/'normal'), 'cancer_type' (e.g. 'BRCA'), and
            'study' (e.g. 'TCGA', 'GSE8671'). An optional 'patient_id' column
            enables matched-pair detection.
    """

    expr: pd.DataFrame
    meta: pd.DataFrame

    def __post_init__(self):
        # Keep only samples present in both the matrix and the metadata, and
        # force a consistent sample order between the two tables. Silently
        # mismatched indices are a very common source of subtle bugs in
        # genomics pipelines (e.g. an off-by-one row shift), so we fail loudly
        # instead if there's no overlap at all.
        shared = self.expr.index.intersection(self.meta.index)
        if len(shared) == 0:
            raise ValueError(
                "No overlapping sample IDs between expression matrix and metadata. "
                "Check that both files use the same sample_id convention."
            )
        if len(shared) < max(len(self.expr), len(self.meta)):
            warnings.warn(
                f"Only {len(shared)} of {len(self.expr)} expression rows and "
                f"{len(self.meta)} metadata rows overlap; dropping the rest."
            )
        self.expr = self.expr.loc[shared]
        self.meta = self.meta.loc[shared]


def load_expression_matrix(path: str, sep: str | None = None, sample_id_col: str = "sample_id") -> pd.DataFrame:
    """Load a samples x genes expression matrix from CSV/TSV.

    Args:
        path: path to the file. # FILL IN: real path, e.g. "data/raw/tcga_brca/expression_matrix.tsv"
        sep: field separator. If None, inferred from the file extension
            (',' for .csv, '\t' otherwise).
        sample_id_col: name of the column holding sample identifiers, which
            becomes the DataFrame index.

    Returns:
        DataFrame indexed by sample_id, one column per gene.
    """
    if sep is None:
        sep = "," if path.endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep)
    if sample_id_col not in df.columns:
        raise ValueError(f"Expected a '{sample_id_col}' column in {path}; found {list(df.columns)[:5]}...")
    df = df.set_index(sample_id_col)
    # Coerce everything to numeric; any stray non-numeric cell (e.g. a text
    # annotation column that slipped in) becomes NaN so it's easy to spot.
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_sample_metadata(path: str, sep: str | None = None, sample_id_col: str = "sample_id") -> pd.DataFrame:
    """Load sample metadata (label, cancer_type, study, optional patient_id).

    Args:
        path: path to the file. # FILL IN: real path, e.g. "data/raw/tcga_brca/sample_metadata.tsv"
        sep: field separator, inferred from extension if None.
        sample_id_col: column to use as the index.

    Returns:
        DataFrame indexed by sample_id with (at minimum) columns
        'label', 'cancer_type', 'study'.
    """
    if sep is None:
        sep = "," if path.endswith(".csv") else "\t"
    meta = pd.read_csv(path, sep=sep).set_index(sample_id_col)

    required = {"label", "cancer_type", "study"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata at {path} is missing required column(s): {missing}")

    meta["label"] = meta["label"].str.lower().str.strip()
    valid_labels = {"tumor", "normal"}
    bad = set(meta["label"].unique()) - valid_labels
    if bad:
        raise ValueError(f"Unexpected label values {bad}; expected one of {valid_labels}")
    return meta


def log_transform(expr: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """Log2-transform expression values.

    Rationale: raw RNA-seq counts/FPKM and microarray intensities are strongly
    right-skewed and heteroscedastic (variance grows with the mean). Most
    downstream statistics (t-test, linear models, PCA, regularized linear
    classifiers) assume roughly symmetric, variance-stabilized input. A
    log2(x + pseudocount) transform is the field-standard fix; the pseudocount
    avoids -inf for zero-count genes.

    If your data is already log-transformed (e.g. a GEO series matrix supplied
    as log2 intensities), do NOT call this again — check the value range first
    (log2 data is typically single digits to ~15-20; raw counts/FPKM are much
    larger and often include exact zeros).

    Args:
        expr: raw expression matrix (samples x genes).
        pseudocount: added before log2 to handle zeros.

    Returns:
        log2(expr + pseudocount), same shape as input.
    """
    if (expr < 0).any().any():
        warnings.warn("Negative values found in expression matrix before log-transform; check input.")
    return np.log2(expr.clip(lower=0) + pseudocount)


def find_matched_pairs(meta: pd.DataFrame) -> pd.DataFrame:
    """Identify patients with both a tumor and a matched-normal sample.

    Matched pairs (same patient, tumor + adjacent normal) are the gold
    standard for tumor-vs-normal differential expression because they cancel
    out inter-patient variability (genetic background, age, etc.), letting a
    paired test (e.g. paired t-test / Wilcoxon signed-rank) isolate the
    tumor-specific signal.

    Requires a 'patient_id' column in `meta`; if unavailable, downstream code
    should fall back to an unpaired test across all tumor vs. all normal
    samples (still valid, just statistically less powerful).

    Args:
        meta: sample metadata with 'patient_id', 'label', 'cancer_type'.

    Returns:
        DataFrame with columns ['patient_id', 'cancer_type', 'tumor_sample_id',
        'normal_sample_id'] — one row per matched pair.
    """
    if "patient_id" not in meta.columns:
        raise ValueError(
            "meta has no 'patient_id' column; cannot find matched pairs. "
            "Fall back to an unpaired tumor-vs-normal comparison instead."
        )

    pairs = []
    for (patient, cancer_type), group in meta.groupby(["patient_id", "cancer_type"]):
        tumor_ids = group.index[group["label"] == "tumor"].tolist()
        normal_ids = group.index[group["label"] == "normal"].tolist()
        # A patient can (rarely) have multiple tumor or normal samples (e.g.
        # multi-region sampling); pair the first of each and note it — being
        # explicit here avoids silently duplicating a patient's contribution.
        if tumor_ids and normal_ids:
            pairs.append(
                {
                    "patient_id": patient,
                    "cancer_type": cancer_type,
                    "tumor_sample_id": tumor_ids[0],
                    "normal_sample_id": normal_ids[0],
                }
            )
    return pd.DataFrame(pairs)


def batch_correct(
    expr_log: pd.DataFrame,
    meta: pd.DataFrame,
    batch_col: str = "study",
    covariate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Correct for batch effects (different studies/platforms) using ComBat.

    Why this matters: TCGA (RNA-seq) and a GEO validation set (often
    microarray) differ systematically in assay chemistry, dynamic range, and
    processing pipeline — differences that are usually much larger than the
    biological tumor/normal signal. If uncorrected, a classifier can learn to
    separate "study A vs. study B" instead of "tumor vs. normal", which
    inflates apparent performance on data it was trained on and collapses on
    truly independent data. ComBat (Johnson et al. 2007) is an
    empirical-Bayes method that models and removes additive + multiplicative
    batch effects while explicitly preserving a specified biological
    covariate (here, tumor/normal label), so it doesn't erase the signal
    you're trying to detect.

    IMPORTANT: only batch-correct within the *training* pool, or with the
    biological covariate protected, and NEVER batch-correct in a way that lets
    information leak from the held-out validation set into the correction of
    the training set (e.g. don't co-normalize train + independent-test data
    together). See `validation.py` for how the independent GEO set is handled
    instead (per-gene z-scoring against its own distribution, not ComBat
    against the training batches).

    Args:
        expr_log: log-transformed expression matrix, samples x genes.
        meta: sample metadata aligned to expr_log's index; must contain
            `batch_col` and 'label'.
        batch_col: metadata column identifying the batch/study.
        covariate_cols: additional biological covariates to preserve (beyond
            'label', which is always protected). E.g. ['cancer_type'].

    Returns:
        Batch-corrected expression matrix, same shape as expr_log.
    """
    meta = meta.loc[expr_log.index]
    batches = meta[batch_col]

    if batches.nunique() <= 1:
        # Nothing to correct for.
        return expr_log

    protected = ["label"] + (covariate_cols or [])
    try:
        # pyComBat's API: pycombat(data, batch, mod) where data is genes x
        # samples (note the transpose relative to our samples x genes
        # convention) and mod is a design matrix of covariates to preserve.
        from combat.pycombat import pycombat

        mod = pd.get_dummies(meta[protected], drop_first=True)
        corrected = pycombat(expr_log.T, batches, mod=mod)
        return corrected.T
    except ImportError:
        warnings.warn(
            "`combat` package not installed (pip install combat, or see "
            "requirements.txt for the pyComBat git install). Falling back to "
            "a simpler per-study mean-centering correction, which removes "
            "additive batch shifts but not the multiplicative/variance "
            "differences that full ComBat models. For a research write-up, "
            "install pyComBat (or run limma::removeBatchEffect / ComBat via "
            "rpy2) rather than relying on this fallback."
        )
        return _mean_center_by_batch(expr_log, batches)


def _mean_center_by_batch(expr_log: pd.DataFrame, batches: pd.Series) -> pd.DataFrame:
    """Fallback batch correction: subtract each batch's per-gene mean and add
    back the global per-gene mean, so batches are centered on a common origin.
    This only removes additive shifts; it does not equalize per-batch variance
    the way ComBat's empirical-Bayes shrinkage does.
    """
    global_mean = expr_log.mean(axis=0)
    corrected = expr_log.copy()
    for batch_name, idx in batches.groupby(batches).groups.items():
        batch_mean = expr_log.loc[idx].mean(axis=0)
        corrected.loc[idx] = expr_log.loc[idx] - batch_mean + global_mean
    return corrected


def load_and_preprocess(
    expr_path: str,
    meta_path: str,
    already_log_transformed: bool = False,
    correct_batches: bool = True,
) -> ExpressionDataset:
    """Convenience wrapper chaining load -> log-transform -> batch-correct.

    Args:
        expr_path: # FILL IN: path to expression matrix.
        meta_path: # FILL IN: path to sample metadata.
        already_log_transformed: set True if the source data is already on a
            log scale (skip re-log-transforming).
        correct_batches: whether to run ComBat/fallback across `study`.

    Returns:
        ExpressionDataset with a clean, log-scale, batch-corrected `expr`.
    """
    expr = load_expression_matrix(expr_path)
    meta = load_sample_metadata(meta_path)

    expr_log = expr if already_log_transformed else log_transform(expr)

    if correct_batches:
        expr_log = batch_correct(expr_log, meta.loc[expr_log.index])

    return ExpressionDataset(expr=expr_log, meta=meta)
