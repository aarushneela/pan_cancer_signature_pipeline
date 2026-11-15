"""
Stage 2 — Differential expression analysis, per cancer type.

For each cancer type separately, test every gene for tumor-vs-normal
differential expression, compute log2 fold-change, and correct for multiple
testing (Benjamini-Hochberg FDR). Doing this per cancer type (rather than
pooling all three) is deliberate: it avoids cancer-type identity itself
dominating the comparison, and it lets Stage 3 explicitly ask "which genes are
recurrently tumor-associated across independent tissue contexts?" rather than
just "which genes best split the pooled dataset."

Two nonparametric/parametric options are provided:
    - Welch's t-test (parametric; assumes approx. normal log-expression,
      does not assume equal variance between groups).
    - Mann-Whitney U (nonparametric; robust to outliers and non-normality,
      recommended default for small sample sizes or skewed data even after
      log-transform).

Note on limma/DESeq2: these are the field-standard tools for RNA-seq/microarray
DE analysis (limma's moderated t-statistic borrows information across genes to
stabilize variance estimates in small samples; DESeq2 models raw counts with a
negative binomial GLM). Neither has a mature pure-Python equivalent as of this
writing. If you want that extra statistical power, the recommended path is
calling R via `rpy2` (sketch provided in `run_limma_via_rpy2` below, commented
out since it requires an R installation with limma). The t-test/Mann-Whitney
approach here is a reasonable, dependency-light substitute for a
proof-of-concept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def differential_expression_one_cancer(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    method: str = "mannwhitney",
    matched_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run tumor-vs-normal DE testing for genes within a single cancer type.

    Args:
        expr: log-scale expression matrix (samples x genes) restricted to one
            cancer type's samples.
        meta: metadata aligned to expr.index with a 'label' column
            ('tumor'/'normal').
        method: 'ttest' (Welch's t-test, unpaired), 'mannwhitney' (unpaired
            Mann-Whitney U, default), or 'paired_wilcoxon' (requires
            `matched_pairs`, uses within-patient tumor-normal differences —
            the statistically strongest option when matched pairs are
            available; see `data_loading.find_matched_pairs`).
        matched_pairs: required if method == 'paired_wilcoxon'; DataFrame with
            'tumor_sample_id' / 'normal_sample_id' columns for this cancer type.

    Returns:
        DataFrame indexed by gene with columns:
            'log2FC'   — mean(tumor) - mean(normal) in log2 space, i.e. the
                         log2 fold-change of tumor relative to normal.
            'pvalue'   — raw p-value from the chosen test.
            'padj'     — Benjamini-Hochberg FDR-adjusted p-value.
        Sorted by padj ascending, then |log2FC| descending.
    """
    genes = expr.columns
    results = {"log2FC": [], "pvalue": [], "gene": []}

    if method == "paired_wilcoxon":
        if matched_pairs is None or matched_pairs.empty:
            raise ValueError("paired_wilcoxon requires a non-empty matched_pairs DataFrame")
        tumor_expr = expr.loc[matched_pairs["tumor_sample_id"]]
        normal_expr = expr.loc[matched_pairs["normal_sample_id"]]
        for gene in genes:
            diff = tumor_expr[gene].to_numpy() - normal_expr[gene].to_numpy()
            diff = diff[~np.isnan(diff)]
            if len(diff) < 3 or np.allclose(diff, 0):
                pval, lfc = 1.0, np.nanmean(diff) if len(diff) else 0.0
            else:
                # Wilcoxon signed-rank test on within-patient differences —
                # the paired analogue of Mann-Whitney U.
                _, pval = stats.wilcoxon(diff)
                lfc = np.mean(diff)
            results["gene"].append(gene)
            results["log2FC"].append(lfc)
            results["pvalue"].append(pval)
    else:
        tumor_mask = meta["label"] == "tumor"
        normal_mask = meta["label"] == "normal"
        tumor_expr = expr.loc[tumor_mask]
        normal_expr = expr.loc[normal_mask]

        for gene in genes:
            t_vals = tumor_expr[gene].dropna().to_numpy()
            n_vals = normal_expr[gene].dropna().to_numpy()
            if len(t_vals) < 2 or len(n_vals) < 2:
                pval = 1.0
            elif method == "ttest":
                # Welch's t-test: does not assume equal variance between
                # tumor and normal groups, which is a safer default than
                # Student's t-test since tumors are frequently more
                # heterogeneous (higher variance) than normal tissue.
                _, pval = stats.ttest_ind(t_vals, n_vals, equal_var=False)
            elif method == "mannwhitney":
                # Rank-based test; robust to the heavy-tailed / non-normal
                # distributions common in expression data, and to a handful
                # of outlier samples.
                _, pval = stats.mannwhitneyu(t_vals, n_vals, alternative="two-sided")
            else:
                raise ValueError(f"Unknown method: {method}")

            results["gene"].append(gene)
            results["log2FC"].append(np.mean(t_vals) - np.mean(n_vals) if len(t_vals) and len(n_vals) else 0.0)
            results["pvalue"].append(pval)

    df = pd.DataFrame(results).set_index("gene")
    # Benjamini-Hochberg controls the false discovery rate across the many
    # thousands of genes tested simultaneously — essential here, since at
    # alpha=0.05 uncorrected you'd expect ~5% of ~20,000 genes (~1000 genes)
    # to look "significant" by chance alone.
    df["padj"] = multipletests(df["pvalue"].fillna(1.0), method="fdr_bh")[1]
    return df.sort_values(["padj", "log2FC"], ascending=[True, False])


def run_differential_expression_per_cancer(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    method: str = "mannwhitney",
    matched_pairs_by_cancer: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Loop `differential_expression_one_cancer` over each cancer type.

    Args:
        expr: full log-scale expression matrix across all cancer types.
        meta: full metadata, must contain 'cancer_type' and 'label'.
        method: see `differential_expression_one_cancer`.
        matched_pairs_by_cancer: optional dict cancer_type -> matched_pairs
            DataFrame (only needed for method='paired_wilcoxon').

    Returns:
        Dict mapping cancer_type -> DE results DataFrame (see
        `differential_expression_one_cancer`).
    """
    results = {}
    for cancer_type, sub_meta in meta.groupby("cancer_type"):
        sub_expr = expr.loc[sub_meta.index]
        pairs = (matched_pairs_by_cancer or {}).get(cancer_type)
        results[cancer_type] = differential_expression_one_cancer(sub_expr, sub_meta, method=method, matched_pairs=pairs)
    return results


def get_significant_upregulated_genes(
    de_results: pd.DataFrame,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Filter one cancer type's DE table down to significant, up-regulated genes.

    "Up-regulated in tumor" (positive log2FC) is the biologically relevant
    direction for a screening signature: genes that switch ON in cancer are
    easier to build a robust positive-detection assay around than genes that
    switch off (down-regulation is more platform/normalization-sensitive).

    Args:
        de_results: output of `differential_expression_one_cancer`.
        padj_threshold: FDR cutoff (default 5%).
        log2fc_threshold: minimum log2 fold-change (default 1.0 = 2-fold
            higher in tumor).
        top_n: if set, keep only the top_n genes by padj after filtering
            (useful to bound the size of the pre-intersection gene lists in
            Stage 3).

    Returns:
        Filtered, still-sorted DataFrame.
    """
    sig = de_results[(de_results["padj"] < padj_threshold) & (de_results["log2FC"] > log2fc_threshold)]
    if top_n is not None:
        sig = sig.head(top_n)
    return sig


# ---------------------------------------------------------------------------
# Optional: calling limma via rpy2, if you have R + Bioconductor's limma
# installed and want the moderated-t / empirical-Bayes variance shrinkage
# that outperforms a plain t-test in small samples. Left commented out since
# it introduces an R dependency; uncomment and adapt if you go this route.
# ---------------------------------------------------------------------------
#
# def run_limma_via_rpy2(expr: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
#     import rpy2.robjects as ro
#     from rpy2.robjects import pandas2ri
#     from rpy2.robjects.packages import importr
#     pandas2ri.activate()
#     limma = importr("limma")
#     base = importr("base")
#     stats_r = importr("stats")
#
#     r_expr = pandas2ri.py2rpy(expr.T)  # limma expects genes x samples
#     design = stats_r.model_matrix(ro.Formula("~ label"), data=pandas2ri.py2rpy(labels.to_frame()))
#     fit = limma.lmFit(r_expr, design)
#     fit = limma.eBayes(fit)
#     top_table = limma.topTable(fit, coef=2, number=ro.r("Inf"))
#     return pandas2ri.rpy2py(top_table)
