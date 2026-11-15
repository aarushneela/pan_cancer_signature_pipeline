"""
Stage 3 — Signature intersection across cancer types.

Takes each cancer type's list of significant, up-regulated genes (Stage 2) and
computes the genes recurrently up-regulated across ALL three cancer types —
the candidate "pan-cancer" signature. Then flags/removes genes that are likely
just markers of generic cell proliferation (present in essentially any rapidly
dividing tissue, cancerous or not) rather than cancer-specific biology, using a
user-supplied reference list.

Why the proliferation filter matters: genes like MKI67, PCNA, and TOP2A are
up-regulated in almost any high-turnover tissue (regenerating epithelium, bone
marrow, wound healing, etc.), not specifically in cancer. A "pan-cancer"
signature that's actually just a "cells are dividing a lot" signature would
overstate how cancer-specific the discriminative power is, which matters a lot
for a proof-of-concept about interpretable, biologically meaningful markers.
"""

from __future__ import annotations

import pandas as pd


def get_top_gene_sets(
    de_results_by_cancer: dict[str, pd.DataFrame],
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    top_n: int | None = 200,
) -> dict[str, set[str]]:
    """Extract each cancer type's significant up-regulated gene set.

    Args:
        de_results_by_cancer: cancer_type -> DE results (from
            `differential_expression.run_differential_expression_per_cancer`).
        padj_threshold, log2fc_threshold, top_n: passed straight through to
            `differential_expression.get_significant_upregulated_genes`.

    Returns:
        cancer_type -> set of gene symbols.
    """
    from .differential_expression import get_significant_upregulated_genes

    gene_sets = {}
    for cancer_type, de_results in de_results_by_cancer.items():
        sig = get_significant_upregulated_genes(
            de_results, padj_threshold=padj_threshold, log2fc_threshold=log2fc_threshold, top_n=top_n
        )
        gene_sets[cancer_type] = set(sig.index)
    return gene_sets


def intersect_signatures(gene_sets: dict[str, set[str]]) -> set[str]:
    """Compute the set of genes significant/up-regulated in every cancer type.

    Args:
        gene_sets: cancer_type -> set of gene symbols (from `get_top_gene_sets`).

    Returns:
        Set of genes common to all cancer types in `gene_sets`.
    """
    if not gene_sets:
        return set()
    sets = list(gene_sets.values())
    shared = sets[0]
    for s in sets[1:]:
        shared = shared & s
    return shared


def filter_proliferation_genes(
    gene_set: set[str],
    proliferation_genes: set[str],
) -> tuple[set[str], set[str]]:
    """Split a candidate signature into non-proliferation and proliferation-flagged genes.

    Args:
        gene_set: candidate pan-cancer signature (e.g. output of
            `intersect_signatures`).
        proliferation_genes: reference list of generic proliferation markers.
            # FILL IN: supply your curated list, e.g.
            #   {"MKI67", "PCNA", "TOP2A", "CCNB1", "CCNB2", "BUB1", "CDK1",
            #    "AURKA", "AURKB", "RRM2", "TYMS", ...}
            # A larger, published starting point is the Whitfield et al. 2002
            # cell-cycle gene set, or the proliferation-score gene list used in
            # PAM50 breast cancer subtyping.

    Returns:
        (retained, flagged) — retained is `gene_set` minus proliferation genes,
        flagged is the intersection (genes present in both), which you may
        still want to inspect/report even though they're excluded from the
        "cancer-specific" signature.
    """
    flagged = gene_set & proliferation_genes
    retained = gene_set - proliferation_genes
    return retained, flagged


def build_pancancer_signature(
    de_results_by_cancer: dict[str, pd.DataFrame],
    proliferation_genes: set[str],
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    top_n: int | None = 200,
) -> dict:
    """End-to-end Stage 3 convenience wrapper.

    Returns a dict with keys:
        'per_cancer_sets'      — cancer_type -> set of significant genes
        'raw_intersection'     — set, before proliferation filtering
        'proliferation_flagged'— set, genes removed as generic proliferation markers
        'final_signature'      — set, the candidate pan-cancer, non-proliferation signature
    """
    per_cancer_sets = get_top_gene_sets(
        de_results_by_cancer, padj_threshold=padj_threshold, log2fc_threshold=log2fc_threshold, top_n=top_n
    )
    raw_intersection = intersect_signatures(per_cancer_sets)
    final_signature, flagged = filter_proliferation_genes(raw_intersection, proliferation_genes)

    return {
        "per_cancer_sets": per_cancer_sets,
        "raw_intersection": raw_intersection,
        "proliferation_flagged": flagged,
        "final_signature": final_signature,
    }
