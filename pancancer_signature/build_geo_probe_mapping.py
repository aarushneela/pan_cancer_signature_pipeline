"""
Map your 5 final signature genes to probe IDs on each of the three GEO
platforms, using each platform's official annotation table (fetched live
via GEOparse from NCBI — not guessed or hardcoded).

Usage:
    pip install GEOparse
    python build_geo_probe_mapping.py
Output:
    Prints, for each platform, which of your 5 genes have a matching probe
    (and its probe ID), and which genes are MISSING from that platform —
    older arrays may simply not include every gene.
    Saves the mapping to data/processed/geo_probe_mapping.json for the next
    script to use.
"""

import json
import os

# Your final signature: gene symbol -> Ensembl ID (from run_pipeline_real.py's output)
FINAL_SIGNATURE = {
    "COL10A1": "ENSG00000123500",
    "NME1": "ENSG00000239672",
    "IQANK1": "ENSG00000203499",
    "IQGAP3": "ENSG00000183856",
    "HJURP": "ENSG00000123485",
}

# Which platform each validation dataset uses
DATASETS = {
    "gse15852_brca": "GPL96",
    "gse32863_luad": "GPL6884",
    "gse8671_coad": "GPL570",
}


def find_gene_symbol_column(annot_columns: list[str]) -> str:
    """Different platforms name their gene-symbol column slightly differently."""
    candidates = ["Gene Symbol", "gene_assignment", "Symbol", "ILMN_Gene", "GENE_SYMBOL"]
    for c in candidates:
        if c in annot_columns:
            return c
    raise ValueError(f"Couldn't find a gene symbol column among: {annot_columns}")


def map_platform(platform_id: str, cache_dir: str = "geo_cache") -> dict[str, list[str]]:
    """Fetch a platform's annotation and find probes for each signature gene.

    Returns:
        Dict of gene_symbol -> list of matching probe IDs (usually 0-3 probes
        per gene; a gene can appear on multiple probes targeting different
        parts of the transcript).
    """
    import GEOparse

    print(f"\nFetching platform annotation for {platform_id} (this downloads from NCBI, may take a minute)...")
    gpl = GEOparse.get_GEO(geo=platform_id, destdir=cache_dir, silent=True)
    annot = gpl.table

    symbol_col = find_gene_symbol_column(list(annot.columns))
    print(f"  Using column '{symbol_col}' for gene symbols ({len(annot)} probes total on this platform)")

    gene_to_probes = {}
    for gene_symbol in FINAL_SIGNATURE:
        # Affymetrix annotation sometimes lists multiple genes per probe,
        # separated by '///' (e.g. a probe hitting a shared exon), and some
        # probes have no gene assigned at all (NaN/float, not a string) —
        # check isinstance explicitly rather than trusting a blanket
        # .astype(str) conversion, which can leave stray non-string values
        # in edge cases.
        def _matches(value, target=gene_symbol):
            if not isinstance(value, str):
                return False
            return any(tok.strip() == target for tok in value.split("///"))

        mask = annot[symbol_col].apply(_matches)
        matches = annot.loc[mask, annot.columns[0]].tolist()  # first column is always the probe ID
        gene_to_probes[gene_symbol] = matches

    return gene_to_probes


def main():
    full_mapping = {}
    for dataset_name, platform_id in DATASETS.items():
        gene_to_probes = map_platform(platform_id)
        full_mapping[dataset_name] = {"platform": platform_id, "gene_to_probes": gene_to_probes}

        print(f"\n{dataset_name} ({platform_id}):")
        for gene, probes in gene_to_probes.items():
            if probes:
                print(f"  {gene}: found on probe(s) {probes}")
            else:
                print(f"  {gene}: NOT FOUND on this platform")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/geo_probe_mapping.json"
    with open(out_path, "w") as f:
        json.dump(full_mapping, f, indent=2)
    print(f"\nSaved mapping to {out_path}")


if __name__ == "__main__":
    main()