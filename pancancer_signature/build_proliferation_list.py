"""
Build a proper proliferation/cell-cycle gene reference list, converting
well-established gene SYMBOLS into Ensembl Gene IDs via a live lookup
(mygene.info) rather than hand-typed ID guesses.

Why symbols instead of hardcoded Ensembl IDs: gene symbols like "MKI67" or
"CCNB1" are stable, well-known, and easy to verify against any cancer biology
reference. Ensembl ID numbers (ENSG00000...) are much easier to mistype or
misremember. Looking the IDs up live, right before you use them, means the
mapping is verifiable and reproducible rather than trusted from memory.

The gene list below is a standard, widely-used proliferation/cell-cycle gene
set spanning G1/S, S, G2, and G2/M phases (in the spirit of the classic
Whitfield et al. 2002 cell-cycle signature) — much broader than the original
10-gene starter list, and includes several mitotic-spindle genes (ANLN, TPX2,
TROAP among them) that your final signature happened to contain.

Usage:
    pip install mygene
    python build_proliferation_list.py
Output:
    data/processed/proliferation_genes_ensembl.txt — one Ensembl ID per line
    Also prints any symbols that failed to map, so you can sanity-check them.
"""

import os

# A standard, broad proliferation / cell-cycle gene symbol set.
# FILL IN / EXTEND: add more if your literature review turns up others.
PROLIFERATION_GENE_SYMBOLS = [
    # Core proliferation markers
    "MKI67", "PCNA",
    # DNA replication / S-phase
    "TOP2A", "RRM1", "RRM2", "TYMS", "TK1", "MCM2", "MCM3", "MCM4", "MCM5",
    "MCM6", "MCM7", "CDC6", "CDT1", "GINS1", "GINS2", "PRIM1", "POLE2",
    # Cyclins / CDKs
    "CCNA2", "CCNB1", "CCNB2", "CCNE1", "CCNE2", "CDK1", "CDK2", "CDK4", "CDK6",
    # Mitotic checkpoint / spindle assembly
    "BUB1", "BUB1B", "BUB3", "MAD2L1", "AURKA", "AURKB", "PLK1", "TTK",
    "TPX2", "ANLN", "TROAP", "ASPM", "NUSAP1", "PRC1", "KIF2C", "KIF23",
    "KIF11", "KIF20A", "CENPA", "CENPE", "CENPF", "NDC80", "NUF2", "SPC25",
    # Cytokinesis / mitotic exit
    "CDC20", "CDC25A", "CDC25C", "UBE2C", "BIRC5", "PTTG1",
    # Transcription factors driving the cell cycle
    "FOXM1", "E2F1", "E2F2", "MYBL2",
    # DNA damage response / repair (often co-expressed with proliferation)
    "RAD51", "RAD54L", "EXO1", "BRCA1", "BRCA2", "FEN1",
]


def build_ensembl_proliferation_list(output_path: str = "data/processed/proliferation_genes_ensembl.txt"):
    """Convert the symbol list above into Ensembl Gene IDs via mygene.info."""
    try:
        import mygene
    except ImportError:
        raise ImportError(
            "The 'mygene' package is required for this lookup. Install it with:\n"
            "    pip install mygene\n"
            "(mygene queries the mygene.info API — a live, citable gene database — "
            "rather than relying on a hardcoded ID list.)"
        )

    mg = mygene.MyGeneInfo()
    print(f"Looking up {len(PROLIFERATION_GENE_SYMBOLS)} gene symbols via mygene.info...")
    results = mg.querymany(
        PROLIFERATION_GENE_SYMBOLS,
        scopes="symbol",
        fields="ensembl.gene",
        species="human",
    )

    ensembl_ids = set()
    failed_symbols = []
    for r in results:
        if "notfound" in r or "ensembl" not in r:
            failed_symbols.append(r["query"])
            continue
        ensembl_field = r["ensembl"]
        # mygene can return either a single dict or a list of dicts if a
        # symbol maps to multiple Ensembl IDs (rare, but happens for a few
        # genes with historical ID splits/merges).
        if isinstance(ensembl_field, list):
            for entry in ensembl_field:
                ensembl_ids.add(entry["gene"])
        else:
            ensembl_ids.add(ensembl_field["gene"])

    print(f"\nMapped {len(ensembl_ids)} Ensembl IDs from {len(PROLIFERATION_GENE_SYMBOLS)} symbols.")
    if failed_symbols:
        print(f"WARNING: {len(failed_symbols)} symbol(s) failed to map — check these manually: {failed_symbols}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for ensembl_id in sorted(ensembl_ids):
            f.write(ensembl_id + "\n")
    print(f"Saved to {output_path}")

    return ensembl_ids


if __name__ == "__main__":
    build_ensembl_proliferation_list()
