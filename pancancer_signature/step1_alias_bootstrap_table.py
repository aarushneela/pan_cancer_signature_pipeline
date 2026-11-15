"""
STEP 1 of the core analyses. Three jobs in one script:

  Part A  Alias-aware probe re-mapping. Old microarray annotation uses old
          gene names, so exact-symbol matching missed IQANK1, HJURP, IQGAP3
          on some platforms. We ask mygene.info for each gene's aliases and
          search the annotation for those too. Every alias-based match is
          printed so YOU can eyeball it (aliases can be ambiguous).
  Part B  Re-run GEO validation with the new mapping, with bootstrap 95% CIs
          on AUC (resampling patients, not samples, because GEO samples are
          matched pairs) and per-gene AUCs.
  Part C  Per-gene expression table (mean log2 FPKM tumor vs normal per
          cancer) for the 10 panel genes + the unidentified ENSG00000175063.

Run from the project root, venv active:
    python step1_alias_bootstrap_table.py
Outputs:
    data/processed/geo_probe_mapping_v2.json
    outputs/geo_validation_bootstrap.csv
    outputs/gene_table_tcga.csv
Copy-paste the whole printed output back to me.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_stage6_validation import (
    GENE_ENSEMBL,
    GEO_DATASETS,
    load_geo_gene_expression,
    load_tcga_combined,
)
from src.data_loading import load_expression_matrix, load_sample_metadata
from src.modeling import build_classifiers
from src.validation import rescale_external_to_training_distribution

PLATFORMS = {
    "gse15852_brca": "GPL96",
    "gse32863_luad": "GPL6884",
    "gse8671_coad": "GPL570",
}

STAGE2_ENSEMBL = {
    "HKDC1": "ENSG00000156510",
    "GCNT3": "ENSG00000140297",
    "ACSL5": "ENSG00000197142",
    "ROS1": "ENSG00000047936",
    "NOX1": "ENSG00000007952",
}
UNKNOWN_ENSEMBL = {"ENSG00000175063": "ENSG00000175063"}

N_BOOT = 2000
SEED = 0


# ----------------------------------------------------------------------
# Part A: alias-aware mapping
# ----------------------------------------------------------------------
def fetch_aliases() -> dict[str, set[str]]:
    """symbol -> {symbol + aliases}, from mygene.info using the Ensembl IDs."""
    import mygene

    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        list(GENE_ENSEMBL.values()) + list(UNKNOWN_ENSEMBL.values()),
        scopes="ensembl.gene", fields="symbol,alias,name", species="human", verbose=False,
    )
    by_ens = {h["query"]: h for h in hits if "symbol" in h}
    out = {}
    for sym, ens in GENE_ENSEMBL.items():
        h = by_ens.get(ens, {})
        al = h.get("alias", [])
        al = [al] if isinstance(al, str) else al
        out[sym] = {sym, h.get("symbol", sym), *al}
    print("Aliases from mygene.info:")
    for s, a in out.items():
        print(f"  {s}: {sorted(a)}")
    unk = by_ens.get("ENSG00000175063", {})
    print(f"\nENSG00000175063 is: {unk.get('symbol')} ({unk.get('name')})\n")
    return out


def tokens(value) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {t.strip() for t in re.split(r"///|//|;|,|\|", value) if t.strip()}


def map_platform_aliases(platform_id: str, aliases: dict[str, set[str]], cache_dir="geo_cache"):
    import GEOparse

    gpl = GEOparse.get_GEO(geo=platform_id, destdir=cache_dir, silent=True)
    annot = gpl.table
    probe_col = annot.columns[0]
    cols = [c for c in annot.columns
            if any(k in c.lower() for k in ("symbol", "synonym", "gene_assignment", "alias"))]
    print(f"\n{platform_id}: searching columns {cols} ({len(annot)} probes)")

    # A probe's PRIMARY symbol column is trusted. The synonyms column is only
    # trusted when the primary symbol is empty. If a probe already has a
    # different official symbol, matching on a synonym (e.g. "NB", "NBS")
    # grabs the WRONG gene, so we reject it and print it for review.
    primary_cols = [c for c in cols if c in ("Symbol", "Gene Symbol", "ILMN_Gene", "GENE_SYMBOL")]
    syn_cols = [c for c in cols if c not in primary_cols]

    result = {}
    for sym, names in aliases.items():
        probes, notes, rejected = [], [], []
        for _, row in annot.iterrows():
            prim = set()
            for c in primary_cols:
                prim |= tokens(row[c])
            syn = set()
            for c in syn_cols:
                syn |= tokens(row[c])
            if prim & names:
                probes.append(row[probe_col])
                notes.append(f"{row[probe_col]} via primary {sorted(prim & names)}")
            elif syn & names:
                if not prim:
                    probes.append(row[probe_col])
                    notes.append(f"{row[probe_col]} via synonym {sorted(syn & names)} (no primary symbol)")
                else:
                    rejected.append(f"{row[probe_col]} synonym {sorted(syn & names)} but primary symbol is {sorted(prim)}")
        result[sym] = {"probes": probes, "notes": notes}
        tag = "FOUND" if probes else "not found"
        print(f"  {sym}: {tag}  {notes[:6]}")
        if rejected:
            print(f"     REJECTED (belongs to another gene): {rejected[:6]}")
    return result


def build_mapping() -> dict:
    aliases = fetch_aliases()
    mapping = {}
    for ds, plat in PLATFORMS.items():
        res = map_platform_aliases(plat, aliases)
        mapping[ds] = {
            "platform": plat,
            "gene_to_probes": {g: v["probes"] for g, v in res.items()},
            "matched_via": {g: v["notes"] for g, v in res.items()},
        }
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/geo_probe_mapping_v2.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print("\nSaved data/processed/geo_probe_mapping_v2.json")
    return mapping


# ----------------------------------------------------------------------
# Part B: GEO validation with bootstrap CIs
# ----------------------------------------------------------------------
def cluster_bootstrap_auc(y, score, clusters, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate_gene_set(genes, ds_name, cancer, folder, mapping):
    g2p = mapping[ds_name]["gene_to_probes"]
    if any(not g2p.get(g) for g in genes):
        return None
    X_train, y_train = load_tcga_combined([GENE_ENSEMBL[g] for g in genes])
    X_train.columns = genes
    model = build_classifiers()["logreg_elasticnet"]
    model.fit(X_train, y_train)

    X_ext = load_geo_gene_expression(folder, g2p)[genes]
    meta = load_sample_metadata(f"{folder}/sample_metadata.tsv")
    meta = meta.loc[meta.index.intersection(X_ext.index)]
    X_ext = X_ext.loc[meta.index]
    y = (meta["label"] == "tumor").astype(int).to_numpy()
    X_ext = rescale_external_to_training_distribution(X_ext, X_train.mean(axis=0), X_train.std(axis=0))

    score = model.predict_proba(X_ext)[:, 1]
    if "patient_id" in meta.columns:
        clusters = meta["patient_id"].to_numpy()
    else:
        clusters = np.arange(len(meta))
    auc = roc_auc_score(y, score)
    lo, hi = cluster_bootstrap_auc(y, score, clusters)

    single = {}
    for g in genes:  # per-gene AUC on external data (raw direction: higher = tumor)
        single[g] = round(roc_auc_score(y, X_ext[g]), 3)
    return {"auc": auc, "lo": lo, "hi": hi, "n": len(y), "n_clusters": len(set(clusters)), "single_gene_auc": single}


def run_geo(mapping):
    rows = []
    for ds_name, (cancer, folder) in GEO_DATASETS.items():
        g2p = mapping[ds_name]["gene_to_probes"]
        avail = [g for g in GENE_ENSEMBL if g2p.get(g)]
        sets = {"2-gene (COL10A1+NME1)": ["COL10A1", "NME1"], f"all available ({len(avail)})": avail}
        print(f"\n{'=' * 70}\n{ds_name} ({cancer}): genes available after alias mapping = {avail}\n{'=' * 70}")
        for label, genes in sets.items():
            r = evaluate_gene_set(genes, ds_name, cancer, folder, mapping)
            if r is None:
                continue
            print(f"  {label} {genes}: AUC {r['auc']:.3f}  95% CI [{r['lo']:.3f}, {r['hi']:.3f}]  "
                  f"n={r['n']} ({r['n_clusters']} patients)")
            print(f"     single-gene AUCs: {r['single_gene_auc']}")
            rows.append({"dataset": ds_name, "cancer": cancer, "gene_set": label, "genes": ",".join(genes),
                         "auc": r["auc"], "ci_lo": r["lo"], "ci_hi": r["hi"], "n": r["n"]})
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/geo_validation_bootstrap.csv", index=False)
    print("\nSaved outputs/geo_validation_bootstrap.csv")


# ----------------------------------------------------------------------
# Part C: per-gene expression table in TCGA
# ----------------------------------------------------------------------
def gene_table():
    allg = {**GENE_ENSEMBL, **STAGE2_ENSEMBL, "ENSG00000175063 (UBE2C?)": "ENSG00000175063"}
    rows = []
    for cancer, folder in {"BRCA": "data/raw/tcga_brca", "LUAD": "data/raw/tcga_luad", "COAD": "data/raw/tcga_coad"}.items():
        expr = load_expression_matrix(f"{folder}/expression_matrix.tsv")
        meta = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        meta = meta.loc[meta.index.intersection(expr.index)]
        for name, ens in allg.items():
            if ens not in expr.columns:
                continue
            t = expr.loc[meta.index[meta["label"] == "tumor"], ens]
            n = expr.loc[meta.index[meta["label"] == "normal"], ens]
            rows.append({"gene": name, "cancer": cancer, "tumor_mean": t.mean(), "normal_mean": n.mean(),
                         "log2FC(T-N)": t.mean() - n.mean(), "n_tumor": len(t), "n_normal": len(n)})
    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/gene_table_tcga.csv", index=False)
    print("\nPer-gene expression table (mean log2(FPKM+1)):")
    print(df.round(2).to_string(index=False))
    print("\nSaved outputs/gene_table_tcga.csv")


if __name__ == "__main__":
    mapping = build_mapping()
    run_geo(mapping)
    gene_table()