"""
STEP 4: Is a proliferation-only 5-gene panel as good as ours when it has to
generalize? (Step 2 showed 68 proliferation genes match our panel INSIDE TCGA.)

Part A  Leave-one-cancer-out: for each held-out cancer, pick the best 5
        proliferation genes (RFE, on the other two cancers only), train there,
        test on the held-out cancer.
Part B  GEO: pick the best 5 proliferation genes on TCGA (among those the GEO
        platform actually measures), rescale GEO onto the TCGA scale (same
        method as before), report AUC with a patient-level bootstrap 95% CI.

Reference numbers from our pan-cancer panel (earlier runs) are printed beside
the proliferation numbers so you can read the comparison directly.

Run from the inner pancancer_signature folder (same one as step2), venv on:
    python step4_proliferation_baseline.py
Takes roughly 5-10 minutes. Paste the whole printed output back to me.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_loading import load_expression_matrix, load_sample_metadata
from src.modeling import build_classifiers
from src.validation import rescale_external_to_training_distribution
from step2_baseline_random_nested import fit_score, gidx, load_data, rfe_select

warnings.filterwarnings("ignore")

GEO_DATASETS = {
    "gse15852_brca": ("BRCA", "GPL96", "data/raw/geo_validation/gse15852_brca"),
    "gse32863_luad": ("LUAD", "GPL6884", "data/raw/geo_validation/gse32863_luad"),
    "gse8671_coad": ("COAD", "GPL570", "data/raw/geo_validation/gse8671_coad"),
}

# reference numbers from earlier runs of OUR panel (not recomputed here)
REF_LOCO = {"BRCA": 0.929, "LUAD": 0.998, "COAD": 0.918}                    # Step 2, true LOCO
REF_GEO = {"gse15852_brca": 0.654, "gse32863_luad": 0.989, "gse8671_coad": 0.830}  # Step 1, 2-gene model
N_BOOT = 2000


def ens_to_symbol(ids):
    import mygene
    hits = mygene.MyGeneInfo().querymany(list(ids), scopes="ensembl.gene", fields="symbol",
                                         species="human", verbose=False)
    return {h["query"]: h.get("symbol") for h in hits if h.get("symbol")}


def tokens(v):
    if not isinstance(v, str):
        return set()
    return {t.strip() for t in re.split(r"///|//|;|,|\|", v) if t.strip()}


def symbol_probe_map(annot: pd.DataFrame, symbols):
    """symbol -> probe list, matching ONLY the platform's primary symbol column."""
    probe_col = annot.columns[0]
    prim = [c for c in annot.columns if c in ("Gene Symbol", "Symbol", "ILMN_Gene", "GENE_SYMBOL")]
    want = set(symbols)
    out = {s: [] for s in symbols}
    for _, row in annot.iterrows():
        t = set()
        for c in prim:
            t |= tokens(row[c])
        for s in t & want:
            out[s].append(row[probe_col])
    return {s: p for s, p in out.items() if p}


def cluster_bootstrap_auc(y, score, clusters, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y, score, clusters = np.asarray(y), np.asarray(score), np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def geo_eval(Xtcga_df, y_tcga, Xgeo_df, ygeo, groups_geo, genes_sel):
    """Train elastic-net LR on TCGA with genes_sel, score GEO (rescaled). Returns (auc, lo, hi)."""
    mu, sd = Xtcga_df[genes_sel].mean(), Xtcga_df[genes_sel].std()
    model = build_classifiers()["logreg_elasticnet"].fit(Xtcga_df[genes_sel].to_numpy(), y_tcga)
    Xe = rescale_external_to_training_distribution(Xgeo_df[genes_sel], mu, sd)
    score = model.predict_proba(Xe[genes_sel].to_numpy())[:, 1]
    lo, hi = cluster_bootstrap_auc(ygeo, score, groups_geo)
    return roc_auc_score(ygeo, score), lo, hi


def main():
    print("Loading TCGA...")
    X, genes, meta, _ = load_data()
    y = (meta["label"] == "tumor").astype(int).to_numpy()
    ctype = meta["cancer_type"].to_numpy()
    cancers = sorted(set(ctype))
    prolif_path = "data/processed/proliferation_genes_ensembl.txt"
    with open(prolif_path) as f:
        prolif = [l.strip() for l in f if l.strip()]
    gset = set(genes)
    prolif = [g for g in prolif if g in gset]
    sym = ens_to_symbol(prolif)
    name = lambda g: sym.get(g, g)
    print(f"{len(prolif)} proliferation genes available in TCGA")

    # ------------------------------ Part A ------------------------------
    print("\n" + "=" * 70 + "\nPart A: proliferation-only 5-gene panel, TRUE leave-one-cancer-out\n" + "=" * 70)
    all_rows = np.arange(len(y))
    resA = []
    for held in cancers:
        tr = all_rows[ctype != held]
        te = all_rows[ctype == held]
        sel = rfe_select(X[tr], y[tr], genes, prolif, 5)
        idx = gidx(genes, sel)
        auc, acc = fit_score(build_classifiers()["logreg_elasticnet"], X[tr][:, idx], y[tr], X[te][:, idx], y[te])
        print(f"  held out {held}: genes {[name(g) for g in sel]}  AUC {auc:.3f}  acc {acc:.3f}   "
              f"| our pan-cancer panel (Step 2) AUC {REF_LOCO[held]:.3f}")
        resA.append({"held_out": held, "genes": [name(g) for g in sel], "auc": auc, "acc": acc})

    # ------------------------------ Part B ------------------------------
    print("\n" + "=" * 70 + "\nPart B: proliferation-only 5-gene panel on independent GEO data\n" + "=" * 70)
    import GEOparse
    resB = []
    for ds, (cancer, plat, folder) in GEO_DATASETS.items():
        try:
            gpl = GEOparse.get_GEO(geo=plat, destdir="geo_cache", silent=True)
            sp = symbol_probe_map(gpl.table, [name(g) for g in prolif])
            expr = load_expression_matrix(f"{folder}/expression_matrix.tsv")
            m = load_sample_metadata(f"{folder}/sample_metadata.tsv")
            m = m.loc[m.index.intersection(expr.index)]
            expr = expr.loc[m.index]
            cols = {}
            for s, probes in sp.items():
                ok = [p for p in probes if p in expr.columns]
                if ok:
                    cols[s] = expr[ok].mean(axis=1)
            Xgeo = pd.DataFrame(cols)
            avail = [g for g in prolif if name(g) in Xgeo.columns]
            if len(avail) < 5:
                print(f"  {ds}: only {len(avail)} proliferation genes on this platform, skipped")
                continue
            ens_idx = gidx(genes, avail)
            Xtc = pd.DataFrame(X[:, ens_idx], columns=[name(g) for g in avail])
            ygeo = (m["label"] == "tumor").astype(int).to_numpy()
            groups = m["patient_id"].to_numpy() if "patient_id" in m.columns else np.arange(len(m))

            sel5 = rfe_select(X, y, genes, avail, 5)       # selection uses TCGA only
            s5 = [name(g) for g in sel5]
            a5, l5, h5 = geo_eval(Xtc, y, Xgeo, ygeo, groups, s5)
            sall = [name(g) for g in avail]
            aa, la, ha = geo_eval(Xtc, y, Xgeo, ygeo, groups, sall)
            print(f"  {ds} ({cancer}, n={len(ygeo)}): {len(avail)} proliferation genes measurable")
            print(f"     best-5 {s5}: AUC {a5:.3f} [{l5:.3f}, {h5:.3f}]   | our 2-gene panel AUC {REF_GEO[ds]:.3f}")
            print(f"     all {len(avail)} proliferation genes: AUC {aa:.3f} [{la:.3f}, {ha:.3f}]")
            resB.append({"dataset": ds, "best5": s5, "auc5": a5, "ci5": [l5, h5], "n_avail": len(avail),
                         "auc_all": aa, "ci_all": [la, ha]})
        except Exception as e:  # keep going if one dataset trips
            print(f"  {ds}: FAILED ({type(e).__name__}: {e})")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/step4_results.json", "w") as f:
        json.dump({"loco": resA, "geo": resB}, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nDone. Paste everything above back to me.")


if __name__ == "__main__":
    main()
