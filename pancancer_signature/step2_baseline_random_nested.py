"""
STEP 2 of the core analyses (Stage 1: tumor vs normal). Answers the questions
a judge or reviewer will ask:

  Q1  "Does 5 genes lose much versus using the whole transcriptome?"
        -> full-transcriptome baseline (top 5,000 most variable genes chosen
           INSIDE each training fold, L2 logistic regression).
  Q2  "Are your 5 genes special, or would any 5 genes work?"
        -> (a) ALL possible 5-gene subsets of the 12-gene candidate pool,
           (b) 1,000 random 5-gene sets from expressed genes genome-wide,
           (c) a model on the ~70 proliferation genes.
  Q3  "Your numbers are optimistic: you picked the genes using all samples."
        -> NESTED selection: the whole pipeline (per-cancer DE, 3-way
           intersection, proliferation filter, RFE to 5) is re-run from
           scratch inside every training fold, then tested on the untouched
           fold. Also a TRUE leave-one-cancer-out where the held-out cancer
           is never used for gene selection.
  Q4  "Does the result depend on the top-200 cutoff?"
        -> pool size and membership at top-N = 200 / 500 / 1000 / no cutoff.

Run from the project root (the folder containing src/ and data/), venv on:
    python step2_baseline_random_nested.py
Takes roughly 10-25 minutes. Paste the printed summary back to me.
Outputs: outputs/step2_results.json, outputs/step2_random_control.png
"""

import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from src.data_loading import load_expression_matrix, load_sample_metadata
from src.feature_selection import _pipeline_importance_getter
from src.modeling import build_classifiers

warnings.filterwarnings("ignore")

TCGA_COHORTS = {"BRCA": "data/raw/tcga_brca", "LUAD": "data/raw/tcga_luad", "COAD": "data/raw/tcga_coad"}
PROLIF_PATH = "data/processed/proliferation_genes_ensembl.txt"

# The published 5-gene detection panel (selected on ALL data, so its CV is optimistic)
FINAL5 = {
    "COL10A1": "ENSG00000123500",
    "NME1": "ENSG00000239672",
    "IQANK1": "ENSG00000203499",
    "IQGAP3": "ENSG00000183856",
    "HJURP": "ENSG00000123485",
}
ID2SYM = {v: k for k, v in FINAL5.items()}
ID2SYM["ENSG00000175063"] = "UBE2C"

N_FOLDS = 5
SEED = 0
N_RANDOM = 1000
N_VAR_GENES = 5000


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data():
    exprs, metas = [], []
    for name, folder in TCGA_COHORTS.items():
        e = load_expression_matrix(f"{folder}/expression_matrix.tsv").astype(np.float32)
        m = load_sample_metadata(f"{folder}/sample_metadata.tsv")
        shared = e.index.intersection(m.index)
        exprs.append(e.loc[shared])
        metas.append(m.loc[shared])
    expr = pd.concat(exprs, axis=0, join="inner")
    meta = pd.concat(metas, axis=0).loc[expr.index]
    with open(PROLIF_PATH) as f:
        prolif = {l.strip() for l in f if l.strip()}
    return expr.to_numpy(dtype=np.float32), np.array(expr.columns), meta, prolif


# ----------------------------------------------------------------------
# Fast per-cancer differential expression (same test as src/, vectorized)
# ----------------------------------------------------------------------
def fast_de(Xt, Xn, genes, chunk=4000):
    """Mann-Whitney U per gene (asymptotic, tie-corrected) + BH FDR + log2FC."""
    pv = np.ones(Xt.shape[1])
    for s in range(0, Xt.shape[1], chunk):
        sl = slice(s, s + chunk)
        _, p = mannwhitneyu(Xt[:, sl], Xn[:, sl], axis=0, method="asymptotic")
        pv[sl] = p
    pv = np.nan_to_num(pv, nan=1.0)
    lfc = Xt.mean(axis=0, dtype=np.float64) - Xn.mean(axis=0, dtype=np.float64)
    padj = multipletests(pv, method="fdr_bh")[1]
    df = pd.DataFrame({"log2FC": lfc, "pvalue": pv, "padj": padj}, index=genes)
    return df.sort_values(["padj", "log2FC"], ascending=[True, False])


def de_by_cancer(X, genes, meta, rows, cancers):
    """rows = integer row indices (training rows). Returns cancer -> DE table."""
    ct = meta["cancer_type"].to_numpy()[rows]
    lab = meta["label"].to_numpy()[rows]
    out = {}
    for c in cancers:
        t = rows[(ct == c) & (lab == "tumor")]
        n = rows[(ct == c) & (lab == "normal")]
        out[c] = fast_de(X[t], X[n], genes)
    return out


def pool_from_de(de_dict, prolif, top_n):
    sets = {}
    for c, d in de_dict.items():
        sig = d[(d["padj"] < 0.05) & (d["log2FC"] > 1.0)]
        if top_n is not None:
            sig = sig.head(top_n)
        sets[c] = set(sig.index)
    inter = set.intersection(*sets.values()) if sets else set()
    return sorted(inter - prolif), sorted(inter & prolif), sorted(inter)


# ----------------------------------------------------------------------
# Modeling helpers
# ----------------------------------------------------------------------
def plain_lr():
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def rfe_select(X, y, genes, pool, k=5):
    """RFE (elastic-net logistic regression) from `pool` down to k genes."""
    if len(pool) <= k:
        return list(pool)
    idx = np.array([np.where(genes == g)[0][0] for g in pool])
    step = 1 if len(pool) <= 30 else 5
    rfe = RFE(build_classifiers()["logreg_elasticnet"], n_features_to_select=k, step=step,
              importance_getter=_pipeline_importance_getter)
    rfe.fit(X[:, idx], y)
    return [g for g, keep in zip(pool, rfe.support_) if keep]


def fit_score(model, Xtr, ytr, Xte, yte):
    m = clone(model).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), accuracy_score(yte, m.predict(Xte))


def cv_small(Xs, y, folds, model=None):
    """CV on a small pre-sliced matrix. Returns (mean AUC, mean acc) over folds."""
    model = model or plain_lr()
    r = [fit_score(model, Xs[tr], y[tr], Xs[te], y[te]) for tr, te in folds]
    return float(np.mean([a for a, _ in r])), float(np.mean([b for _, b in r]))


def summarize(vals):
    a = np.array(vals)
    return f"{a.mean():.3f} +/- {a.std():.3f}"


def gidx(genes, ids):
    lookup = {g: i for i, g in enumerate(genes)}
    return np.array([lookup[i] for i in ids])


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs("outputs", exist_ok=True)
    print("Loading TCGA (BRCA+LUAD+COAD)...")
    X, genes, meta, prolif = load_data()
    y = (meta["label"] == "tumor").astype(int).to_numpy()
    groups = meta["patient_id"].to_numpy()
    cancers = sorted(meta["cancer_type"].unique())
    print(f"{X.shape[0]} samples x {X.shape[1]} genes; tumor={y.sum()}, normal={(1 - y).sum()}")
    prolif_in = [g for g in prolif if g in set(genes)]

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y, groups=groups))
    all_rows = np.arange(X.shape[0])
    results = {}

    # ---------------- Q4 + candidate pool on FULL data ----------------
    print("\n" + "=" * 70 + "\nQ4: candidate pool vs top-N cutoff (full data; the original analysis)\n" + "=" * 70)
    de_full = de_by_cancer(X, genes, meta, all_rows, cancers)
    pools = {}
    for top_n in (200, 500, 1000, None):
        kept, flagged, raw = pool_from_de(de_full, prolif, top_n)
        pools[top_n] = kept
        tag = "no cutoff" if top_n is None else f"top {top_n}"
        in_pool = [s for s, e in FINAL5.items() if e in kept]
        print(f"  {tag:>9}: raw intersection {len(raw)}, proliferation-flagged {len(flagged)}, "
              f"retained {len(kept)}; final-5 genes inside retained pool: {in_pool}")
        if top_n == 200:
            print(f"            flagged: {[ID2SYM.get(g, g) for g in flagged]}")
            print(f"            retained pool ids: {kept}")
        results[f"pool_top{top_n}"] = {"raw": len(raw), "flagged": len(flagged), "retained": kept}
    pool200 = pools[200]

    # ---------------- fixed 5-gene panel (selection-biased) ----------------
    print("\n" + "=" * 70 + "\nFixed 5-gene panel, patient-grouped CV (selected on ALL data -> optimistic)\n" + "=" * 70)
    fixed_idx = gidx(genes, list(FINAL5.values()))
    X5 = X[:, fixed_idx]
    fixed_auc, fixed_acc = [], []
    for tr, te in folds:
        a, b = fit_score(build_classifiers()["logreg_elasticnet"], X5[tr], y[tr], X5[te], y[te])
        fixed_auc.append(a); fixed_acc.append(b)
    print(f"  elastic-net LR: AUC {summarize(fixed_auc)}  accuracy {summarize(fixed_acc)}")
    real_plain_auc, real_plain_acc = cv_small(X5, y, folds)
    print(f"  plain LR (used for the random controls below): AUC {real_plain_auc:.3f}  accuracy {real_plain_acc:.3f}")
    results["fixed5"] = {"auc": fixed_auc, "acc": fixed_acc, "plain_auc": real_plain_auc, "plain_acc": real_plain_acc}

    # ---------------- Q3: NESTED selection ----------------
    print("\n" + "=" * 70 + "\nQ3: NESTED selection (DE + intersection + proliferation filter + RFE inside each training fold)\n" + "=" * 70)
    nested_auc, nested_acc, nested_sets = [], [], []
    for k, (tr, te) in enumerate(folds, 1):
        de = de_by_cancer(X, genes, meta, tr, cancers)
        kept, _, _ = pool_from_de(de, prolif, 200)
        sel = rfe_select(X[tr], y[tr], genes, kept, 5)
        idx = gidx(genes, sel)
        a, b = fit_score(build_classifiers()["logreg_elasticnet"], X[tr][:, idx], y[tr], X[te][:, idx], y[te])
        nested_auc.append(a); nested_acc.append(b); nested_sets.append(sel)
        print(f"  fold {k}: pool={len(kept)} -> selected {[ID2SYM.get(g, g) for g in sel]}  AUC {a:.3f}  acc {b:.3f}")
    print(f"  NESTED 5-gene: AUC {summarize(nested_auc)}  accuracy {summarize(nested_acc)}")
    counts = pd.Series([g for s in nested_sets for g in s]).value_counts()
    print("  Gene selection stability (folds out of 5 in which each gene was chosen):")
    for g, c in counts.items():
        print(f"     {ID2SYM.get(g, g):>16}: {c}/5")
    results["nested"] = {"auc": nested_auc, "acc": nested_acc, "sets": nested_sets}

    # ---------------- Q1: full-transcriptome baseline ----------------
    print("\n" + "=" * 70 + f"\nQ1: full-transcriptome baseline (top {N_VAR_GENES} most variable genes picked inside each fold)\n" + "=" * 70)
    full_auc, full_acc = [], []
    for k, (tr, te) in enumerate(folds, 1):
        var = X[tr].var(axis=0)
        top = np.argsort(var)[-N_VAR_GENES:]
        a, b = fit_score(plain_lr(), X[tr][:, top], y[tr], X[te][:, top], y[te])
        full_auc.append(a); full_acc.append(b)
        print(f"  fold {k}: AUC {a:.3f}  acc {b:.3f}")
    print(f"  FULL-TRANSCRIPTOME: AUC {summarize(full_auc)}  accuracy {summarize(full_acc)}")
    results["full"] = {"auc": full_auc, "acc": full_acc}

    # ---------------- proliferation-only ----------------
    print("\n" + "=" * 70 + f"\nProliferation-only model ({len(prolif_in)} cell-cycle genes)\n" + "=" * 70)
    pidx = gidx(genes, prolif_in)
    pa, pb = cv_small(X[:, pidx], y, folds)
    print(f"  proliferation-only: AUC {pa:.3f}  accuracy {pb:.3f}")
    results["prolif_only"] = {"auc": pa, "acc": pb}

    # ---------------- Q2: random / exhaustive controls ----------------
    print("\n" + "=" * 70 + "\nQ2: are the 5 genes special? (plain logistic regression, same folds)\n" + "=" * 70)
    # (a) exhaustive: every 5-subset of the 12-gene pool
    if 5 < len(pool200) <= 16:
        combos = list(itertools.combinations(pool200, 5))
        pidx_pool = {g: i for i, g in enumerate(pool200)}
        Xpool = X[:, gidx(genes, pool200)]
        res = Parallel(n_jobs=-1)(
            delayed(cv_small)(Xpool[:, [pidx_pool[g] for g in c]], y, folds) for c in combos)
        aucs_pool = np.array([r[0] for r in res]); accs_pool = np.array([r[1] for r in res])
        real_in_pool = all(e in pool200 for e in FINAL5.values())
        print(f"  (a) all {len(combos)} five-gene subsets of the {len(pool200)}-gene pool: "
              f"AUC min {aucs_pool.min():.3f} / median {np.median(aucs_pool):.3f} / max {aucs_pool.max():.3f}; "
              f"accuracy median {np.median(accs_pool):.3f} / max {accs_pool.max():.3f}")
        if real_in_pool:
            rank = int((aucs_pool > real_plain_auc).sum()) + 1
            print(f"      real panel AUC {real_plain_auc:.3f} ranks {rank} of {len(combos)} subsets "
                  f"({(aucs_pool >= real_plain_auc).mean() * 100:.1f}% of subsets are at least as good)")
        else:
            print(f"      (real panel is not fully inside the pool at top-200: "
                  f"{[s for s, e in FINAL5.items() if e not in pool200]} missing)")
        results["pool_combos"] = {"n": len(combos), "aucs": aucs_pool.tolist(), "accs": accs_pool.tolist()}
    else:
        aucs_pool = None
        print(f"  (a) skipped: pool size {len(pool200)} not in 6..16")

    # (b) random 5-gene sets from expressed genes genome-wide
    rng = np.random.default_rng(SEED)
    expressed = np.where(X.mean(axis=0) > 1.0)[0]
    sets = [rng.choice(expressed, 5, replace=False) for _ in range(N_RANDOM)]
    res = Parallel(n_jobs=-1)(delayed(cv_small)(X[:, s], y, folds) for s in sets)
    aucs_rand = np.array([r[0] for r in res]); accs_rand = np.array([r[1] for r in res])
    print(f"  (b) {N_RANDOM} random 5-gene sets from {len(expressed)} expressed genes: "
          f"AUC median {np.median(aucs_rand):.3f}, 95th pct {np.percentile(aucs_rand, 95):.3f}, "
          f"99th pct {np.percentile(aucs_rand, 99):.3f}, max {aucs_rand.max():.3f}")
    print(f"      accuracy median {np.median(accs_rand):.3f}, 95th pct {np.percentile(accs_rand, 95):.3f}, max {accs_rand.max():.3f}")
    print(f"      sets with AUC >= real panel ({real_plain_auc:.3f}): {(aucs_rand >= real_plain_auc).sum()} of {N_RANDOM}")
    results["random_expressed"] = {"aucs": aucs_rand.tolist(), "accs": accs_rand.tolist()}

    # ---------------- true leave-one-cancer-out ----------------
    print("\n" + "=" * 70 + "\nTRUE leave-one-cancer-out: held-out cancer is NOT used for gene selection\n" + "=" * 70)
    loco = []
    for held in cancers:
        train_rows = all_rows[meta["cancer_type"].to_numpy() != held]
        test_rows = all_rows[meta["cancer_type"].to_numpy() == held]
        train_c = [c for c in cancers if c != held]
        de = de_by_cancer(X, genes, meta, train_rows, train_c)
        kept, _, _ = pool_from_de(de, prolif, 200)
        sel = rfe_select(X[train_rows], y[train_rows], genes, kept, 5)
        idx = gidx(genes, sel)
        a, b = fit_score(build_classifiers()["logreg_elasticnet"], X[train_rows][:, idx], y[train_rows],
                         X[test_rows][:, idx], y[test_rows])
        print(f"  held out {held}: pool={len(kept)} selected {[ID2SYM.get(g, g) for g in sel]}  AUC {a:.3f}  acc {b:.3f}")
        loco.append({"held_out": held, "pool": len(kept), "genes": sel, "auc": a, "acc": b})
    results["loco_true"] = loco

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(aucs_rand, bins=40, color="#9aa5b1", label="random 5-gene sets (expressed genes)")
        if aucs_pool is not None:
            ax[0].hist(aucs_pool, bins=40, color="#5b8def", alpha=0.7, label="all 5-subsets of the pool")
        ax[0].axvline(real_plain_auc, color="#d1495b", lw=2, label="published 5-gene panel")
        ax[0].set_xlabel("patient-grouped CV AUC"); ax[0].set_ylabel("number of gene sets"); ax[0].legend(fontsize=8)
        names = ["Fixed 5-gene\n(optimistic)", "Nested 5-gene\n(honest)", f"Full transcriptome\n(top {N_VAR_GENES})", "Proliferation\nonly"]
        vals = [np.mean(fixed_auc), np.mean(nested_auc), np.mean(full_auc), pa]
        ax[1].bar(names, vals, color=["#9aa5b1", "#d1495b", "#5b8def", "#9aa5b1"])
        ax[1].set_ylim(min(vals) - 0.02, 1.0); ax[1].set_ylabel("CV AUC")
        for i, v in enumerate(vals):
            ax[1].text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout(); plt.savefig("outputs/step2_random_control.png", dpi=200)
        print("\nSaved outputs/step2_random_control.png")
    except Exception as e:  # figure is optional
        print("Figure skipped:", e)

    with open("outputs/step2_results.json", "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    print("\n" + "=" * 70 + "\nSUMMARY (patient-grouped 5-fold CV)\n" + "=" * 70)
    print(f"  Fixed 5-gene (selected on all data, optimistic): AUC {np.mean(fixed_auc):.3f}  acc {np.mean(fixed_acc):.3f}")
    print(f"  NESTED 5-gene (honest):                          AUC {np.mean(nested_auc):.3f}  acc {np.mean(nested_acc):.3f}")
    print(f"  Full transcriptome (top {N_VAR_GENES} var genes):          AUC {np.mean(full_auc):.3f}  acc {np.mean(full_acc):.3f}")
    print(f"  Proliferation-only ({len(prolif_in)} genes):                AUC {pa:.3f}  acc {pb:.3f}")
    print(f"  Random 5 expressed genes: median AUC {np.median(aucs_rand):.3f}, 95th pct {np.percentile(aucs_rand, 95):.3f}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Paste everything above back to me.")


if __name__ == "__main__":
    main()
