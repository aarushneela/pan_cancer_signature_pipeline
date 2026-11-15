"""
STEP 3: honest (nested) numbers for Stage 2 (cancer TYPE) and for the full
4-class pipeline (Normal / BRCA / LUAD / COAD).

Why: the earlier 98.2% (Stage 2) and 96.9% (combined) used genes that were
picked on ALL samples before cross-validation. Here, inside every training
fold we re-do BOTH gene selections from scratch:
    Stage 1: per-cancer DE -> 3-way intersection -> proliferation filter -> RFE to 5
    Stage 2: Kruskal-Wallis top 300 (tumor samples only) -> Random-Forest RFE to 5
              (run two ways: as first written, and with an expressed-genes filter)
then train both stages on the training fold and test on the untouched fold.

Also looks up the gene symbols of every gene that got selected (needs internet,
uses mygene like before).

Run from the same folder as step2 (it imports helper functions from it):
    python step3_nested_stage2_combined.py
Takes roughly 15-30 minutes. Paste the whole printed output back to me.
Output: outputs/step3_results.json
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"   # also silences the loky worker processes

import json
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_selection import _pipeline_importance_getter
from src.modeling import build_classifiers
from step2_baseline_random_nested import (
    ID2SYM,
    N_FOLDS,
    SEED,
    de_by_cancer,
    gidx,
    load_data,
    pool_from_de,
    rfe_select,
)

warnings.filterwarnings("ignore")


def rf_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)),
    ])


def kruskal_stat(X, labels, chunk=4000):
    """Kruskal-Wallis H per gene across the cancer-type groups (vectorized)."""
    classes = np.unique(labels)
    H = np.zeros(X.shape[1])
    for s in range(0, X.shape[1], chunk):
        sl = slice(s, s + chunk)
        h, _ = kruskal(*[X[labels == c][:, sl] for c in classes], axis=0)
        H[sl] = h
    return np.nan_to_num(H, nan=0.0)


STD_ID = re.compile(r"^ENSG\d{11}$")   # drops versioned / PAR_Y duplicate IDs like ENSG00000002586.1


def select_stage2(X, genes, labels, expressed_filter, top=300, k=5):
    """Kruskal-Wallis top-300 then Random-Forest RFE to k genes.
    expressed_filter=True first keeps only well-measured genes: a standard ID
    and median log2(FPKM+1) >= 1 in at least one cancer type (measurable by a
    real assay). expressed_filter=False is the version we ran first."""
    mask = np.ones(len(genes), dtype=bool)
    if expressed_filter:
        classes = np.unique(labels)
        med = np.vstack([np.median(X[labels == c], axis=0) for c in classes]).max(axis=0)
        mask = (med >= 1.0) & np.array([bool(STD_ID.match(g)) for g in genes])
    H = np.zeros(len(genes))
    H[mask] = kruskal_stat(X[:, mask], labels)
    cand = np.argsort(H)[-top:]
    rfe = RFE(rf_pipeline(), n_features_to_select=k, step=20, importance_getter=_pipeline_importance_getter)
    rfe.fit(X[:, cand], labels)
    return [str(genes[i]) for i, keep in zip(cand, rfe.support_) if keep], int(mask.sum())


def lookup_symbols(ids):
    try:
        import mygene
        base = {i: i.split(".")[0] for i in ids}
        hits = mygene.MyGeneInfo().querymany(sorted(set(base.values())), scopes="ensembl.gene",
                                             fields="symbol,name", species="human", verbose=False)
        by = {h["query"]: h.get("symbol", "?") for h in hits}
        return {i: by.get(b, "?") for i, b in base.items()}
    except Exception as e:
        print("(symbol lookup skipped:", e, ")")
        return {}


def main():
    t0 = time.time()
    os.makedirs("outputs", exist_ok=True)
    print("Loading TCGA...")
    X, genes, meta, prolif = load_data()
    y = (meta["label"] == "tumor").astype(int).to_numpy()
    groups = meta["patient_id"].to_numpy()
    ctype = meta["cancer_type"].to_numpy()
    cancers = sorted(set(ctype))
    final = np.where(y == 1, ctype, "Normal").astype(object)
    labels4 = ["Normal", "BRCA", "LUAD", "COAD"]
    VARIANTS = {"unfiltered": False, "expressed-genes filter": True}

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, final, groups=groups))

    preds = {v: np.empty(len(y), dtype=object) for v in VARIANTS}
    s1_sets = []
    per_fold = {v: [] for v in VARIANTS}
    for k, (tr, te) in enumerate(folds, 1):
        print(f"\n--- fold {k}/{N_FOLDS} ---")
        de = de_by_cancer(X, genes, meta, tr, cancers)
        pool, _, _ = pool_from_de(de, prolif, 200)
        g1 = rfe_select(X[tr], y[tr], genes, pool, 5)
        i1 = gidx(genes, g1)
        m1 = build_classifiers()["logreg_elasticnet"].fit(X[tr][:, i1], y[tr])
        s1_sets.append(g1)
        print(f"  Stage 1 genes: {[ID2SYM.get(g, g) for g in g1]}  (pool {len(pool)})")
        called_tumor = m1.predict(X[te][:, i1]) == 1

        tr_t = tr[y[tr] == 1]
        te_t = te[y[te] == 1]
        for v, flt in VARIANTS.items():
            g2, n_cand = select_stage2(X[tr_t], genes, ctype[tr_t], flt)
            i2 = gidx(genes, g2)
            m2 = rf_pipeline().fit(X[tr_t][:, i2], ctype[tr_t])
            tr_acc = accuracy_score(ctype[tr_t], m2.predict(X[tr_t][:, i2]))
            p_t = m2.predict(X[te_t][:, i2])
            acc = accuracy_score(ctype[te_t], p_t)
            dist = pd.Series(p_t).value_counts().to_dict()
            p = np.array(["Normal"] * len(te), dtype=object)
            if called_tumor.any():
                p[called_tumor] = m2.predict(X[te][called_tumor][:, i2])
            preds[v][te] = p
            per_fold[v].append({"fold": k, "genes": g2, "n_candidates": n_cand, "train_acc": tr_acc,
                                "test_acc": acc, "pred_dist": dist})
            print(f"  Stage 2 [{v}] ({n_cand} eligible genes): train acc {tr_acc:.3f}, TEST acc {acc:.3f}, "
                  f"predicted-type counts on test tumors {dist}")

    all_ids = sorted({g for s in s1_sets for g in s} | {g for v in VARIANTS for r in per_fold[v] for g in r["genes"]})
    sym = lookup_symbols(all_ids)
    def name(g):
        return ID2SYM.get(g) or sym.get(g, g)

    out = {"stage1_sets": s1_sets, "symbols": sym, "variants": {}}
    for v in VARIANTS:
        print("\n" + "=" * 70 + f"\nRESULTS, Stage 2 variant = {v} (all selection inside training folds)\n" + "=" * 70)
        accs = [r["test_acc"] for r in per_fold[v]]
        print("Per-fold Stage 2 test accuracy:", [round(a, 3) for a in accs])
        print(f"Stage 2 (type) accuracy given true tumors: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        acc = accuracy_score(final, preds[v])
        print(f"Combined 4-class accuracy: {acc:.4f}   (earlier, selection on all data: 0.969)")
        cm = confusion_matrix(final, preds[v], labels=labels4)
        print("\nConfusion matrix (rows = truth, columns = prediction):")
        print(pd.DataFrame(cm, index=labels4, columns=labels4))
        print("Per-class accuracy:", {lab: round(cm[i, i] / cm[i].sum(), 3) for i, lab in enumerate(labels4)})
        print("\nGenes chosen per fold:")
        for r in per_fold[v]:
            print(f"  fold {r['fold']}: {[name(g) for g in r['genes']]}")
        counts = pd.Series([g for r in per_fold[v] for g in r["genes"]]).value_counts()
        print("Stability (folds out of 5): " + ", ".join(f"{name(g)} {c}" for g, c in counts.items()))
        out["variants"][v] = {"per_fold": per_fold[v], "combined_acc": acc, "confusion": cm.tolist()}

    counts = pd.Series([g for s in s1_sets for g in s]).value_counts()
    print("\nStage 1 gene stability (folds out of 5): " + ", ".join(f"{name(g)} {c}" for g, c in counts.items()))

    with open("outputs/step3_results.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Paste everything above back to me.")


if __name__ == "__main__":
    main()
