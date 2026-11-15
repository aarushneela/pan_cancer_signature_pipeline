"""
STEP 5: collect every number the paper still needs, from the output files the
earlier steps already wrote. Runs in a few seconds. It does NOT retrain anything.

Run from the inner pancancer_signature folder (same one as step2), venv on:
    python step5_collect_paper_numbers.py
To also save it to a text file (PowerShell):
    python step5_collect_paper_numbers.py | Tee-Object outputs\\paper_numbers.txt
Then paste the whole printed output back to me.

It reads:  outputs/step2_results.json, outputs/step3_results.json,
           outputs/step4_results.json, outputs/gene_table_tcga.csv
"""

import json
import os
import platform
import sys
from importlib import metadata

import numpy as np
import pandas as pd


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  (could not read {path}: {e})")
        return None


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():.3f} +/- {v.std():.3f}"


sym_map = {}
try:
    from step2_baseline_random_nested import ID2SYM
    sym_map.update(ID2SYM)
except Exception:
    pass


def S(g):
    g = str(g)
    return sym_map.get(g) or sym_map.get(g.split(".")[0]) or g


def header(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# --------------------------------------------------------------------- STEP 2
header("STEP 2: pool, nested, baselines, controls, true LOCO")
r2 = load("outputs/step2_results.json")
if r2:
    for top in ["200", "500", "1000", "None"]:
        p = r2.get(f"pool_top{top}")
        if p:
            print(f"  pool, top {top}: raw intersection {p['raw']}, proliferation-flagged {p['flagged']}, "
                  f"retained {len(p['retained'])}")
            if top == "200":
                print(f"     retained genes: {[S(g) for g in p['retained']]}")
    f5 = r2.get("fixed5")
    if f5:
        print(f"\n  fixed 5-gene (elastic net, all-data selection): AUC {ms(f5['auc'])}  acc {ms(f5['acc'])}")
        print(f"  same panel, PLAIN logistic regression (used for the random controls): "
              f"AUC {f5['plain_auc']:.3f}  acc {f5['plain_acc']:.3f}")
    n = r2.get("nested")
    if n:
        print(f"\n  NESTED 5-gene: AUC {ms(n['auc'])}  acc {ms(n['acc'])}")
        for i, s in enumerate(n["sets"], 1):
            print(f"     fold {i}: {[S(g) for g in s]}")
        counts = pd.Series([g for s in n["sets"] for g in s]).value_counts()
        print("     stability: " + ", ".join(f"{S(g)} {c}/5" for g, c in counts.items()))
    if r2.get("full"):
        print(f"\n  full transcriptome (top 5000 variable genes, chosen inside each fold, plain LR): "
              f"AUC {ms(r2['full']['auc'])}  acc {ms(r2['full']['acc'])}")
    if r2.get("prolif_only"):
        po = r2["prolif_only"]
        print(f"  proliferation-only (ALL 68 proliferation genes together, plain LR; NOT a 5-gene panel): "
              f"AUC {po['auc']:.3f}  acc {po['acc']:.3f}")
    plain = r2.get("fixed5", {}).get("plain_auc")
    pc = r2.get("pool_combos")
    if pc and plain is not None:
        a = np.array(pc["aucs"])
        print(f"\n  all {pc['n']} five-gene subsets of the pool: AUC min {a.min():.3f} / median {np.median(a):.3f} "
              f"/ max {a.max():.3f}; real panel rank {int((a > plain).sum()) + 1} of {pc['n']}")
    rr = r2.get("random_expressed")
    if rr and plain is not None:
        a = np.array(rr["aucs"])
        print(f"\n  {len(a)} random 5-gene sets (drawn from ALL genes with mean log2(FPKM+1) > 1, plain LR):")
        print(f"     AUC median {np.median(a):.3f}, 95th pct {np.percentile(a, 95):.3f}, "
              f"99th pct {np.percentile(a, 99):.3f}, max {a.max():.3f}; "
              f"sets >= real panel ({plain:.3f}): {(a >= plain).sum()}")
    if r2.get("loco_true"):
        print("\n  TRUE leave-one-cancer-out (our panel; the JSON stores AUC and accuracy only):")
        for r in r2["loco_true"]:
            print(f"     held out {r['held_out']}: pool {r['pool']}, genes {[S(g) for g in r['genes']]}, "
                  f"AUC {r['auc']:.3f}, acc {r['acc']:.3f}")

# --------------------------------------------------------------------- STEP 3
header("STEP 3: nested Stage 2 and the 4-class pipeline")
r3 = load("outputs/step3_results.json")
if r3:
    sym_map.update(r3.get("symbols", {}))
    labels4 = ["Normal", "BRCA", "LUAD", "COAD"]
    for v, d in r3["variants"].items():
        print(f"\n--- variant: {v} ---")
        accs = [r["test_acc"] for r in d["per_fold"]]
        print(f"  Stage 2 accuracy per fold: {[round(a, 3) for a in accs]}   mean {ms(accs)}")
        print(f"  training accuracy per fold: {[round(r['train_acc'], 3) for r in d['per_fold']]}")
        print(f"  combined 4-class accuracy: {d['combined_acc']:.4f}")
        cm = np.array(d["confusion"])
        print("  confusion matrix (rows = truth):")
        print(pd.DataFrame(cm, index=labels4, columns=labels4).to_string())
        print("  per-class accuracy:", {l: round(cm[i, i] / cm[i].sum(), 3) for i, l in enumerate(labels4)})
        n_norm = cm[0].sum()
        print(f"  normals called tumor: {n_norm - cm[0, 0]} of {n_norm}")
        print(f"  tumors called Normal: {cm[1:, 0].sum()} of {cm[1:].sum()}  "
              f"(by cancer: {dict(zip(labels4[1:], cm[1:, 0].tolist()))})")
        errs = [(f"{labels4[i]}->{labels4[j]}", int(cm[i, j])) for i in range(1, 4) for j in range(1, 4) if i != j]
        print("  type errors among tumors called tumor (largest first):", sorted(errs, key=lambda t: -t[1]))
        for r in d["per_fold"]:
            print(f"  fold {r['fold']} genes: {[S(g) for g in r['genes']]}")
        counts = pd.Series([g for r in d["per_fold"] for g in r["genes"]]).value_counts()
        print("  stability: " + ", ".join(f"{S(g)} {c}/5" for g, c in counts.items()))

# --------------------------------------------------------------------- STEP 4
header("STEP 4: proliferation-only comparison")
r4 = load("outputs/step4_results.json")
if r4:
    print("  Proliferation-only 5-gene panel, TRUE leave-one-cancer-out:")
    for r in r4["loco"]:
        print(f"     held out {r['held_out']}: genes {r['genes']}, AUC {r['auc']:.3f}, acc {r['acc']:.3f}")
    print("\n  Proliferation-only on GEO (AUC with 95% patient-bootstrap CI):")
    for r in r4["geo"]:
        print(f"     {r['dataset']}: best-5 {r['best5']}  AUC {r['auc5']:.3f} [{r['ci5'][0]:.3f}, {r['ci5'][1]:.3f}]   "
              f"| all {r['n_avail']} measurable: AUC {r['auc_all']:.3f} [{r['ci_all'][0]:.3f}, {r['ci_all'][1]:.3f}]")

# ------------------------------------------------------------------ GENE TABLE
header("Per-gene expression (mean log2(FPKM+1)), tumor vs normal")
try:
    gt = pd.read_csv("outputs/gene_table_tcga.csv")
    print(gt.round(2).to_string(index=False))
except Exception as e:
    print(f"  (could not read outputs/gene_table_tcga.csv: {e})")

# --------------------------------------------------------------- ENVIRONMENT
header("Environment (for Appendix H and Section 2.17)")
print("  Python:", sys.version.split()[0], "|", platform.platform())
for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "mygene", "GEOparse", "matplotlib", "joblib"]:
    try:
        print(f"  {pkg}=={metadata.version(pkg)}")
    except Exception:
        print(f"  {pkg}: not found")

print("\nDone. Paste everything above back to me.")
