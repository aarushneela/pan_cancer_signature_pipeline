# Minimal Gene-Expression Signatures for Pan-Cancer Detection and Classification

**Veritas AI Hackathon 2026 · Track: Health**
**Team:** Aarush Neela

> **Prototype notice:** This project is a research prototype. It is not medical advice and is not a diagnostic tool. It was developed and tested only on public research datasets.

## Summary

Gene-expression data can reveal whether tissue is cancerous and what organ a tumor came from, but many studies use thousands of genes and report results on the same samples used to choose those genes. We asked: **can five genes or fewer separate tumor from normal tissue, and then classify breast (BRCA), lung (LUAD) and colorectal (COAD) cancer, while keeping the evaluation honest?**

We built a two-stage pipeline:

1. **Stage 1, detection:** tumor vs. normal using a five-gene panel (COL10A1, NME1, IQANK1, IQGAP3, HJURP).
2. **Stage 2, cancer type:** BRCA vs. LUAD vs. COAD using a separate five-gene panel.

The main point is the evaluation. Every gene-selection step was repeated inside each training fold, patients were kept on one side of every split, a held-out cancer was excluded from all selection steps, and results were compared against random, proliferation-only and 5,000-gene baselines.

## Key results

| Test | Result |
|---|---|
| Stage 1 detection, nested cross-validation (2,329 TCGA samples) | AUC **0.997 ± 0.001**, accuracy 98.7% ± 0.3% |
| 5,000-gene model | AUC 1.000 |
| 1,000 random five-gene sets | median AUC 0.875, best 0.990 |
| All 68 proliferation genes together | AUC 0.997 (matches the panel in TCGA) |
| Leave-one-cancer-out | AUC 0.998 (LUAD), 0.929 (BRCA), 0.918 (COAD) |
| External GEO test, two-gene model (COL10A1 + NME1) | AUC 0.989 (lung), 0.830 (colon adenoma), 0.654 (breast) |
| Stage 2 cancer type, nested | 97.5% ± 1.3% accuracy |
| Combined four-class pipeline (Normal, BRCA, LUAD, COAD) | 96.5% accuracy |

## What we found, including what did not work

- Very small panels keep most of the signal in this three-cancer setting.
- The detection task is easy: a model of all 68 proliferation genes matched our panel in TCGA, so **we do not claim the panel beats proliferation markers.**
- COL10A1 and NME1 were chosen in every nested fold. The other three panel genes were largely interchangeable with other genes in the pool.
- External validation was mixed. Breast AUC (0.654) was weak, and model probabilities did not transfer across platforms, only the ranking did.
- Stage 2 failed in 2 of 5 folds without an expressed-genes filter. We added the filter after seeing this, report both versions, and flag it as a post-hoc choice.

## What is new here

The idea of small gene panels is not new. What we add is a more careful test of them: nested selection to measure how much results are inflated, patient-grouped splits to prevent leakage, true leave-one-cancer-out testing, proliferation and random baselines, and external validation across three platforms with documented probe mapping.


## Interactive demo

A browser demo shows how the pipeline reasons as you move gene-expression sliders: https://claude.ai/artifact/Nw7y9mdrY2xuw8njBzJzME
It is an **illustrative approximation** built from average expression values, not the fitted classifiers, and is not independent validation.

## How to run

```bash
pip install -r requirements.txt
```

Data are **not** stored in this repository because of their size. Download them yourself:

- TCGA-BRCA, TCGA-LUAD, TCGA-COAD (STAR FPKM, GDC hub) from UCSC Xena: https://xenabrowser.net
- GEO series GSE15852, GSE32863, GSE8671: https://www.ncbi.nlm.nih.gov/geo/

Place downloads in `data/` and `geo_cache/`, then run the scripts in this order:

1. `convert_tcga.py`, `convert_geo.py`, `build_geo_probe_mapping.py`, `build_proliferation_list.py`
2. `step1_alias_bootstrap_table.py` (probe mapping, bootstrap intervals, per-gene table)
3. `step2_baseline_random_nested.py` (nested selection, baselines, leave-one-cancer-out)
4. `step3b_stage2_two_variants.py` (nested Stage 2 and four-class pipeline)
5. `step4_proliferation_baseline.py` (proliferation-only comparison)
6. `step5_collect_paper_numbers.py` (collects results into `outputs/paper_numbers.txt`)

Results are written to `outputs/`. Random seeds are fixed (cross-validation seed 0).

## Limitations

- Only three cancers; the classifier has no "other" class.
- Normal samples are tumor-adjacent tissue, not healthy-donor tissue.
- Single patient-grouped CV split; small external datasets (64 to 116 samples); colon validation used adenoma, not carcinoma.
- No batch-effect correction; not tested on RT-qPCR or formalin-fixed samples; not a blood test.
- Biological interpretations (for example COL10A1 and extracellular-matrix remodeling) are hypotheses, not tested findings.

## Credits

**Data**
- The Cancer Genome Atlas, via UCSC Xena: https://xenabrowser.net (Goldman et al., 2020)
- NCBI GEO: GSE15852, GSE32863, GSE8671: https://www.ncbi.nlm.nih.gov/geo/
- mygene.info for gene-ID mapping: https://mygene.info

**Software**
- Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, GEOparse, Matplotlib (versions in `requirements.txt`)

**AI tools**
- Claude (Anthropic), https://claude.ai: used to help write and debug analysis code, draft and edit the manuscript, build the interactive demo, and draft this write-up. The author designed the study, ran all analyses, checked the outputs, and is responsible for all results and interpretations. 

**Methods references:** PAM50 (Parker et al., 2009); CancerTYPE ID (Kerr et al., 2012); recursive feature elimination (Guyon et al., 2002); elastic net (Zou & Hastie, 2005); full list in the paper.

## Team

| Name | Email | School |
|---|---|---|
| Aarush Neela | aarushneela.3@gmail.com | Chirec |
