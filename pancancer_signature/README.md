# Pan-Cancer Minimal Gene Signature Pipeline

Identify a minimal (≤5 gene) pan-cancer expression signature that separates tumor
from matched normal tissue in breast (BRCA), lung (LUAD), and colon (COAD) cancer,
and test how much discriminative power a tiny gene set retains relative to the full
transcriptome.

---

## 1. Where to get the data

### 1a. Training data — TCGA (BRCA / LUAD / COAD)

You need **matched tumor + adjacent-normal** samples, so use a source that keeps
that pairing explicit and gives you a plain samples × genes matrix (avoid raw BAM/FASTQ
— you don't need to run an alignment pipeline for this project).

| Source | What you get | Why use it |
|---|---|---|
| **GDC Data Portal** – https://portal.gdc.cancer.gov/ | Official TCGA RNA-seq (HTSeq counts / FPKM / TPM) per project (`TCGA-BRCA`, `TCGA-LUAD`, `TCGA-COAD`), plus clinical/sample metadata (`sample_type`: "Primary Tumor" vs "Solid Tissue Normal"). | Authoritative source; use the GDC "Repository" filter by project + data category "Transcriptome Profiling" + workflow "STAR - Counts". |
| **UCSC Xena** – https://xenabrowser.net/ | Pre-merged TCGA cohorts as flat gene-by-sample TSVs, already gene-symbol indexed, with a matching phenotype/clinical TSV. Also has a combined **TCGA TARGET GTEx** hub with unified batch-normalized values across studies. | Easiest for this project — no need to parse GDC's per-file JSON manifest; download one TSV per cohort and go straight to `data_loading.py`. |
| **cBioPortal** – https://www.cbioportal.org/ | TCGA PanCancer Atlas studies (`brca_tcga_pan_can_atlas_2018`, `luad_tcga_pan_can_atlas_2018`, `coadread_tcga_pan_can_atlas_2018`) with bulk download of mRNA expression + clinical attributes via the "Download" tab or the cBioPortal API/`cbioportalR` / `bravado` client. | Good if you also want mutation/CNV data later; slightly fewer normal samples than GDC. |
| **Recount3 / recount2** – https://rna.recount.bio/ | Uniformly re-processed TCGA RNA-seq counts (same pipeline across all projects), which sidesteps batch effects introduced by different processing pipelines. | Best choice if batch effects between cancer types worry you more than batch effects between studies. |

Practical tip: TCGA has far fewer matched normals than tumors (e.g. BRCA ≈ 1100
tumor vs. ~110 normal; COAD/LUAD similarly skewed). Keep only samples with
`sample_type` == Primary Tumor or Solid Tissue Normal, and where possible restrict to
patients who have **both** (true matched pairs) for the differential expression step —
`data_loading.py` below has a helper for this.

### 1b. Independent validation data — GEO

Use a **different platform and a different study** than TCGA so that a signature
that "works" is truly generalizing, not just fitting one pipeline's normalization
quirks. Three real, matched tumor/normal series (one per cancer type) confirmed via
literature review, all deposited in NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/):

| Cancer type | Suggested GEO series | Platform | Design |
|---|---|---|---|
| Breast | **GSE15852** | Affymetrix HG-U133A (GPL96) | 43 breast tumors with paired adjacent normal tissue |
| Lung | **GSE32863** | Illumina HumanWG-6 v3 (GPL6884) | 58 lung adenocarcinoma tumor / 58 paired adjacent normal |
| Colon | **GSE8671** | Affymetrix HG-U133 Plus 2 (GPL570) | 32 colorectal tumor / 32 paired adjacent normal |

Download via the GEO web UI ("Download family" → Series Matrix File, or supplementary
raw CEL/IDAT files) or programmatically with `GEOparse` (Python) or Bioconductor's
`GEOquery` (R). Series matrix files come pre-normalized per-platform; if you pull raw
CEL/IDAT files you'll need platform-specific preprocessing (RMA for Affymetrix, etc.)
before they're comparable to TCGA's log2(FPKM+1)/log2(TPM+1) scale — this is exactly
what the batch-correction step in `data_loading.py` is for.

Other useful general-purpose resources if you want to swap datasets later:
- **TNMplot** (https://tnmplot.com/) — a web tool that already aggregates GEO + TCGA
  tumor/normal/metastasis comparisons per gene, good for a quick sanity check of a
  candidate gene before committing it to the signature.
- **GDC portal search** across projects if you want to add more cancer types later.

### 1c. Reference gene lists you'll need to supply

- A **proliferation marker gene list** (you mentioned MKI67, PCNA, TOP2A — add more
  from a curated proliferation signature, e.g. the "proliferation" arm of the PAM50 or
  the Whitfield cell-cycle gene set) — used to flag/exclude generic proliferation
  signal from the pan-cancer intersection.
- Optionally, a **housekeeping gene list** (e.g. ACTB, GAPDH, B2M) if you want an
  internal normalization sanity check.

---

## 2. How to store the data (repo layout)

```
pancancer_signature/
├── README.md                     <- this file
├── requirements.txt
├── data/
│   ├── raw/                      <- put untouched downloads here, one subfolder per source
│   │   ├── tcga_brca/
│   │   │   ├── expression_matrix.tsv     # samples x genes, from GDC/Xena/cBioPortal
│   │   │   └── sample_metadata.tsv       # sample_id, label, cancer_type, study, patient_id
│   │   ├── tcga_luad/
│   │   ├── tcga_coad/
│   │   └── geo_validation/
│   │       ├── gse15852_brca/
│   │       ├── gse32863_luad/
│   │       └── gse8671_coad/
│   └── processed/                <- pipeline writes normalized / batch-corrected matrices here
│       ├── train_expression_log2.parquet
│       ├── train_metadata.parquet
│       └── validation_expression_log2.parquet
├── src/                          <- pipeline code (this deliverable)
│   ├── data_loading.py
│   ├── differential_expression.py
│   ├── signature_intersection.py
│   ├── feature_selection.py
│   ├── modeling.py
│   ├── cross_cancer.py
│   ├── validation.py
│   └── interpretability.py
├── outputs/                      <- figures, ranked gene tables, trained model artifacts
└── run_pipeline_example.py       <- synthetic end-to-end smoke test (run this first)
```

**Expected file formats** (CSV/TSV, `data_loading.py` handles either):

`expression_matrix.tsv` — rows = samples, columns = genes, first column = `sample_id`:

| sample_id | GENE1 | GENE2 | ... |
|---|---|---|---|
| TCGA-A1-0001-01 | 12.4 | 3.1 | ... |

`sample_metadata.tsv` — one row per `sample_id`:

| sample_id | label | cancer_type | study | patient_id |
|---|---|---|---|---|
| TCGA-A1-0001-01 | tumor | BRCA | TCGA | TCGA-A1-0001 |
| TCGA-A1-0001-11 | normal | BRCA | TCGA | TCGA-A1-0001 |

`patient_id` is what lets `data_loading.py` identify **matched** tumor/normal pairs
(same patient, different `sample_id` suffix) versus unpaired samples.

Everywhere in the code below that needs one of these real paths/lists, it's marked
`# FILL IN:`.

---

## 3. Running the code

```bash
pip install -r requirements.txt
python run_pipeline_example.py     # synthetic data, sanity-checks the whole pipeline
```

Once that runs cleanly end to end, point `data_loading.py`'s loader functions at your
real TCGA files (see paths above) and re-run stage by stage — each `src/*.py` module
can also be imported and used interactively/in a notebook.
