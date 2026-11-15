"""
Quick peek at a GEO series matrix file's structure — prints the first ~80
lines with line numbers so we can find the sample-label metadata rows and
where the actual expression data table begins.

Usage: python peek_geo.py
"""

import gzip

path = "data/raw/geo_validation/gse8671_coad/GSE8671_series_matrix.txt.gz"  # FILL IN: change path to peek at other files

with gzip.open(path, "rt") as f:
    for i, line in enumerate(f):
        print(i, line[:150])
        if i > 80:
            break
