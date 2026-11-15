"""Look up gene symbols for the cancer-TYPE signature's Ensembl IDs."""
import mygene
 
ids = [
    "ENSG00000156510",
    "ENSG00000140297",
    "ENSG00000197142",
    "ENSG00000047936",
    "ENSG00000007952",
]
 
mg = mygene.MyGeneInfo()
results = mg.querymany(ids, scopes="ensembl.gene", fields="symbol,name", species="human")
for r in results:
    print(r.get("query"), "->", r.get("symbol", "NOT FOUND"), "-", r.get("name", ""))
 