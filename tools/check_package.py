"""Run a fast metadata and run-count audit; never initiates optimization."""
from pathlib import Path
import csv, json, hashlib, zipfile, sys

ROOT=Path(__file__).resolve().parents[1]
CITIES={"Afyon":(14,5,2,4,10000,435,280,9),"Aydin":(31,5,3,5,18900,245,140,9),
        "Canakkale":(37,5,2,5,10000,325,125,9),"Isparta":(48,5,2,6,10000,240,75,5)}
PROFILES=["Equal-Priority","Tiered","Demand-Correlated","Negative-Correlated","Hyper-Critical"]
fail=[]; warnings=[]; counts={"input_files":0,"anchors":0,"production_workbooks":0,"curves":0,"benchmark_workbooks":0}

def need(path,key=None):
    if not path.is_file(): fail.append(f"Missing: {path.relative_to(ROOT)}");return False
    if key:counts[key]+=1
    return True

for city,(n,m,ng,nv,L,W,Q,k) in CITIES.items():
    for ext in ["customers.csv","depots.csv","distance.csv","duration.csv","params.csv"]:
        need(ROOT/'Inputs'/f"{city}_{ext}","input_files")
    for profile in PROFILES:
        af=(f"{city}_Tcheby_Anchor_Validation_parallel.xlsx" if profile=="Equal-Priority"
            else f"{city}_{profile}_Tcheby_Anchor_Validation_parallel.xlsx")
        need(ROOT/'Outputs'/af,"anchors")
        stem=f"{city}_{profile}_Tcheby_Production"
        book=ROOT/'Outputs'/'Tcheby_Production'/f"{stem}.xlsx"
        curve=ROOT/'Outputs'/'Tcheby_Production'/f"{stem}_Curve.csv"
        if need(book,"production_workbooks"):
            with zipfile.ZipFile(book) as z:
                for part in ("xl/workbook.xml","[Content_Types].xml"):
                    if part not in z.namelist():fail.append(f"Malformed XLSX: {book.name} missing {part}")
        if need(curve,"curves"):
            with curve.open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            got={float(r['Lambda']) for r in rows if r.get('Lambda') not in (None,'')}
            expected={0.0,1.0}|({0.1,0.3,0.5,0.7,0.9} if city=='Isparta' else {i/10 for i in range(1,10)})
            if got!=expected:fail.append(f"Lambda grid mismatch: {city}/{profile} {sorted(got)}")

for method in ['baseline','extended']:
    folder=ROOT/'benchmarks'/'results'/method
    files=sorted(folder.glob('Merged_Analysis_*.xlsx'))
    counts['benchmark_workbooks']+=len(files)
    if len(files)!=10:fail.append(f"Expected 10 {method} merged benchmark workbooks; got {len(files)}")

need(ROOT/'code'/'LLRP_CORE_PARALLEL_HPC.py')
need(ROOT/'docs'/'current_manuscript.pdf')
need(ROOT/'MANIFEST_SHA256.csv')
print(json.dumps({"status":"PASS" if not fail else "FAIL", "counts":counts, "issues":fail,"notes":warnings},indent=2))
sys.exit(0 if not fail else 1)
