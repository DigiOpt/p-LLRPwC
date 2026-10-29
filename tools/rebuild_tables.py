"""Read archived CSV files and rebuild compact manuscript-oriented comparisons.
No exact solves and no heuristic runs are performed.
"""
from pathlib import Path
import csv, math, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'derived_tables';OUT.mkdir(exist_ok=True)
CITIES=['Afyon','Aydin','Canakkale','Isparta']
PROFILES=['Equal-Priority','Tiered','Demand-Correlated','Negative-Correlated','Hyper-Critical']

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def num(x):
    try: return float(x)
    except (TypeError,ValueError):return math.nan

def save(name,records,columns):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(records)
    print('Wrote',OUT/name,len(records),'rows')

records=[];anchor_records=[]
for city in CITIES:
    rows=read(ROOT/'Outputs'/'Routing_Cost_Analysis'/city/f'{city}_All_Profiles_Routing_Cost_Analysis.csv')
    for r in rows:
        profile=r.get('Priority_Profile','')
        if profile not in PROFILES:continue
        lam=num(r.get('Lambda'))
        if math.isnan(lam):continue
        rec={'City':city,'Profile':profile,'Lambda':lam,'Financial_Cost':r.get('Financial_Cost'),
           'Travel_Cost':r.get('Routing_Cost'), 'Priority_Weighted_Latency':r.get('Priority_Weighted_Latency'),
           'Travel_Cost_Change_pct':r.get('Routing_Cost_Increase_pct'),
           'Latency_Reduction_pct':r.get('Latency_Reduction_pct'),
           'Dominated_Reported':r.get('Dominated'),'Routing_Dominated_Reported':r.get('Routing_Dominated')}
        records.append(rec)
        if lam in (0.0,1.0):anchor_records.append(rec)
cols=['City','Profile','Lambda','Financial_Cost','Travel_Cost','Priority_Weighted_Latency','Travel_Cost_Change_pct','Latency_Reduction_pct','Dominated_Reported','Routing_Dominated_Reported']
save('preference_paths_all_available.csv',records,cols)
save('anchor_route_summaries.csv',anchor_records,cols)
shared=[r for r in records if r['Lambda'] in {0.0,0.1,0.3,0.5,0.7,0.9,1.0}]
save('preference_paths_common_grid.csv',shared,cols)
# Table 4 / Appendix I input is preserved and machine-readable, not reconstructed from unpublished raw experiments.
from shutil import copyfile
for fname in ['ablation_summary.csv','ablation_instance.csv','benchmark_robustness.csv']:
    copyfile(ROOT/'benchmarks'/'ablation'/'analysis'/fname,OUT/fname)
print('Note: These are stored-output derived summaries, not regenerated GRASP results or exact optimality proofs.')
