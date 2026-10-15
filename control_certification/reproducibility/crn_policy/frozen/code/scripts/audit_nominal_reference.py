"""Recompute the new descriptive reference metric from immutable banks."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/pendulum_nominal_reference'
OLD=ROOT/'code/results/pendulum_regional_study/final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert sha(ROOT/'code/scripts/pendulum_nominal_reference.py')==protocol['source_sha256']
    for k,v in protocol['input_hashes'].items():assert sha(ROOT/k)==v
    for k,v in json.loads((OUT/'COMPLETE.json').read_text())['outputs'].items():assert sha(OUT/k)==v
    sums=[];rows=[r'\begin{tabular}{rrrrr}',r'\toprule',r'Seed & Actuator gain & Repair gain & Nominal gap & Fraction (\%) \\',r'\midrule']
    with np.load(OUT/'nominal_returns.npz') as nominal:
        for rec in summary['records']:
            seed=rec['seed']
            with np.load(OLD/f'seed{seed}/evaluation.npz') as data:r=data['returns']
            gain=float(np.mean(r[1]-r[0]));gap=float(np.mean(nominal[f'nominal_seed{seed}']-r[0]));sums.append([gain,gap])
            assert abs(gain-rec['mean_gain'])<1e-16 and abs(gap-rec['nominal_gap'])<1e-16
            rows.append(f'{seed} & {rec["actuator_gain"]:.4f} & {gain*1e5:.4f} & {gap*1e5:.4f} & {100*gain/gap:.2f} '+r'\\')
    totals=np.sum(sums,axis=0);rho=totals[0]/totals[1]
    assert abs(rho-summary['ratio_of_means'])<1e-14
    cfg=json.loads((OLD/'protocol.json').read_text())['config']
    ceiling=(cfg['R']-cfg['rho'])*(1-cfg['kappa'])/cfg['action_gain']
    assert cfg['d']<ceiling and abs(ceiling-summary['constant_displacement_tube_sufficient_upper'])<1e-15
    rows.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'manuscript/tables/v2_pendulum_nominal_reference.tex').write_text('\n'.join(rows)+'\n')
    record=dict(all_passed=True,input_hashes_verified=len(protocol['input_hashes']),reference_plants=8,ratio_of_sums=float(rho),
                audit_source_sha256=sha(Path(__file__)),original_adoption_outputs_unchanged=True)
    (OUT/'independent_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
