"""Independent Riccati iteration, scalar-rollout audit, and table aggregation."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/pendulum_oracle_reference'
OLD=ROOT/'code/results/pendulum_regional_study/final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text());cfg=protocol['config']
    assert sha(ROOT/'code/scripts/pendulum_oracle_reference.py')==protocol['source_sha256']
    for k,v in protocol['input_hashes'].items():assert sha(ROOT/k)==v
    for k,v in json.loads((OUT/'COMPLETE.json').read_text())['outputs'].items():assert sha(OUT/k)==v
    table=[r'\begin{tabular}{rrrrr}',r'\toprule',r'Seed & Learned recovery (\%) & Capped LQR (\%) & Learned/LQR (\%) & Uncapped LQR (\%) \\',r'\midrule']
    matrix_error=return_error=0.;steps=0;totals={k:0. for k in protocol['kinds']};lg=gap=0.
    with np.load(OUT/'oracle_returns.npz') as bank:
        for row in summary['records']:
            g,c,seed=row['gain'],row['damping'],row['seed'];K=np.array(cfg['K']);T=np.array(cfg['T'])
            A=np.array([[1+.05*.75,.05*(1-.05*c)],[.75,1-.05*c]]);G=g*np.array([.0075,.15]);Q=np.diag([1.,.1]);S=Q.copy()
            for _ in range(2000):
                k=(G@S@A)/(.001+G@S@G);nextS=Q+A.T@S@(A-np.outer(G,k))
                if np.max(abs(nextS-S))<1e-13:S=nextS;break
                S=nextS
            kt=(G@S@A)/(.001+G@S@G);err=float(np.max(abs(kt-row['target_K'])));matrix_error=max(matrix_error,err);assert err<1e-10
            rng=np.random.default_rng([seed,4]);theta=rng.uniform(0,2*np.pi,4096);radius=np.sqrt(rng.uniform(size=4096))*.19
            starts=np.column_stack([radius*np.cos(theta),radius*np.sin(theta)])@np.linalg.inv(T).T
            noises=rng.uniform(-.03,.03,(40,4096))
            with np.load(OLD/f'seed{seed}/evaluation.npz') as old:returns=old['returns']
            for kind in protocol['kinds']:
                values=bank[f'{kind}_seed{seed}'];actual_gain=float(np.mean(values-returns[0]));totals[kind]+=actual_gain
                assert abs(actual_gain-row['references'][kind]['mean_gain'])<1e-16
                for i in range(64):
                    x=starts[i].copy();ret=0.
                    for t in range(40):
                        b=float(np.clip(-K@x,-2,2));res=float((K-kt)@x);h=.0125*min(1,float(np.sqrt((T@x)@(T@x)))/.19)
                        if kind=='capped_residual':u=b+max(-h,min(h,res))
                        elif kind=='capped_direction':u=b+h*np.sign(res)
                        else:u=float(-kt@x)
                        u=max(-2,min(2,u))
                        if kind.startswith('capped'):assert abs(u-b)<=h+1e-12
                        angle=(x[0]+np.pi)%(2*np.pi)-np.pi
                        ret-=angle*angle+.1*x[1]*x[1]+.001*u*u
                        v=np.clip((1-.05*c)*x[1]+.75*np.sin(x[0])+.15*g*u+.15*noises[t,i],-8,8)
                        x=np.array([x[0]+.05*v,v]);steps+=1
                    err=abs(ret-values[i]);return_error=max(return_error,err);assert err<1e-12
            lg+=row['learned_gain'];gap+=row['nominal_gap'];refs=row['references']
            table.append(f'{seed} & {100*row["learned_gain"]/row["nominal_gap"]:.4f} & {100*refs["capped_residual"]["recovery_fraction"]:.4f} & {100*refs["capped_residual"]["learned_over_reference_gain"]:.3f} & {100*refs["uncapped_target_lqr"]["recovery_fraction"]:.2f} '+r'\\')
    for k in protocol['kinds']:
        a=summary['aggregates'][k];assert abs(totals[k]/gap-a['recovery_fraction'])<1e-14
        assert abs(lg/totals[k]-a['learned_over_reference_gain'])<1e-14
    table.extend([r'\midrule',f'Aggregate & {100*lg/gap:.4f} & {100*totals["capped_residual"]/gap:.4f} & {100*lg/totals["capped_residual"]:.3f} & {100*totals["uncapped_target_lqr"]/gap:.2f} '+r'\\',r'\bottomrule',r'\end{tabular}'])
    (ROOT/'manuscript/tables/v2_pendulum_oracle_reference.tex').write_text('\n'.join(table)+'\n')
    result=dict(all_passed=True,source_and_input_hashes_verified=len(protocol['input_hashes'])+1,plants=8,reference_policies=3,
        independent_Riccati_max_gain_error=matrix_error,independent_scalar_transitions=steps,independent_scalar_return_max_error=float(return_error),
        learned_over_capped_LQR=float(lg/totals['capped_residual']),capped_over_uncapped_LQR=float(totals['capped_residual']/totals['uncapped_target_lqr']),
        audit_source_sha256=sha(Path(__file__)))
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
