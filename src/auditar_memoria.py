"""Reproduce table data and exact checks. Run from repo root with .venv/bin/python src/auditar_memoria.py.
Timings: three uninstrumented solver calls, excluding loading and validation.
"""
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import sympy
from sympy import Matrix, eye
from sympy.matrices.normalforms import hermite_normal_form
from utils.loader import cargar_instancia_ggh, cargar_instancia_lwe
from ggh.tier1_solver import resolver_ggh_tier1
from ggh.tier2_solver import resolver_ggh_tier2
from ggh.tier3_solver import resolver_ggh_tier3
from lwe.solver_lwe_lll_timer import parse_lwe, primal_lll, bit_from_secret

ROOT = Path(__file__).resolve().parents[1]

def row_hnf(rows):
    return hermite_normal_form(Matrix(rows).T).T

def exact_membership(v, generators):
    H = row_hnf(generators)
    z = Matrix([list(v)]) * H.inv()
    return all(x.q == 1 for x in z)

def trace_timer_primal(A, b, q):
    """Observe the actual timer function without editing it or its arithmetic."""
    trace = {}
    def capture(frame, event, result):
        if event == 'return' and frame.f_code is primal_lll.__code__:
            values = frame.f_locals
            trace.update(
                generators_shape=[len(b), len(A[0]) + len(b)],
                basis_shape=list(values['basis'].shape),
                attempts=[dict(M=int(values['factor']),
                    embedding_shape=list(values['embedded'].shape),
                    short_rows=[[int(x) for x in row]
                                for row in values['reduced'].tolist()])])
    previous = sys.getprofile()
    try:
        sys.setprofile(capture)
        found = primal_lll(A, b, q)
    finally:
        sys.setprofile(previous)
    return found, trace


def main():
    lwe_only = '--lwe-only' in sys.argv
    previous = json.loads((ROOT/'docs/audit/results.json').read_text()) if lwe_only else None
    results = [r for r in previous['results'] if r['scheme']=='GGH'] if previous else []
    traces = {f: t for f, t in previous['traces'].items() if t['result']['scheme']=='GGH'} if previous else {}
    for tier, solver in ([] if lwe_only else [(1,resolver_ggh_tier1),(2,resolver_ggh_tier2),(3,resolver_ggh_tier3)]):
        for path in sorted((ROOT / f'data/ggh/t{tier}').glob('*')):
            if path.suffix not in ('.txt','.csv'): continue
            inst = cargar_instancia_ggh(path)
            B = inst['base']; t = inst['target']; kind = inst['problema'].upper(); q = inst.get('modulo')
            if len(t) != B.shape[1]:
                assert 'SVP' in kind or np.all(t == 0)
                t = np.zeros(B.shape[1])
            args = dict(target=t,base_publica=B,tipo_problema=kind)
            if tier == 3: args['modulo'] = q
            times = []
            for _ in range(3):
                start = perf_counter(); v,R = solver(**args); times.append((perf_counter()-start)*1000)
            G = Matrix(B.astype(int).tolist())
            if q is not None: G = G.col_join(int(q)*eye(B.shape[1]))
            same = row_hnf(G) == row_hnf(R.tolist())
            member = exact_membership(v.tolist(), G)
            svp = 'SVP' in kind
            metric2 = sum(int(x)**2 for x in v) if svp else sum((sympy.Rational(str(float(a)))-int(b))**2 for a,b in zip(t,v))
            optimal = member and ((svp and metric2 == 1) or (not svp and metric2 == 0))
            ref = inst.get('solucion')
            record = dict(scheme='GGH',tier=tier,file=path.name,kind=kind,n=B.shape[1],q=q,times_ms=times,mean_ms=sum(times)/len(times),metric2=str(metric2),metric=float(sympy.sqrt(metric2)),same_lattice=same,member=member,optimal=optimal,reference=None if ref is None else bool(np.array_equal(ref,v)),v=v.tolist())
            results.append(record)
            if tier in (2,3):
                trace={}; v2,R2=solver(**args,trace=trace)
                assert np.array_equal(v,v2) and np.array_equal(R,R2)
                if path.name in ('Tier02T04_Latt.txt','Tier03T03_Latt.txt'):
                    traces[path.name]=dict(trace=trace,result=record,target=t.tolist())
            print(json.dumps(record),flush=True)
    for tier in (1,2):
        for path in sorted((ROOT/f'data/lwe/t{tier}').glob('*')):
            if path.suffix not in ('.txt','.csv'): continue
            A,b,n,q,crypto=parse_lwe(path)
            times=[]
            for _ in range(3):
                start=perf_counter(); found=primal_lll(A,b,q); times.append((perf_counter()-start)*1000)
            assert found is not None
            score,s,e,M=found
            congruences=all((sum(x*y for x,y in zip(row,s))+ei-bi)%q==0 for row,ei,bi in zip(A,e,b))
            centered=[(bi-sum(x*y for x,y in zip(row,s))+q//2)%q-q//2 for row,bi in zip(A,b)]
            assert centered==e
            record=dict(scheme='LWE',source='src/lwe/solver_lwe_lll_timer.py:primal_lll',tier=tier,file=path.name,n=n,m=len(b),q=q,times_ms=times,mean_ms=sum(times)/3,metric2=score,metric=score**0.5,max_error=max(map(abs,e)),congruences=congruences,bound_ok=max(map(abs,e)) <= (5 if tier==1 else 6),s=s,e=e,M=M)
            if crypto: record['decoded']=bit_from_secret(crypto,s,q)
            if previous:
                old = next(r for r in previous['results'] if r['file']==path.name and r['scheme']=='LWE')
                assert all(record[k] == old[k] for k in ('s','e','M','metric2','max_error'))
            results.append(record)
            found2,trace=trace_timer_primal(A,b,q); assert found2==found
            if path.name=='Tier2T05_LWE.txt':
                traces[path.name]=dict(trace=trace,result=record)
            print(json.dumps(record),flush=True)
    output=dict(environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,sympy=sympy.__version__),repetitions=3,results=results,traces=traces)
    (ROOT/'docs/audit/results.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__': main()
