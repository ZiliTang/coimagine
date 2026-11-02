"""Sequential warm-process timing; run without concurrent test/audit workers."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,platform,statistics,time
from pathlib import Path
from experiments.option_testbench import execution_efficiency_eval as X
from coimagine.random_start.lidar import scan_lidar,scan_lidar_batch
from coimagine.random_start.geometry import box_polygon


def main(out):
    start=time.perf_counter();case=next(c for c in X.cases() if c['name']=='push_follow')
    variants=('eff_scalar','eff_batch','eff_prepare');records=[]
    for v in variants:X.run(case,v)
    for k in range(5):
        for v in variants[k%3:]+variants[:k%3]:
            m=X.run(case,v)['metric'];records.append(dict(repeat=k,variant=v,wall_seconds=m['wall_seconds'],scan_seconds=m['scan_seconds'],physics_seconds=m['physics_seconds'],sim_seconds=m['seconds'],exits=m['exits']))
    args=((5.,4.),(box_polygon(1.3,2.5,.3,.15,.4),),(2.5,2.,.17),[(2.2,1.5,.3,0,0,0),(2.8,1.5,.1,0,0,0)],0,0,0.)
    scans=[]
    for k in range(5):
        for name,fn in [('scalar',scan_lidar),('numpy_batch',scan_lidar_batch)][::1 if k%2==0 else -1]:
            t=time.perf_counter()
            for _ in range(200):fn(*args)
            scans.append(dict(repeat=k,backend=name,seconds_per_scan=(time.perf_counter()-t)/200))
    medians={v:statistics.median(r['wall_seconds'] for r in records if r['variant']==v) for v in variants}
    scan_medians={v:statistics.median(r['seconds_per_scan'] for r in scans if r['backend']==v) for v in ('scalar','numpy_batch')}
    result=dict(case=case,wall_timing_scope='one warm process, 5 repeats, rotated variant order; simulation plus metrics; excludes imports, process startup and JSON output; no concurrent agent test/audit worker',
        full_option=records,wall_medians=medians,scalar_to_batch_speedup=medians['eff_scalar']/medians['eff_batch'],scalar_to_prepare_speedup=medians['eff_scalar']/medians['eff_prepare'],
        scan=scans,scan_medians=scan_medians,scan_speedup=scan_medians['scalar']/scan_medians['numpy_batch'],elapsed=time.perf_counter()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),software=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),pymunk=importlib.metadata.version('pymunk')))
    (out/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('case','full_option','scan')}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
