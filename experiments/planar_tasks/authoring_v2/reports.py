"""Deterministic summaries from legal observations only; never read execution truth."""
from collections import Counter
import math
from experiments.planar_tasks.interface import KINEMATICS_FIELDS, FEEDBACK_FIELDS, SCAN_FIELDS, PATCH_FIELDS
from experiments.planar_tasks.option_search_v3.geometry import estimate

GROUPS={'own','execution','contacts','scan','patches','goal','legal_pairs'}
CONTACTS=('contacting_object','contacting_static','contacting_teammate')

def pick(d,keys):return {k:d.get(k) for k in keys}
def stamp(o):
    rs=o.get('robots') or []
    return rs[0].get('time',rs[0].get('elapsed_seconds')) if rs else None

def project(o,fields=('own','execution','contacts'),robots=(0,1)):
    if not isinstance(fields,(list,tuple)) or not fields or set(fields)-GROUPS:raise ValueError('unsupported field group')
    if not isinstance(robots,(list,tuple)) or not robots or any(type(i)is not int or i not in (0,1) for i in robots):raise ValueError('robot IDs must be 0/1')
    out=dict(decision=o.get('decision'),time=stamp(o),robots=[])
    if 'legal_pairs' in fields:out['legal_pairs']=o.get('legal_pairs')
    for r in o.get('robots',[]):
        if r.get('robot_id') not in robots:continue
        d=dict(robot_id=r['robot_id'])
        if 'own' in fields:d['own']=pick(r.get('own') or {},KINEMATICS_FIELDS)
        if 'execution' in fields:d['execution']=pick(r.get('execution') or {},FEEDBACK_FIELDS)
        if 'contacts' in fields:d.update(pick(r,CONTACTS))
        if 'scan' in fields:d['scan']=pick(r.get('scan') or {},SCAN_FIELDS)
        if 'patches' in fields:d['patches']=[pick(p,PATCH_FIELDS) for p in r.get('patches',[])]
        if 'goal' in fields:d.update(pick(r,('goal_xy','goal_radius')))
        out['robots'].append(d)
    return out


def summarize(observations):
    observations=list(observations);events=[];stages=[Counter(),Counter()];contact=[0,0]
    previous=None;first_pose=last_pose=None;pose=None;invalid_estimates=0
    first_pose_time=last_pose_time=None;carried_estimates=0
    for index,o in enumerate(observations):
        p=project(o);t=stamp(o)
        try:pose=estimate(o,pose)
        except (ValueError,KeyError,TypeError,IndexError,ArithmeticError):pose=None
        if pose is None or not all(math.isfinite(v) for v in pose):invalid_estimates+=1;pose=None
        else:
            if first_pose is None:first_pose=list(pose);first_pose_time=t
            last_pose=list(pose);last_pose_time=t
            if not any(p.get('valid') and p.get('rms_m',1)<.002 and p.get('span_m',0)>.035 for r in o['robots'] for p in r.get('patches',[])):
                carried_estimates+=1
        for r in p['robots']:
            i=r['robot_id'];ex=r['execution'];stages[i][str(ex.get('stage'))]+=1
            contact[i]+=r.get('contacting_object') is True
            before=next((x for x in previous['robots'] if x['robot_id']==i),None) if previous else None
            if before:
                for key in ('stage','done','pause_reason','direction_unavailable'):
                    a=before['execution'].get(key);b=ex.get(key)
                    if a!=b:events.append(dict(kind=key+'_change',row=index,time=t,robot=i,before=a,after=b))
                for key in CONTACTS:
                    if before.get(key)!=r.get(key):events.append(dict(kind=key+'_change',row=index,time=t,robot=i,before=before.get(key),after=r.get(key)))
        previous=p
    delta=None
    if first_pose is not None and last_pose is not None:
        delta=[last_pose[0]-first_pose[0],last_pose[1]-first_pose[1],(last_pose[2]-first_pose[2]+math.pi)%(2*math.pi)-math.pi]
    robot_delta=[]
    if observations:
        for a,b in zip(observations[0]['robots'],observations[-1]['robots']):
            robot_delta.append([b['own']['x']-a['own']['x'],b['own']['y']-a['own']['y']])
    return dict(source='legal_observation_boundary_samples',samples=len(observations),
        start_time=stamp(observations[0]) if observations else None,end_time=stamp(observations[-1]) if observations else None,
        stage_sample_counts=[dict(x) for x in stages],contact_sample_counts=contact,
        robot_net_displacement=robot_delta,estimated_object_delta=delta,
        estimate_source='frozen_legal_rectangle_estimator_not_truth',missing_estimate_samples=invalid_estimates,
        carried_estimate_samples=carried_estimates,estimate_start_time=first_pose_time,estimate_end_time=last_pose_time,
        event_count=len(events),sampling_note='Boundary samples can miss within-command events; counts are not duration.',events=events)


def search_summary(traces):
    horizons=[t.get('maximum_forecast_ticks',0) for t in traces]
    residuals=[e['position_residual_m'] for e in prediction_events(traces) if e.get('position_residual_m') is not None]
    # These are logs from the fixed framework; arbitrary stderr is not promoted to ground truth.
    return dict(source='framework_reported_diagnostics',logged_decisions=len(traces),
        selected_status=dict(Counter(t.get('selected_status') for t in traces)),
        maximum_forecast_ticks=max(horizons,default=0),
        diagnostics=dict(Counter(x for t in traces for x in t.get('diagnostics',[]))),
        hook_error_decisions=sum(bool(t.get('hook_errors')) for t in traces),
        comparable_prediction_samples=len(residuals),max_position_residual_m=max(residuals,default=None),
        model_error_nodes=sum(e.get('status')=='model_error' for t in traces for e in t.get('events',[])))


def prediction_events(traces):
    events=[]
    for t in traces:
        f=t.get('prediction_feedback')
        if not isinstance(f,dict):continue
        e=dict(kind='prediction_feedback',decision=t.get('decision'),source='framework_reported_successive_legal_estimates_not_truth',duration_comparable=f.get('duration_comparable') is True)
        x=f.get('position_residual_m')
        if e['duration_comparable'] and type(x) in (float,int) and math.isfinite(x) and x>=0:e['position_residual_m']=x
        events.append(e)
    return events
