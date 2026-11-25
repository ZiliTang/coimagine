"""Frozen public-semantic joint effect models; room-balanced local support.

Raw outcomes remain paired. Numeric deltas and boolean next-values are predicted
jointly. Calibration rooms never enter nearest-neighbour fitting. Certification is
empirical and local; independent admission-subset coverage remains an evaluation.
"""
from collections import defaultdict
from copy import deepcopy
import math
import numpy as np

from . import frontend
from .core import digest, evaluate


def feature_names(action, profile='legacy-v1'):
    d = frontend.option_table()[action]['heading']
    if profile=='public-context-v2':
        fields=[('distance',.75),('gx',1.5),('gy',1.5),('speed',.3),('stagnation',2.),('uncertainty',.2),
                ('contact0',1.),('contact1',1.),('static_contact',1.),('peer_contact',1.)]
        fields += [(f'gap_{j}',1.5)for j in range(8)]+[(f'wall_seen_{j}',1.)for j in range(8)]
        headings=range(8)if d is None else ((d-1)%8,d,(d+1)%8)
        fields += [(f'legal:push_{role}_{j}_6',1.)for j in headings for role in ('both','r0','r1')]
        if d is not None:fields += [(f'align_{d}',.5)]
        return fields
    fields = [('distance', 2.), ('speed', .3), ('stagnation', 5.), ('uncertainty', .2),
              ('contact0', 1.), ('contact1', 1.), ('static_contact', 1.), ('peer_contact', 1.)]
    if d is not None:
        fields += [('align_%d' % d, .5), ('gap_%d' % d, 1.), ('wall_seen_%d' % d, 1.)]
    else:
        fields += [('gx', 2.), ('gy', 2.)]
    return fields


def distance(a, b, action, profile='legacy-v1'):
    fields = feature_names(action,profile)
    value=math.sqrt(sum(((float(a[k])-float(b[k]))/scale)**2 for k, scale in fields)/len(fields))
    if profile=='public-context-v2':
        if any(a[k]!=b[k]for k in ('contact0','contact1','seen')):return float('inf')
        value=max(value,abs(a['distance']-b['distance'])/.75,
                  math.hypot(a['gx']-b['gx'],a['gy']-b['gy'])/1.5)
    return value


def reset_epistemic(z):
    return {k: (True if k.startswith('permit:') else v) for k, v in z.items()}


def point_outcome(row, current, schema):
    before, after = row['semantic'], row['next_semantic']
    result = {}
    for key, spec in schema['metrics'].items():
        if key.startswith('permit:'):
            result[key] = True
        elif spec['unit'] == 'bool':
            result[key] = bool(after[key])
        else:
            value = float(current[key])+float(after[key])-float(before[key])
            lo, hi = spec.get('domain', [None, None])
            result[key] = max(lo, value) if lo is not None else value
            if hi is not None: result[key] = min(hi, result[key])
    return result


def outcome_valid(policy, action, z, *, path_safe):
    choice = policy.select(z)
    promise = policy.program.get('expect', {}).get(action)
    promised = bool(evaluate(promise, policy.values(z))) if promise else True
    return bool(path_safe and promised and choice['executable']), choice


class EffectModel:
    def __init__(self, packet):
        self.packet = deepcopy(packet)
        self.rows = self.packet['rows']
        self.groups = defaultdict(list)
        for r in self.rows: self.groups[r['action']].append(r)
        self.feature_profile=self.packet.get('features_profile','legacy-v1')
        self.matrices={};self.columns={};self.contact_modes={}
        for action,group in self.groups.items():
            fields=feature_names(action,self.feature_profile);self.columns[action]=fields
            self.matrices[action]=np.asarray([[float(r['semantic'][k])/scale for k,scale in fields]for r in group])
            self.contact_modes[action]=np.asarray([[r['semantic'][k]for k in ('contact0','contact1','seen')]for r in group],dtype=bool)
        self.sha256 = digest(packet)

    @classmethod
    def fit(cls, rows, calibration=(), *, version=0, parent=None):
        # Room IDs are provenance/group weights only, never predictor features.
        training_rooms = sorted({r['room'] for r in rows})
        calibration_rooms = sorted({r['room'] for r in calibration})
        if set(training_rooms) & set(calibration_rooms):
            raise ValueError('calibration and fitting rooms overlap')
        packet = dict(schema='public-semantic-joint-knn-v2',features_profile='public-context-v2', version=version, parent=parent,
                      training_rooms=training_rooms, calibration_rooms=calibration_rooms,
                      max_distance=.35, min_rooms=4, neighbors=8, rows=list(rows), calibration={})
        model = cls(packet)
        scores = defaultdict(list)
        flags = defaultdict(list)
        from .scaffold import host_schema
        schema = host_schema()
        for row in calibration:
            prediction = model.predict(row['action'], row['semantic'], schema, calibrated=False)
            if not prediction['samples']: continue
            truth = row['next_semantic']
            errors = {}
            for key, spec in schema['metrics'].items():
                if key.startswith('permit:'): continue
                samples = [s['values'][key] for s in prediction['samples']]
                if spec['unit'] == 'bool':
                    flags[row['action']].append(int(truth[key] not in samples))
                else:
                    errors[key] = max(0., min(samples)-truth[key], truth[key]-max(samples))
            scores[row['action']].append(dict(room=row['room'], residuals=errors))
        for action, values in scores.items():
            # Room maximum residual, then finite-sample 90% upper order statistic.
            per_room = defaultdict(dict)
            for row in values:
                for key, err in row['residuals'].items():
                    per_room[row['room']][key] = max(err, per_room[row['room']].get(key, 0.))
            rank = math.ceil(.90*(len(per_room)+1))
            padding = {k: sorted(r.get(k, 0.) for r in per_room.values())[rank-1]
                       for k in next(iter(per_room.values()))} if rank <= len(per_room) else None
            packet['calibration'][action] = dict(rooms=len(per_room), padding=padding,
                                                categorical_misses=sum(flags[action]), categorical_checks=len(flags[action]))
        return cls(packet)

    def predict(self, action, current, schema, *, calibrated=True):
        nearest = {}
        group=self.groups.get(action,[])
        if group:
            fields=self.columns[action]
            query=np.asarray([float(current[k])/scale for k,scale in fields])
            delta=self.matrices[action]-query
            distances=np.sqrt(np.mean(delta**2,axis=1))
            if self.feature_profile=='public-context-v2':
                distances=np.maximum(distances,np.abs(delta[:,0]))
                distances=np.maximum(distances,np.hypot(delta[:,1],delta[:,2]))
                mode=np.asarray([current[k]for k in ('contact0','contact1','seen')],dtype=bool)
                distances=np.where(np.all(self.contact_modes[action]==mode,axis=1),distances,np.inf)
            indices=np.flatnonzero(distances<=self.packet['max_distance'])
        else:indices=[]
        for index in indices:
            d=distances[index];row=group[index]
            if d <= self.packet['max_distance'] and (row['room'] not in nearest or d < nearest[row['room']][0]):
                nearest[row['room']] = (float(d), row)
        ordered = sorted(nearest.values(), key=lambda pair: (pair[0], pair[1]['room']))[:self.packet['neighbors']]
        samples = [dict(values=point_outcome(r, current, schema), path_safe=not r['audit']['object_static'],
                        room=r['room'], distance=d) for d, r in ordered]
        calibration = self.packet['calibration'].get(action, {})
        return dict(samples=samples, local_rooms=len(nearest),
                    supported=len(nearest) >= self.packet['min_rooms'],
                    calibrated=calibrated and calibration.get('padding') is not None,
                    calibration=calibration, source='empirical_joint_effect_model', model_sha256=self.sha256)

    def interval_outcomes(self, prediction, schema):
        """Preserve discrete modes; enclose continuous outputs within each mode."""
        modes = defaultdict(list)
        bool_names = [k for k, s in schema['metrics'].items() if s['unit'] == 'bool']
        for sample in prediction['samples']:
            key = tuple(sample['values'][n] for n in bool_names)+(sample['path_safe'],)
            modes[key].append(sample['values'])
        pad = prediction.get('calibration', {}).get('padding') or {}
        result = []
        for key, samples in modes.items():
            updates = dict(zip(bool_names, key[:-1]))
            for n, spec in schema['metrics'].items():
                if spec['unit'] == 'bool': continue
                domain = spec.get('domain', [None, None]); margin = pad.get(n, 0.)
                lo, hi = min(v[n] for v in samples)-margin, max(v[n] for v in samples)+margin
                if domain[0] is not None: lo = max(domain[0], lo)
                if domain[1] is not None: hi = min(domain[1], hi)
                updates[n] = [lo, hi]
            result.append(dict(updates=updates, path_safe=key[-1]))
        return result
