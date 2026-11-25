"""Extractable, typed CAP expressions with identical concrete and SMT selection.

The verifier reasons about the actual priority/score selector, not an existential
alternative successor. Predictions are unions of joint boxes. An omitted written
metric is unknown, never silently copied. UNSAT proves only the supplied model;
empirical model coverage is a separate measured claim.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math

import z3


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def extract(source):
    """Read a literal declaration without executing any authored Python."""
    found = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'SYMBOLIC_POLICY' for t in n.targets)]
    if len(found) != 1:
        raise ValueError('exactly one literal SYMBOLIC_POLICY required')
    return ast.literal_eval(found[0])


def refs(expr):
    if not isinstance(expr, list) or not expr:
        raise ValueError('expression must be a nonempty list')
    if expr[0] == 'var':
        return {expr[1]}
    if expr[0] == 'lit':
        return set()
    return set().union(*(refs(e) for e in expr[1:]))


def unit(expr, units):
    op, *args = expr
    if op == 'var':
        if len(args) != 1 or args[0] not in units:
            raise ValueError('unknown metric: ' + repr(args))
        return units[args[0]]
    if op == 'lit':
        if len(args) != 2 or not isinstance(args[1], str):
            raise ValueError('literal requires value and unit')
        v, u = args
        if u == 'bool':
            if type(v) is not bool:
                raise ValueError('boolean literal required')
        elif type(v) not in (int, float) or not math.isfinite(v):
            raise ValueError('finite numeric literal required')
        return u
    us = [unit(a, units) for a in args]
    if op in ('and', 'or') and us and set(us) == {'bool'}:
        return 'bool'
    if op == 'not' and us == ['bool']:
        return 'bool'
    if op in ('lt', 'le', 'gt', 'ge', 'eq', 'ne') and len(us) == 2 and us[0] == us[1]:
        if us[0] == 'bool' and op not in ('eq', 'ne'):
            raise ValueError('ordered comparison on boolean')
        return 'bool'
    if op in ('add', 'sub', 'min', 'max') and len(us) == 2 and us[0] == us[1] != 'bool':
        return us[0]
    if op in ('abs', 'neg') and len(us) == 1 and us[0] != 'bool':
        return us[0]
    if op == 'mul' and len(us) == 2 and 'bool' not in us and '1' in us:
        return us[1] if us[0] == '1' else us[0]
    # Restrict division to literal nonzero dimensionless denominators: no hidden
    # undefined branch, unit conversion or nonlinear rational singularity.
    if (op == 'div' and len(us) == 2 and us[0] != 'bool' and us[1] == '1'
            and args[1][0] == 'lit' and args[1][1] != 0):
        return us[0]
    if op == 'if' and len(us) == 3 and us[0] == 'bool' and us[1] == us[2]:
        return us[1]
    raise ValueError('invalid operator, arity or units: ' + repr(expr))


def evaluate(expr, values, symbolic=False):
    op, *args = expr
    if op == 'var':
        return values[args[0]]
    if op == 'lit':
        if not symbolic:
            return args[0]
        return z3.BoolVal(args[0]) if args[1] == 'bool' else z3.RealVal(str(args[0]))
    a = [evaluate(v, values, symbolic) for v in args]
    if op == 'and': return z3.And(*a) if symbolic else all(a)
    if op == 'or': return z3.Or(*a) if symbolic else any(a)
    if op == 'not': return z3.Not(a[0]) if symbolic else not a[0]
    if op == 'if': return z3.If(*a) if symbolic else (a[1] if a[0] else a[2])
    if op == 'add': return a[0] + a[1]
    if op == 'sub': return a[0] - a[1]
    if op == 'mul': return a[0] * a[1]
    if op == 'div': return a[0] / a[1]
    if op == 'neg': return -a[0]
    if op == 'abs': return z3.If(a[0] >= 0, a[0], -a[0]) if symbolic else abs(a[0])
    if op == 'min': return z3.If(a[0] <= a[1], a[0], a[1]) if symbolic else min(a)
    if op == 'max': return z3.If(a[0] >= a[1], a[0], a[1]) if symbolic else max(a)
    if op == 'lt': return a[0] < a[1]
    if op == 'le': return a[0] <= a[1]
    if op == 'gt': return a[0] > a[1]
    if op == 'ge': return a[0] >= a[1]
    if op == 'eq': return a[0] == a[1]
    if op == 'ne': return a[0] != a[1]
    raise ValueError('unknown operator ' + op)


TRUE = ['lit', True, 'bool']
FALSE = ['lit', False, 'bool']


class Policy:
    def __init__(self, program, host):
        self.program, self.host = deepcopy(program), deepcopy(host)
        if set(program) - {'version', 'derived', 'rules', 'terminal', 'expect'}:
            raise ValueError('unknown policy fields')
        if program.get('version') != 1 or not program.get('rules'):
            raise ValueError('nonempty version-1 rules required')
        self.units = {k: v['unit'] for k, v in host['metrics'].items()}
        self.base = set(self.units)
        self.derived = program.get('derived', {})
        self.expanded_refs = {n: {n} for n in self.base}
        # Declaration order is a topological order; reject cycles and forward refs.
        for name, expr in self.derived.items():
            if name in self.units:
                raise ValueError('derived metric shadows primitive ' + name)
            self.units[name] = unit(expr, self.units)
            self.expanded_refs[name] = set().union(*(self.expanded_refs[x] for x in refs(expr)))
        self.rules = program['rules']
        names = set()
        for r in self.rules:
            if set(r) - {'name', 'when', 'option', 'score'} or r['name'] in names:
                raise ValueError('invalid or duplicate rule')
            names.add(r['name'])
            if r['option'] not in host['options'] or unit(r['when'], self.units) != 'bool':
                raise ValueError('unknown option or nonboolean guard')
            if unit(r.get('score', ['lit', 0, '1']), self.units) != '1':
                raise ValueError('rule score must be dimensionless')
        if unit(program.get('terminal', FALSE), self.units) != 'bool':
            raise ValueError('terminal condition must be boolean')
        for name, option in host['options'].items():
            if unit(option['requires'], self.units) != 'bool':
                raise ValueError('option precondition must be boolean')
            if not set(option['writes']) <= self.base:
                raise ValueError('unknown written metric')
            if set(option['writes']) & set(option.get('frame', [])):
                raise ValueError('written metric cannot be framed')
            if set(option['writes']) | set(option.get('frame', [])) != self.base:
                raise ValueError('every primitive must be written or explicitly framed')
        for name, expr in program.get('expect', {}).items():
            if name not in host['options'] or unit(expr, self.units) != 'bool':
                raise ValueError('invalid effect promise')
        self.sha256 = digest(dict(program=program, host=host))

    def values(self, primitive):
        if set(primitive) != self.base:
            raise ValueError('concrete semantic packet must contain every primitive exactly once')
        v = dict(primitive)
        for name, spec in self.host['metrics'].items():
            x = v[name]
            if spec['unit'] == 'bool':
                if type(x) is not bool: raise ValueError('nonboolean metric ' + name)
            elif type(x) not in (int, float) or not math.isfinite(x):
                raise ValueError('nonfinite metric ' + name)
            domain = spec.get('domain')
            if domain and ((domain[0] is not None and x < domain[0]) or
                           (domain[1] is not None and x > domain[1])):
                raise ValueError('metric outside declared domain ' + name)
        for name, expr in self.derived.items(): v[name] = evaluate(expr, v)
        return v

    def select(self, primitive):
        v = self.values(primitive)
        if evaluate(self.program.get('terminal', FALSE), v):
            return dict(rule=None, option=None, terminal=True, executable=True)
        active = [(evaluate(r.get('score', ['lit', 0, '1']), v), -i, r)
                  for i, r in enumerate(self.rules) if evaluate(r['when'], v)]
        if not active:
            return dict(rule=None, option=None, terminal=False, executable=False)
        r = max(active, key=lambda x: x[:2])[2]
        return dict(rule=r['name'], option=r['option'], terminal=False,
                    executable=bool(evaluate(self.host['options'][r['option']]['requires'], v)))

    def dependencies(self, changed):
        changed = set(changed)
        # Definition edits can begin at a derived symbol, not only a primitive
        # value. Propagate through direct references in topological order.
        for name,expr in self.derived.items():
            if refs(expr)&changed:changed.add(name)
        affected = lambda e: bool(refs(e)&changed)
        rules=[r['name']for r in self.rules if affected(r['when'])or affected(r.get('score',['lit',0,'1']))]
        return dict(derived=[n for n in self.derived if n in changed],
                    rules=rules,selection_competitors=[r['name']for r in self.rules]if rules else [],
                    requirements=[n for n, o in self.host['options'].items() if affected(o['requires'])],
                    promises=[n for n, e in self.program.get('expect', {}).items() if affected(e)])

    def _symbolic(self, prefix):
        v = {n: (z3.Bool(prefix+n) if u['unit'] == 'bool' else z3.Real(prefix+n))
             for n, u in self.host['metrics'].items()}
        for n, e in self.derived.items(): v[n] = evaluate(e, v, True)
        return v

    def _box(self, v, bounds):
        if set(bounds) - self.base:
            raise ValueError('prediction contains unknown or derived metric')
        constraints = []
        for n, spec in self.host['metrics'].items():
            b = bounds.get(n)
            if spec['unit'] == 'bool':
                if b is not None:
                    if type(b) is not bool: raise ValueError('boolean bound must be bool or null')
                    constraints.append(v[n] == b)
                continue
            for interval in (spec.get('domain'), b):
                if interval is None: continue
                if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                    raise ValueError('numeric interval required')
                lo, hi = interval
                if any(x is not None and (type(x) not in (int, float) or not math.isfinite(x)) for x in interval):
                    raise ValueError('invalid numeric interval')
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError('reversed prediction interval')
                if lo is not None: constraints.append(v[n] >= z3.RealVal(str(lo)))
                if hi is not None: constraints.append(v[n] <= z3.RealVal(str(hi)))
        return constraints

    def _selection(self, v):
        guards = [evaluate(r['when'], v, True) for r in self.rules]
        scores = [evaluate(r.get('score', ['lit', 0, '1']), v, True) for r in self.rules]
        terminal = evaluate(self.program.get('terminal', FALSE), v, True)
        choices = [z3.And(z3.Not(terminal), g, *[
            z3.Or(z3.Not(guards[j]), scores[i] > scores[j] if j < i else scores[i] >= scores[j])
            for j in range(len(guards)) if j != i]) for i, g in enumerate(guards)]
        return choices, terminal

    def _merged_guards_sound(self, timeout_ms):
        """Certify when existence is equivalent to safe actual selection.

        This shortcut is valid only if EVERY enabled rule implies its own
        executable condition. A merely available safe alternative never suffices.
        Finite typed scores and the fixed tie break then choose a safe member.
        """
        if hasattr(self,'_guard_certificate'):return self._guard_certificate
        v=self._symbolic('guard-certificate:')
        constraints=self._box(v,{})
        for rule in self.rules:
            bad=z3.simplify(z3.And(evaluate(rule['when'],v,True),
                z3.Not(evaluate(self.host['options'][rule['option']]['requires'],v,True))))
            if z3.is_false(bad):continue
            if self._query(constraints,bad,v,timeout_ms)['status']!='proved':
                self._guard_certificate=False;return False
        self._guard_certificate=True;return True

    @staticmethod
    def _query(constraints, predicate, v, timeout, include_witness=True):
        s = z3.Solver(); s.set(timeout=timeout); s.add(*constraints, predicate)
        status = s.check()
        if status == z3.unsat: return dict(status='proved')
        if status == z3.unknown: return dict(status='unknown', reason=s.reason_unknown())
        if not include_witness:return dict(status='counterexample')
        m = s.model()
        witness = {n: str(m.eval(x, model_completion=True)) for n, x in v.items()}
        return dict(status='counterexample', witness=witness)

    def check(self, current, action, outcomes, *, quality='empirical', timeout_ms=1000, enumerate_rules=True):
        if action not in self.host['options'] or quality not in ('model', 'empirical', 'unknown'):
            raise ValueError('invalid action or model quality')
        if not outcomes: return dict(status='unknown', reason='no_model_support', branches=[], requests=[])
        option = self.host['options'][action]
        pre = self._symbolic('pre:'); post = self._symbolic('post:')
        constraints = self._box(pre, current)
        feasible = self._query(constraints, z3.BoolVal(True), pre, timeout_ms, include_witness=False)
        if feasible['status'] == 'proved': raise ValueError('empty current state set')
        if feasible['status'] == 'unknown': return feasible
        issues = []
        start = self._query(constraints, z3.Not(evaluate(option['requires'], pre, True)), pre, timeout_ms)
        if start['status'] != 'proved': issues.append(dict(property='current_executable', **start))
        branches = []
        merged=self._merged_guards_sound(timeout_ms)
        if merged:
            terminal=evaluate(self.program.get('terminal',FALSE),post,True)
            guards=[evaluate(r['when'],post,True)for r in self.rules]
            no_action=z3.And(z3.Not(terminal),z3.Not(z3.Or(*guards)))
            bad_selected=z3.BoolVal(False)
            selected=self._selection(post)[0]if enumerate_rules else None
        else:
            selected, terminal = self._selection(post)
            no_action = z3.And(z3.Not(terminal), z3.Not(z3.Or(*selected)))
            bad_selected = z3.Or(*[z3.And(ch, z3.Not(evaluate(self.host['options'][r['option']]['requires'], post, True)))
                                  for ch, r in zip(selected, self.rules)])
        for index, outcome in enumerate(outcomes):
            if set(outcome) - {'updates', 'path_safe', 'label'}:
                raise ValueError('unexpected outcome fields')
            updates = outcome['updates']
            if not set(updates) <= set(option['writes']): raise ValueError('effect modifies framed state')
            c = constraints + self._box(post, updates)
            c += [post[n] == pre[n] for n in option.get('frame', [])]
            possible = self._query(c, z3.BoolVal(True), post, timeout_ms, include_witness=False)
            if possible['status'] == 'proved': raise ValueError('empty outcome branch')
            row = dict(index=index, label=outcome.get('label'), checks={}, possible_rules=[])
            if possible['status'] == 'unknown': row['checks']['feasibility'] = possible
            row['checks']['successor_selection'] = self._query(c, z3.Or(no_action, bad_selected), post, timeout_ms)
            if action in self.program.get('expect', {}):
                row['checks']['effect_promise'] = self._query(c, z3.Not(evaluate(self.program['expect'][action], post, True)), post, timeout_ms)
            safety = outcome.get('path_safe')
            if safety is not True:
                row['checks']['path_safety'] = dict(status='counterexample' if safety is False else 'unknown')
            if enumerate_rules:
                for ch, r in zip(selected, self.rules):
                    q = self._query(c, ch, post, timeout_ms, include_witness=False)
                    if q['status'] != 'proved': row['possible_rules'].append(r['name'])
                row['productive_possible'] = any(r['name'] in row['possible_rules'] and
                        self.host['options'][r['option']].get('productive', True) for r in self.rules)
            branches.append(row)
            issues.extend(dict(branch=index, property=k, **q) for k, q in row['checks'].items() if q['status'] != 'proved')
        status = 'counterexample' if any(i['status'] == 'counterexample' for i in issues) else ('unknown' if issues else ('proved' if quality == 'model' else 'conditional' if quality == 'empirical' else 'unknown'))
        dependencies = self.dependencies(option['writes'])
        return dict(status=status, action=action, quality=quality, policy_sha256=self.sha256,
                    selection_reasoning='certified_merged_guards'if merged else 'exact_score_and_priority',
                    branches=branches, issues=issues, affected=dependencies,
                    requests=[dict(action=action, metrics=sorted(option['writes']), reason=i['property'],
                                   branch=i.get('branch')) for i in issues])
