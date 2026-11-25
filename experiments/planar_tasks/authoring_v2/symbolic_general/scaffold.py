"""Host-owned source carrier; only the literal rule declaration is author editable."""
import ast
from functools import lru_cache
from pathlib import Path
import z3
from . import frontend
from .core import Policy, extract, digest


def host_schema():
    metrics = {k: dict(unit='bool') for k in ['seen', 'contact0', 'contact1', 'static_contact', 'peer_contact']}
    metrics.update({k: dict(unit=u, domain=b) for k, u, b in [
        ('distance', 'm', [0, 20]), ('gx', 'm', [-20, 20]), ('gy', 'm', [-20, 20]),
        ('uncertainty', 'm', [0, 5]), ('stagnation', 's', [0, 10000]), ('speed', 'm/s', [0, 100])]})
    for d in range(8):
        metrics['align_%d' % d] = dict(unit='1', domain=[-1.0000001, 1.0000001])
        metrics['gap_%d' % d] = dict(unit='m', domain=[-2, 5])
        metrics['wall_seen_%d' % d] = dict(unit='bool')
    options = {}
    for name, opt in frontend.option_table().items():
        metrics['legal:'+name] = dict(unit='bool')
        metrics['permit:'+name] = dict(unit='bool')
        options[name] = dict(requires=['var', 'legal:'+name], productive=opt['family'] != 'brake',
                             command=opt)
    for opt in options.values():
        opt.update(writes=list(metrics), frame=[])
    return dict(metrics=metrics, options=options)


def starter():
    rules = []
    for name, opt in frontend.option_table().items():
        when = ['and', ['var', 'legal:'+name], ['var', 'permit:'+name]]
        d = opt['heading']
        if opt['family'] == 'push':
            when += [['gt', ['var', 'align_%d' % d], ['lit', .70, '1']],
                     ['gt', ['var', 'gap_%d' % d], ['lit', .08, 'm']]]
            score = ['add', ['var', 'align_%d' % d], ['lit', .20 if 'both' in name else .0, '1']]
            if opt['ticks'] == 6: score = ['add', score, ['lit', .01, '1']]
        elif opt['family'] == 'orbit':
            score = ['lit', -.1 if name == 'orbit_0_ccw' else -.2, '1']
        elif opt['family'] == 'move':
            when += [['not', ['var', 'seen']]]
            score = ['var', 'align_%d' % d]
        else:
            score = ['lit', -100, '1']
        rules.append(dict(name=name, option=name, when=when, score=score))
    return dict(version=1, derived={}, rules=rules, expect={})


PORTABLE = '''
def expression(expr, values):
    op = expr[0]
    if op == 'var': return values[expr[1]]
    if op == 'lit': return expr[1]
    a = [expression(e,values) for e in expr[1:]]
    if op == 'and': return all(a)
    if op == 'or': return any(a)
    if op == 'not': return not a[0]
    if op == 'if': return a[1] if a[0] else a[2]
    if op == 'add': return a[0]+a[1]
    if op == 'sub': return a[0]-a[1]
    if op == 'mul': return a[0]*a[1]
    if op == 'div': return a[0]/a[1]
    if op == 'neg': return -a[0]
    if op == 'abs': return abs(a[0])
    if op == 'min': return min(a)
    if op == 'max': return max(a)
    if op == 'lt': return a[0]<a[1]
    if op == 'le': return a[0]<=a[1]
    if op == 'gt': return a[0]>a[1]
    if op == 'ge': return a[0]>=a[1]
    if op == 'eq': return a[0]==a[1]
    if op == 'ne': return a[0]!=a[1]
    raise ValueError('unsupported expression')

def choose_symbolic(memory,observation):
    z, commands = semantic_packet(memory,observation)
    for name,expr in SYMBOLIC_POLICY.get('derived',{}).items():
        z[name]=expression(expr,z)
    candidates=[]
    for index,r in enumerate(SYMBOLIC_POLICY['rules']):
        if expression(r['when'],z):
            candidates.append((expression(r.get('score',['lit',0,'1']),z),-index,r['option']))
    if not candidates:
        return None,None
    name=max(candidates)[2]
    return name,commands[name]

def review_push(memory, observation, pair):
    z, commands=semantic_packet(memory,observation)
    matches=[name for name,cmd in commands.items() if [cmd['a0'],cmd['a1']]==pair and name.startswith('push_')]
    return dict(verdict='pass' if any(z['legal:'+n] and z['permit:'+n] for n in matches) else 'unknown',peer_robots=[])

def audit_policy(memory, observation, menu):
    name, command=choose_symbolic(memory,observation)
    selected=[command['a0'],command['a1']] if name and name.startswith('push_') else None
    reviews=[pair for pair in menu if review_push(memory,observation,pair)['verdict']=='pass']
    return dict(version=1,proposed=reviews,selected=selected,claim='selected' if selected else 'abstain',widths=[],peer_predictions=[])

def prepare_context(memory, observation):
    name,command=choose_symbolic(memory,observation)
    if name and name.startswith('push_'):
        review=review_push(memory,observation,[command['a0'],command['a1']])
        if review['verdict']!='pass':
            raise ValueError('selected PUSH fails declared executable condition')
    return dict(values=[0.],state=[observation['tick']],proposals=[command] if command else [])

def evaluate_candidates(context: cython.const[cython.double][:],
                        candidates: cython.const[cython.double][:, :],
                        out: cython.double[:, :]) -> None:
    i: cython.Py_ssize_t
    for i in range(candidates.shape[0]):
        out[i,0]=0.
'''


def source(program):
    checked = Policy(program, host_schema())
    state = checked._symbolic('gate:')
    for rule in checked.rules:
        from .core import evaluate
        action = rule['option']
        solver = z3.Solver(); solver.set(timeout=1000)
        solver.add(*checked._box(state, {}), evaluate(rule['when'], state, True),
                   z3.Not(z3.And(state['legal:'+action], state['permit:'+action])))
        if solver.check() != z3.unsat:
            raise ValueError('rule must imply option legality and epistemic permission: '+rule['name'])
    if 'terminal' in program:
        raise ValueError('physical task termination is host-owned; do not author a success condition')
    text = ('import cython\nSEMANTIC_API = "typed-stream-v1"\nNAMES = ["constant"]\n'
            'STATE_NAMES = ["tick"]\nHISTORY_LENGTH = 1\n'
            'POLICY_CHECK_RULES = {"version":1,"widths":{},"peer_models":{}}\n'
            'SYMBOLIC_POLICY = '+repr(program)+'\n'+Path(frontend.__file__).read_text()+'\n'+PORTABLE)
    if len(text.encode()) > 49152:
        raise ValueError('generated policy exceeds typed carrier size')
    return text


@lru_cache(maxsize=64)
def verify_source(text):
    program = extract(text)
    if ast.dump(ast.parse(text)) != ast.dump(ast.parse(source(program))):
        raise ValueError('policy carrier changed; only SYMBOLIC_POLICY is editable in this profile')
    return Policy(program, host_schema())
