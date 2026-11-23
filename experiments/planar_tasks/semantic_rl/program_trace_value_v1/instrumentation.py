"""Native-compatible AST instrumentation of frozen explicit-memory CAP programs.

Records scalar assignments outside loops, control-flow tests outside loops, and
executed return sites. Loops still execute normally; inner-loop microstate is
deliberately not expanded into features. Helper calls are traced, with separate
robot slots for functions whose `r` argument is a public robot observation.
No outcome, fixture, environment, external ID, or future event enters this code.
"""
import ast
import hashlib
import importlib.util
from functools import lru_cache

from ..typed_compiler import validate_source
from ..trusted_native import compile_host

HELPERS = '''
def _tv(buffer, index, value):
    if buffer is not None and type(value) in (int, float, bool):
        if math.isfinite(value):
            buffer[index] = float(value)
    return value

def _tt(buffer, index, value):
    result = bool(value)
    _tv(buffer, index, result)
    return result
'''


class Instrument(ast.NodeTransformer):
    def __init__(self, selected):
        self.selected = selected
        self.function = None
        self.loop_depth = 0
        self.slots = []
        self.bound = False

    def slot(self, node, kind, name):
        index = len(self.slots)
        for robot in (0, 1) if self.bound else (None,):
            self.slots.append(dict(function=self.function, line=node.lineno,
                                   column=node.col_offset, kind=kind, name=name, robot=robot))
        base = ast.Constant(index)
        return ast.BinOp(base, ast.Add(), ast.Name('_tr', ast.Load())) if self.bound else base

    def record(self, node, kind, name, value, test=False):
        return ast.Call(ast.Name('_tt' if test else '_tv', ast.Load()),
                        [ast.Name('_tb', ast.Load()), self.slot(node, kind, name), value], [])

    def visit_FunctionDef(self, node):
        if node.name not in self.selected:
            return node
        self.function = node.name
        self.bound = any(a.arg == 'r' for a in node.args.args)
        node.args.kwonlyargs.append(ast.arg(arg='_tb'))
        node.args.kw_defaults.append(ast.Constant(None))
        self.generic_visit(node)
        if self.bound:
            node.body.insert(0, ast.parse("_tr = 0 if r['robot_id'] == 0 else 1").body[0])
        self.function = None
        self.bound = False
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.function and isinstance(node.func, ast.Name) and node.func.id in self.selected:
            node.keywords.append(ast.keyword(arg='_tb', value=ast.Name('_tb', ast.Load())))
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if not self.function or self.loop_depth:
            return node
        names = [n.id for target in node.targets for n in ast.walk(target)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        return [node] + [ast.Expr(self.record(node, 'assignment', n, ast.Name(n, ast.Load()))) for n in names]

    def visit_If(self, node):
        self.generic_visit(node)
        if self.function and not self.loop_depth:
            node.test = self.record(node, 'branch', ast.unparse(node.test), node.test, test=True)
        return node

    def visit_For(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1
        return node

    visit_While = visit_For

    def visit_Return(self, node):
        self.generic_visit(node)
        if not self.function:
            return node
        # Mark reachability; preserve the returned object without copying/re-evaluating it.
        return [ast.Expr(self.record(node, 'return', 'reached', ast.Constant(1.0))), node]


def instrument(source):
    validate_source(source)
    tree = ast.parse(source)
    if {'_tb', '_tv', '_tt', '_tr'}.intersection(n.id for n in ast.walk(tree) if isinstance(n, ast.Name)):
        raise ValueError('instrumentation identifier collision')
    selected = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name != 'evaluate_candidates' and len(n.body) > 1}
    if not {'update_memory', 'prepare_context'}.issubset(selected):
        raise ValueError('explicit-memory CAP contract required')
    transform = Instrument(selected)
    tree = transform.visit(tree)
    tree.body.extend(ast.parse(HELPERS).body)
    generated = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    if len(transform.slots) > 1024:
        raise ValueError('trace capacity exceeded; do not silently truncate')
    validate_source(generated)
    schema = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                  slots=transform.slots, width=len(transform.slots),
                  aggregation='last finite scalar per static site per decision; absent is NaN',
                  loop_policy='no assignment/test probes inside loops; helper calls and returns retained')
    return generated, schema


@lru_cache(maxsize=32)
def load_native(source):
    directory, manifest = compile_host(source)
    spec = importlib.util.spec_from_file_location('_coimagine_semantics', directory / 'semantics.so')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, manifest
