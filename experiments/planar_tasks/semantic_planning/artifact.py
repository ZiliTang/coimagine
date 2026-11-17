"""Build reproducible revisions without executing authored Python on the host."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .runtime import validate_trees

HERE = Path(__file__).parent
FUNCTIONS = {'initial_memory', 'update_memory', 'semantic_features'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source(source):
    tree = ast.parse(source)
    names = set()
    trees = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import) and all(n.name in ('math', 'copy') and n.asname is None for n in node.names):
            continue
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS and not node.decorator_list:
            if node.name in names or node.args.defaults or node.args.kw_defaults or node.returns:
                raise ValueError('duplicate function, defaults or annotations are not supported')
            args = node.args
            expected = {'initial_memory': 0, 'update_memory': 2, 'semantic_features': 3}[node.name]
            if args.vararg or args.kwarg or args.kwonlyargs or len(args.posonlyargs + args.args) != expected:
                raise ValueError('semantic function signature mismatch')
            if any(a.annotation for a in args.posonlyargs + args.args):
                raise ValueError('annotations are not supported')
            names.add(node.name)
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TREES' and trees is None:
            trees = ast.literal_eval(node.value)
            continue
        raise ValueError('source permits a docstring, math/copy imports, three functions and literal TREES only')
    if names != FUNCTIONS:
        raise ValueError('all three semantic functions required')
    validate_trees(trees)
    return trees


def compile_source(source, runtime_source=None):
    validate_source(source)
    runtime_source = runtime_source if runtime_source is not None else (HERE / 'runtime.py').read_text()
    identity = hashlib.sha256((runtime_source + '\n' + source).encode()).hexdigest()
    hooks = f'''
_SEMANTIC = SemanticRuntime(initial_memory, update_memory, semantic_features, TREES, {identity!r})

def evaluate_state(node, memory):
    # v3 keeps a raw context list; this profile uses only bounded recursive memory.
    memory['history'] = []
    return _SEMANTIC.evaluate(node, memory['context'])

def rank_actions(parent, candidates, memory):
    return _SEMANTIC.rank(parent, candidates, memory['context'])
'''
    result = runtime_source + '\n' + source + '\n' + hooks
    compile(result, 'heuristic.py', 'exec')
    return result, identity


def create_revision(source_path, destination, *, parent=None, evidence=(), reason='initial interface seed'):
    """Persist source, embedded runtime, compiled hooks, and optional evidence copies."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source = Path(source_path).read_text()
    runtime = (HERE / 'runtime.py').read_text()
    compiled, identity = compile_source(source, runtime)
    parent_info = None
    if parent is not None:
        previous = verify_revision(parent)
        parent_info = dict(knowledge_id=previous['knowledge_id'], manifest_sha256=sha(Path(parent) / 'manifest.json'))
        if not evidence or not reason.strip():
            raise ValueError('a revision requires evidence files and a reason')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / 'revision'
        stage.mkdir()
        for name, content in (('knowledge.py', source), ('runtime.py', runtime), ('heuristic.py', compiled)):
            (stage / name).write_text(content)
        evidence_paths = []
        for i, path in enumerate(evidence):
            name = f'evidence/{i:03d}_{Path(path).name}'
            (stage / name).parent.mkdir(exist_ok=True)
            shutil.copyfile(path, stage / name)
            evidence_paths.append(name)
        manifest = dict(version='history-semantic-v0', knowledge_id=identity,
                        parent=parent_info, reason=reason, evidence=evidence_paths,
                        evidence_status='author-supplied; not an automatic correctness certificate',
                        files={str(p.relative_to(stage)): sha(p) for p in stage.rglob('*') if p.is_file()})
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        stage.rename(destination)
    return manifest


def verify_revision(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['version'] != 'history-semantic-v0':
        raise ValueError('unsupported revision')
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p != path / 'manifest.json'}
    if actual != set(manifest['files']) or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('revision file set mismatch')
    for name, value in manifest['files'].items():
        if sha(path / name) != value:
            raise ValueError('revision content changed: ' + name)
    compiled, identity = compile_source((path / 'knowledge.py').read_text(), (path / 'runtime.py').read_text())
    if compiled != (path / 'heuristic.py').read_text() or identity != manifest['knowledge_id']:
        raise ValueError('compiled revision mismatch')
    return manifest
