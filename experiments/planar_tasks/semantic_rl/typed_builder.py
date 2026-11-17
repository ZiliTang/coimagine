"""Trusted compiler driver, invoked only inside the build namespace."""
import json
import ast
import subprocess
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, '/toolchain')
from Cython.Compiler.Main import compile as cython_compile
import Cython

source = '/build/_coimagine_semantics.py'
# Cython3.0 pure-Python const-memoryview annotations are not sufficient:
# they may silently become Python objects. Supply a host-owned augmented pxd
# and remove only this function's annotations from the compiler input. The
# original submitted source remains separately preserved and hashed.
original=Path(source).read_text()
Path('/build/submitted_source.py').write_text(original)
tree=ast.parse(original)
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=='evaluate_candidates':
        for arg in node.args.args:arg.annotation=None
        node.returns=None
Path(source).write_text(ast.unparse(tree)+'\n')
Path('/build/_coimagine_semantics.pxd').write_text(
    'cpdef object evaluate_candidates(const double[:] context, const double[:, :] candidates, double[:, :] out)\n')
result = cython_compile(source, compiler_directives={
    'language_level': 3, 'boundscheck': True, 'wraparound': True,
    'initializedcheck': True, 'nonecheck': True, 'cdivision': False,
})
if result.num_errors:
    raise RuntimeError('Cython compilation failed')
import re
c_source=Path(result.c_file).read_text()
if not re.search(r'static PyObject \*[^\n]*evaluate_candidates\([^\n]*__Pyx_memviewslice[^\n]*__Pyx_memviewslice[^\n]*__Pyx_memviewslice',c_source):
    raise RuntimeError('compiler did not produce three native memoryview arguments')
command = ['/usr/bin/gcc', '-shared', '-fPIC', '-O2', '-fno-fast-math',
           '-I'+sysconfig.get_path('include'), result.c_file,
           '-o', '/build/semantics.so']
subprocess.run(command, check=True)
Path('/build/build.json').write_text(json.dumps(dict(
    cython=Cython.__version__, python=sys.version, soabi=sysconfig.get_config_var('SOABI'),
    command=command, boundscheck=True, fast_math=False,native_memoryview_signature_verified=True,
    lowering='host-owned augmented pxd for readonly memoryview arguments')))
