"""Linux bubblewrap boundary for standard-library-only synthesized policies.

Fail closed if bubblewrap/user namespaces are unavailable. The synthesis agent
itself is NOT sandboxed by this module; only the frozen deployment subprocess is.
"""
from pathlib import Path
import shutil

SANDBOX = dict(kind='bubblewrap', network=False, repository=False, credentials=False,
               writable='ephemeral /tmp only', memory_bytes=256*1024*1024,
               cpu_seconds=30, file_bytes=1024*1024, python='system python3 -I -u')


def policy_command(path, *, artifacts=()):
    path = Path(path).resolve(strict=True)
    if not shutil.which('bwrap') or not shutil.which('prlimit'):
        raise RuntimeError('bubblewrap and prlimit required; no unsandboxed fallback')
    command = ['prlimit', '--as=268435456', '--cpu=30', '--fsize=1048576', '--',
               'bwrap', '--unshare-all', '--die-with-parent', '--new-session',
               '--cap-drop', 'ALL', '--clearenv', '--setenv', 'PYTHONHASHSEED', '0']
    for root in ('/usr', '/lib', '/lib64'):
        if Path(root).exists():
            command += ['--ro-bind', root, root]
    command += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/work']
    for source, name in artifacts:
        if Path(name).name != name:raise ValueError('artifact basename required')
        command += ['--ro-bind',str(Path(source).resolve(strict=True)),'/work/'+name]
    return command + ['--ro-bind', str(path), '/work/policy.py', '--chdir', '/work',
        '/usr/bin/python3', '-I', '-u', '/work/policy.py']
