"""C++ backend for this exact ordered set of ten regional CAPs."""

import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from tools import cap_regional_extension_run as study

SOURCES = Path(__file__).with_name("regional_cap_cpp")
CACHE = study.ext.old.ROOT / "outputs/toolchains/regional-cap-cpp"
SID_ORDER = ("s004", "s008", "s011", "s013", "s019", "n_s000", "n_s001", "n_s002", "n_s005", "n_s009")
REFERENCE = "reference.json"
FLAGS = [
    "-std=c++17",
    "-O2",
    "-shared",
    "-fPIC",
    "-fvisibility=hidden",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-builtin-sin",
    "-fno-builtin-cos",
]
NATIVE = None


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def build():
    frozen = study.read(study.OUT / "frozen.json")
    original = study.read(SOURCES / REFERENCE)
    assert list(SID_ORDER) == frozen["sids"] and frozen["sources"] == original
    assert all(study.sha(study.ext.sourcepath(sid)) == original[sid] for sid in SID_ORDER)
    source = {
        p.name: study.sha(p)
        for p in (SOURCES / name for name in ["module.cpp", "geometry.hpp", "first.hpp", "next.hpp"])
    }
    bind = Path(torch.__file__).resolve().parent / "include/pybind11"
    inc = Path(sysconfig.get_path("include")).resolve()
    spec = dict(
        source=source,
        original=original,
        flags=FLAGS,
        soabi=sysconfig.get_config_var("SOABI"),
        compiler=study.sha("/usr/bin/g++"),
        python_headers=digest({str(p.relative_to(inc)): study.sha(p) for p in inc.rglob("*.h")}),
        pybind_headers=digest({str(p.relative_to(bind)): study.sha(p) for p in bind.rglob("*.h")}),
    )
    key = digest(spec)
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / key
    with (CACHE / (key + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
                working = Path(tmp)
                command = [
                    "prlimit",
                    "--as=3221225472",
                    "--cpu=180",
                    "--",
                    "bwrap",
                    "--unshare-all",
                    "--die-with-parent",
                    "--new-session",
                    "--cap-drop",
                    "ALL",
                    "--clearenv",
                    "--setenv",
                    "PATH",
                    "/usr/bin:/bin",
                ]
                for path in ["/usr", "/lib", "/lib64"]:
                    if Path(path).exists():
                        command.extend(["--ro-bind", path, path])
                command += [
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--tmpfs",
                    "/tmp",
                    "--ro-bind",
                    str(inc),
                    "/pyheaders",
                    "--ro-bind",
                    str(bind),
                    "/binding/pybind11",
                    "--ro-bind",
                    str(SOURCES.resolve()),
                    "/src",
                    "--bind",
                    tmp,
                    "/build",
                    "--chdir",
                    "/build",
                    "/usr/bin/g++",
                    *FLAGS,
                    "-I/pyheaders",
                    "-I/binding",
                    "/src/module.cpp",
                    "-o",
                    "/build/_regional_cap_cpp.so",
                ]
                with (working / "build.log").open("wb") as stream:
                    proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=210)
                if proc.returncode:
                    raise RuntimeError((working / "build.log").read_text()[-12000:])
                binary = study.sha(working / "_regional_cap_cpp.so")
                study.save(working / "manifest.json", dict(key=key, binary=binary, spec=spec))
                working.rename(dest)
        manifest = study.read(dest / "manifest.json")
        assert manifest["key"] == key and study.sha(dest / "_regional_cap_cpp.so") == manifest["binary"]
    return dest, manifest


def load():
    global NATIVE
    directory, manifest = build()
    if NATIVE is None:
        spec = importlib.util.spec_from_file_location(
            "_regional_cap_cpp", directory / "_regional_cap_cpp.so"
        )
        NATIVE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(NATIVE)
    return NATIVE, manifest


class Proposals:
    """Preserve GA feature construction and update all ten independent CPP policies once."""

    def __init__(self):
        self.module, _ = load()
        self.memory = [self.module.initial_memory(i) for i in range(10)]
        self.geometry = study.dq.RelativeGeometry(8)

    def panel(self, history):
        dq = study.dq
        event = dq.current_event(history)
        obs = event["observation"]
        self.geometry.advance(history)
        event = dict(
            event,
            executed_ticks=None if history.previous_tick is None else obs["tick"] - history.previous_tick,
        )
        raw = self.module.advance_all(self.memory, event)
        advice = np.full((10, 86), np.nan, dtype=np.float32)
        advice[:, -1] = 0
        pool = []
        sources = []
        for i, action in enumerate(raw):
            if action is None:
                continue
            if action in pool:
                pos = pool.index(action)
                advice[i] = advice[sources[pos][0]]
                sources[pos].append(i)
                continue
            try:
                dq.validate_action(action, obs)
            except (ValueError, KeyError, TypeError):
                continue
            advice[i, :69] = dq.action_encoding(action, self.geometry.goal_id, self.geometry.theta)
            advice[i, 69:85] = self.geometry.candidate_geometry(action)
            advice[i, -1] = 1
            pool.append(action)
            sources.append([i])
        if not pool:
            action = dict(target_id=None, a0=44, a1=44, ticks=1)
            dq.validate_action(action, obs)
            pool = [action]
            sources = [[]]
        packet = dict(
            g=self.geometry.vector,
            advice=advice,
            pool=pool,
            sources=sources,
            trace=[],
            budget=[(1800 - obs["tick"]) / 1800, (300 - obs["decision"]) / 300],
        )
        return dq.flatten(packet), *dq.full_data.candidate_arrays(packet, self), pool, raw
