"""Durable artifacts, source provenance and per-experiment locks."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

from .bootstrap import ROOT, BASELINE_ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def atomic_destination(path, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    with atomic_destination(path) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_torch(path, value):
    import torch
    with atomic_destination(path, binary=True) as stream:
        torch.save(value, stream)


def save_npz(path, **arrays):
    import numpy as np
    with atomic_destination(path, binary=True) as stream:
        np.savez_compressed(stream, **arrays)


@contextmanager
def run_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {folder}") from exc
        stream.seek(0)
        stream.truncate()
        json.dump({"pid": os.getpid(), "host": platform.node(), "started": now()}, stream)
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_manifest():
    """Pin numerical implementations; record host-specific path/catalog support."""
    frozen = json.loads((ROOT / "BASELINE_SOURCES.json").read_text())
    for name, expected in frozen["files"].items():
        source = BASELINE_ROOT / name
        if not source.is_file() or hash_file(source) != expected:
            raise RuntimeError(f"Baseline source differs from the pinned version: {source}")
    support = {name: hash_file(BASELINE_ROOT / name) for name in frozen.get("support_files", [])}
    local = {str(p.relative_to(ROOT)): hash_file(p)
             for p in sorted(ROOT.rglob("*.py"))
             if "tests" not in p.relative_to(ROOT).parts}
    return {"baseline": frozen["files"], "baseline_support": support, "variants": local,
            "sha256": digest({"baseline": frozen["files"], "support": support, "variants": local})}
