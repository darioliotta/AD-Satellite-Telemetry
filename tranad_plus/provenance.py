"""Verify historical parents across the documented package relocation."""
import json
from pathlib import Path
from .core.common import hash_file


def compatible_core_sources(old, current):
    """Permit only recorded before->after hashes, never arbitrary source changes."""
    left = {k: v for k, v in old.items() if k != 'planning.py'}
    right = {k: v for k, v in current.items() if k != 'planning.py'}
    if left.keys() != right.keys():
        return False
    root = Path(__file__).resolve().parent
    audit = json.loads((root / 'relocation.json').read_text())['files']
    for name, previous in left.items():
        actual = right[name]
        if hash_file(root / 'core' / name) != actual:
            return False
        if previous == actual:
            continue
        entry = audit.get('tranad_plus/core/' + name, {})
        if (entry.get('before_sha256'), entry.get('after_sha256')) != (previous, actual):
            return False
    return True
