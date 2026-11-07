#!/usr/bin/env python3
"""SAFE, idempotent migration: old output trees -> project/results/legacy.

Default: preview only. --apply copies with SHA-256 validation and NEVER deletes
or overwrites divergent files. Source may be the original active code directory.
Do not run against a live training process or active checkpoint writers.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

TREES = {
    'results': Path(''),
    'logs': Path('_logs'),
    'TranADPlus/Checkpoints': Path('_native_checkpoints'),
    'TranADPlus/Pickles': Path('_native_pickles'),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path) -> str:
    # Preserve data even when interrupted. Never replace an existing distinct target.
    if dst.exists():
        if not dst.is_file() or digest(src) != digest(dst):
            raise RuntimeError(f'COLLISION; review manually: {src} -> {dst}')
        return 'identical'
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.migrate_', dir=str(dst.parent))
    os.close(fd)
    tmp = Path(temp)
    try:
        shutil.copy2(src, tmp)
        if digest(src) != digest(tmp):
            raise RuntimeError(f'Copy SHA-256 mismatch: {src}')
        if dst.exists():
            if digest(dst) != digest(tmp):
                raise RuntimeError(f'Concurrent collision: {dst}')
        else:
            os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return 'copied'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Original tranad_plus directory')
    parser.add_argument('--results', required=True, type=Path, help='Project-level results/ directory')
    parser.add_argument('--apply', action='store_true', help='Copy and SHA-256 verify; never delete original')
    args = parser.parse_args()
    old = args.source.expanduser().resolve()
    root = args.results.expanduser().resolve()
    if not old.is_dir(): parser.error(f'Not a directory: {old}')
    if root == old or old in root.parents:
        parser.error('Results destination must be outside the original tranad_plus code directory')
    counts={'copy':0,'identical':0,'skip_symlink':0}
    for original, new in TREES.items():
        folder=old/original
        if folder.is_symlink():
            print(f'SKIP directory symlink: {folder}');counts['skip_symlink']+=1;continue
        if not folder.is_dir():
            continue
        for item in sorted(folder.rglob('*')):
            if item.is_symlink():
                print(f'SKIP symlink (requires manual review): {item}')
                counts['skip_symlink']+=1
                continue
            if not item.is_file() or item.name == '.DS_Store' or item.name.startswith('._'):
                continue
            destination=root/'legacy'/new/item.relative_to(folder)
            if args.apply:
                kind=copy_verified(item,destination)
                counts['identical' if kind=='identical' else 'copy']+=1
            else:
                if destination.is_file():
                    kind='identical' if digest(item)==digest(destination) else 'COLLISION'
                else:kind='would copy'
                if kind=='COLLISION':
                    raise RuntimeError(f'COLLISION: {item} -> {destination}')
                counts['identical' if kind=='identical' else 'copy']+=1
            print(f'{"APPLY" if args.apply else "PLAN"}: {kind}: {item} -> {destination}')
    print('Summary:',counts)
    if not args.apply:
        print('DRY RUN ONLY. Add --apply after review, with all running writers stopped.')
    else:
        print('Copy verified; original files were NOT deleted. Review/backup before manual cleanup.')

if __name__=='__main__':main()
