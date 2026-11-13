#!/usr/bin/env python3
"""Remove obsolete sibling source roots after installing the unified tree.

Never removes results/, data/, checkpoints, archives or generic backups.
Dry-run by default. Use --apply after copying the current tranad_plus/.
"""
import argparse
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    current=Path(__file__).resolve().parent
    if not (current/'campaigns/integer/run.py').is_file() or not (current/'core/models/__init__.py').is_file():
        raise SystemExit('Unified installation is incomplete; refusing cleanup')
    targets=[]
    for name in ('tranad_variants','tranad_v14','tranad_v15','tranad_v16'):
        path=current.parent/name
        if not path.exists():continue
        if path.is_symlink():raise SystemExit('Refusing symlink: '+str(path))
        for file in path.rglob('*'):
            if file.is_symlink():raise SystemExit('Review symlink before removing: '+str(file))
            if file.is_file() and (file.suffix.lower() in ('.ckpt','.pt','.pth','.pkl','.npy','.npz','.csv','.pdf','.png','.svg','.tar','.gz','.zip')
                                   or any(n in ('data','results','checkpoints') for n in file.relative_to(path).parts)):
                raise SystemExit('Research artifact found; move/review it before cleanup: '+str(file))
        targets.append(path)
    for path in targets:
        print(('REMOVE ' if args.apply else 'WOULD REMOVE ')+str(path))
    if args.apply:
        for path in targets:shutil.rmtree(path)
    print('Obsolete source roots: '+str(len(targets))+'. results/ and data/ are untouched.')


if __name__=='__main__':main()
