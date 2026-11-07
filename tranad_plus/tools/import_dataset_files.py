#!/usr/bin/env python3
"""Safely place the supplied NASA SMAP/MSL archive and OPS-SAT segments.csv.

Only materializes the expected .npy and label CSV entries; no arbitrary ZIP
paths, silent overwrite, or dataset files inside the source tree. This is an
OPTIONAL first-time import, not preprocessing. See README for preprocessing.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as src:
        for chunk in iter(lambda:src.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def copy_verified(source,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if digest(source)!=digest(dest):raise FileExistsError(f'Destination exists with DIFFERENT content: {dest}')
        return False
    with tempfile.NamedTemporaryFile(dir=dest.parent,prefix='.import_',delete=False) as f:
        temporary=Path(f.name)
    try:
        shutil.copyfile(source,temporary)
        temporary.replace(dest)
    finally:temporary.unlink(missing_ok=True)
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True,help='archive.zip with data/data/{train,test}/*.npy')
    ap.add_argument('--segments',type=Path,required=True,help='OPS-SAT segments.csv')
    ap.add_argument('--data-root',type=Path,default=ROOT.parent/'data')
    args=ap.parse_args()
    data=args.data_root.expanduser().resolve()
    if not args.archive.is_file() or not args.segments.is_file():ap.error('Input ZIP or OPS-SAT CSV missing')
    root=data/'raw'/'SMAP_MSL';count=0
    with zipfile.ZipFile(args.archive) as archive:
        names=archive.namelist()
        selected=[]
        for n in names:
            parts=n.split('/')
            if (len(parts)==4 and parts[:2]==['data','data'] and
                parts[2] in ('train','test') and parts[3].endswith('.npy') and
                '/' not in parts[3] and '..' not in parts[3]):
                selected.append((n,Path(parts[2])/parts[3]))
            elif n=='labeled_anomalies.csv':
                selected.append((n,Path('labeled_anomalies.csv')))
        if len(selected)<100 or not any(str(p)=='labeled_anomalies.csv' for _,p in selected):
            raise ValueError('Archive does not contain the expected SMAP/MSL NASA dataset layout')
        for member,destination in selected:
            target=root/destination
            if target.exists():
                # Full hash checking applies also to existing files.
                with archive.open(member) as src:
                    check=hashlib.sha256()
                    for buf in iter(lambda:src.read(4*1024**2),b''):check.update(buf)
                if check.hexdigest()!=digest(target):raise FileExistsError(f'Conflicting raw file: {target}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.import_',delete=False) as f:
                temporary=Path(f.name)
            try:
                with archive.open(member) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
                temporary.replace(target)
                count+=1
            finally:temporary.unlink(missing_ok=True)
    created=copy_verified(args.segments,data/'raw'/'OPSSAT'/'segments.csv')
    print(f'Imported {count} SMAP/MSL files, OPS-SAT {"copied" if created else "already identical"}.')
    print('Raw dataset directory:',data/'raw')
    print('Next: python preprocess.py --dataset SMAP && python preprocess.py --dataset MSL')

if __name__=='__main__':main()
