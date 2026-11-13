"""Read current job, phase and epoch without importing PyTorch."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[3]/'results/v16')
    root=p.parse_args().output_root;path=root/'campaign_latest.json'
    if not path.exists():raise SystemExit('No campaign yet: '+str(path))
    r=json.loads(path.read_text());done=sum(x['status'] in ('complete','skipped_verified_complete') for x in r['runs'])
    failed=[x for x in r['runs'] if x['status']=='failed']
    print(f"V16 {r['status']} | {done}/{r['planned_jobs']} complete | failed={len(failed)} | unavailable formats={r.get('tables',{}).get('unavailable_formats',0)}")
    job=r.get('current_job')
    if job:
        print('Current: '+job);progress=root/job/'progress.json'
        if progress.exists():
            live=json.loads(progress.read_text());print(json.dumps(live,indent=2))
            if live.get('detail_folder'):
                detail=root/job/live['detail_folder']/'progress.json'
                if detail.exists():print(json.dumps(json.loads(detail.read_text()),indent=2))
    for f in failed:print(f['id']+': '+f['reason'])


if __name__=='__main__':main()
