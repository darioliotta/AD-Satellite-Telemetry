"""Show the live V15 campaign, active configuration, epoch and estimated time."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1]/'results/v15')
    args = p.parse_args()
    path = args.output_root/'campaign_latest.json'
    if not path.exists():
        raise SystemExit(f'No V15 campaign yet: {path}')
    report = json.loads(path.read_text())
    done = sum(r['status'] in ('complete', 'skipped_verified_complete') for r in report['runs'])
    failed = [r for r in report['runs'] if r['status']=='failed']
    print(f"V15 {report['status']} | {done}/{report['planned_jobs']} complete | failed={len(failed)}")
    current = report.get('current_job')
    if current:
        print('Current: '+current)
        live = args.output_root/current/'progress.json'
        if live.exists():
            print(json.dumps(json.loads(live.read_text()), indent=2))
    for record in failed:
        print(record['id']+': '+record['reason'])


if __name__ == '__main__':
    main()
