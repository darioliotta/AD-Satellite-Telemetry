#!/usr/bin/env python3
"""One current source tree, independent versioned result campaigns.

Run from tranad_plus/: python -B run_campaign.py integer --dry-run
"""
from pathlib import Path
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
CAMPAIGNS = {
    'architectures': 'tranad_plus.core.runner',
    'exploration': 'tranad_plus.campaigns.exploration.run',
    'temporal': 'tranad_plus.campaigns.temporal.run',
    'precision': 'tranad_plus.campaigns.precision.run',
    'integer': 'tranad_plus.campaigns.integer.run',
    'thresholds': 'tranad_plus.campaigns.thresholds.run',
    'regularization': 'tranad_plus.campaigns.regularization.run',
    'mechanisms': 'tranad_plus.campaigns.mechanisms.run',
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print('Usage: python -B run_campaign.py {' + ','.join(CAMPAIGNS) + '} [campaign options]')
        return 0
    name = argv.pop(0)
    if name not in CAMPAIGNS:
        raise SystemExit('Unknown campaign: ' + name)
    return importlib.import_module(CAMPAIGNS[name]).main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
