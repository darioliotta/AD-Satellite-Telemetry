"""Audited FP32 batching correction; preserve completed V17 experiments intact."""
from copy import deepcopy
import json
from pathlib import Path
from tranad_plus.core.common import atomic_destination, digest, hash_file, now, write_json

PATCH = Path(__file__).with_name('fp32_batch_recovery.json')


def batch_policy(spec, profile, parent):
    ds = profile['dataset']
    original = parent['evaluation_settings']['score_batch_size_by_dataset'][ds]
    integer = spec['evaluation']['score_batch_size_by_dataset'][ds]
    if not isinstance(original, int) or isinstance(original, bool) or original < 1:
        raise ValueError('Invalid recorded parent score batch size')
    return {'revision': 'fp32_parent_batch_v1', 'fp32_score_batch_size': original,
            'integer_score_batch_size': integer, 'score_tolerances_unchanged': True,
            'source_transition_sha256':hash_file(PATCH)}


def evaluation_settings(spec, profile, parent, format_name):
    policy = batch_policy(spec, profile, parent)
    result = deepcopy(spec['evaluation'])
    if format_name == 'fp32':
        result['score_batch_size_by_dataset'][profile['dataset']] = policy['fp32_score_batch_size']
    elif format_name not in spec['formats']:
        raise ValueError('Unknown precision')
    return result


def verify_artifacts(folder, status):
    folder = Path(folder)
    for name, expected in status['artifacts'].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise RuntimeError('Artifact path escapes experiment directory')
        if not path.is_file() or hash_file(path) != expected:
            raise RuntimeError('Changed result artifact: '+str(path))


def approved_transition(previous, current):
    """Only this recorded source revision, identical data/parent/hardware/plan."""
    patch = json.loads(PATCH.read_text())
    if patch.get('schema_version') != 1 or patch.get('id') != 'v17_fp32_parent_batch_v1':
        raise RuntimeError('Unknown recovery patch')
    if current['sources'] != patch['after_sources']:
        raise RuntimeError('Current code differs from the audited FP32 batch fix')
    allowed = patch['allowed_before_sources']
    if previous['sources'].keys() != allowed.keys():
        raise RuntimeError('Historical source file set is not an approved V17 predecessor')
    for name, value in previous['sources'].items():
        if value not in allowed[name]:
            raise RuntimeError('Historical source hash is not approved: '+name)
    if 'evaluation_policy' in previous:
        raise RuntimeError('Only the original V17 binding can use this recovery transition')
    policy = current.get('evaluation_policy', {})
    if policy.get('revision') != 'fp32_parent_batch_v1' or policy.get('score_tolerances_unchanged') is not True:
        raise RuntimeError('Invalid recovery evaluation policy')
    if policy.get('source_transition_sha256') != hash_file(PATCH):
        raise RuntimeError('Recovery patch hash differs from the current binding')
    left = {k:v for k,v in previous.items() if k != 'sources'}
    right = {k:v for k,v in current.items() if k not in ('sources','evaluation_policy')}
    if left != right:
        raise RuntimeError('Recovery cannot change data, checkpoint, profile, plan or hardware')
    ds = current['profile']['dataset']
    if policy['integer_score_batch_size'] != current['plan']['evaluation']['score_batch_size_by_dataset'][ds]:
        raise RuntimeError('Integer evaluation batch changed')
    return patch['id']


def inspect_existing(folder, binding):
    """Read-only: verify an existing run before deciding skip, resume or rebind."""
    folder = Path(folder)
    path = folder / 'manifest.json'
    signature = digest(binding)
    if not path.exists():
        if (folder/'status.json').exists():
            raise RuntimeError('Experiment has a status but no source manifest')
        return {'action': 'new', 'signature': signature}
    manifest = json.loads(path.read_text())
    previous = manifest['binding']
    old_signature = manifest['signature']
    if digest(previous) != old_signature:
        raise RuntimeError('Stored manifest signature does not match its binding')
    status_path = folder/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    if status and status['signature'] != old_signature:
        # An interruption between the two atomic manifest/status writes is
        # recoverable only through this exact recorded transition.
        if (old_signature == signature and status['status'] != 'complete'
                and status['signature'] == manifest.get('recovery_from_signature')):
            record = folder / manifest['recovery_record']
            if not record.resolve().is_relative_to((folder/'recovery').resolve()):
                raise RuntimeError('Recovery record escapes its archive')
            transition = json.loads(record.read_text())
            archived = json.loads((record.parent/'manifest.json').read_text())
            if (transition['new_signature'] != signature
                    or transition['old_signature'] != status['signature']
                    or digest(archived['binding']) != status['signature']
                    or approved_transition(archived['binding'], binding) != transition['patch_id']):
                raise RuntimeError('Interrupted recovery record differs')
            verify_artifacts(record.parent, {'artifacts':transition['archived_artifacts']})
            return {'action':'finish_rebind_status', 'signature':signature,
                    'previous_signature':status['signature'], 'patch_id':transition['patch_id']}
        raise RuntimeError('Stored status/manifest signatures differ')
    complete = status and status['status'] == 'complete'
    if old_signature == signature:
        if complete:
            verify_artifacts(folder, status)
        elif manifest.get('recovery_record'):
            record = folder/manifest['recovery_record']
            if not record.resolve().is_relative_to((folder/'recovery').resolve()):
                raise RuntimeError('Recovery record escapes its archive')
            transition = json.loads(record.read_text())
            if transition['new_signature'] != signature:
                raise RuntimeError('Recovery signature differs')
            verify_artifacts(record.parent, {'artifacts':transition['archived_artifacts']})
        return {'action': 'skip_complete' if complete else 'resume', 'signature': signature}
    patch_id = approved_transition(previous, binding)
    if complete:
        verify_artifacts(folder, status)
    return {'action': 'preserve_complete' if complete else 'rebind_incomplete',
            'signature': signature, 'previous_signature': old_signature, 'patch_id': patch_id,
            'previous_fp32_batch_size': previous['plan']['evaluation']['score_batch_size_by_dataset'][previous['profile']['dataset']]}


def rebind_incomplete(folder, binding):
    """Called under the job lock; archive a failed attempt, then bind new work."""
    folder = Path(folder)
    decision = inspect_existing(folder, binding)
    if decision['action'] != 'rebind_incomplete':
        raise RuntimeError('Recovery requires a verified incomplete predecessor')
    archive = folder/'recovery'/decision['previous_signature']
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json','status.json','error.json','progress.json','metrics.json'):
        source, target = folder/name, archive/name
        if source.exists() and not target.exists():
            with atomic_destination(target, binary=True) as stream:
                stream.write(source.read_bytes())
        if name == 'manifest.json' and hash_file(source) != hash_file(target):
            raise RuntimeError('Recovery manifest archive changed')
    # Incomplete format cases are rerun under the new binding, never silently
    # relabelled. Finished jobs are never passed to this function.
    for name in ('evaluation','models'):
        source, target = folder/name, archive/name
        if source.exists():
            if target.exists():
                raise RuntimeError('Ambiguous partial recovery archive: '+str(target))
            source.rename(target)
    # Keep errors from old attempts in the archive, rather than presenting a
    # stale parity error as the current job's outcome after a successful retry.
    error = folder/'error.json'
    if error.exists():
        target = archive/('error-'+hash_file(error)+'.json')
        if not target.exists():
            with atomic_destination(target, binary=True) as stream:
                stream.write(error.read_bytes())
        if hash_file(error) != hash_file(target):
            raise RuntimeError('Error archive differs')
        error.unlink()
    transition = archive/'transition.json'
    if transition.exists():
        saved = json.loads(transition.read_text())
        if saved['new_signature'] != decision['signature'] or saved['patch_id'] != decision['patch_id']:
            raise RuntimeError('Conflicting recovery transition')
    else:
        write_json(transition, {'patch_id':decision['patch_id'], 'created':now(),
                              'old_signature':decision['previous_signature'], 'new_signature':decision['signature'],
                              'evaluation_policy':binding['evaluation_policy'],
                              'archived_artifacts':{str(p.relative_to(archive)):hash_file(p)
                                                    for p in sorted(archive.rglob('*')) if p.is_file()}})
    write_json(folder/'manifest.json', {'signature':decision['signature'], 'binding':binding,
                                       'created':now(), 'recovery_from_signature':decision['previous_signature'],
                                       'recovery_record':str(transition.relative_to(folder))})
    # Pair status with the new manifest before any evaluation resumes.
    write_json(folder/'status.json', {'status':'running', 'signature':decision['signature'], 'updated':now()})
    return decision
