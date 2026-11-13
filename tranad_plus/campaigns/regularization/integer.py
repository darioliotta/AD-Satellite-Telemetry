"""Integer PTQ of saved controlled checkpoints, with no duplicate weight archive."""
import json
from pathlib import Path
import torch

from tranad_plus.core.common import digest, hash_file, write_json
from tranad_plus.campaigns.integer.runtime import convert
from tranad_plus.campaigns.integer.kernel import kernel_audit
from .training import configured_model
from .evaluation import evaluate


def evaluate_integer(job, bundle, folder, signature, spec, device, *, backend='cuda_triton'):
    """CUDA production path only; reference CPU kernels are reserved for tests."""
    if backend not in ('cuda_triton', 'reference_cpu'):
        raise ValueError('Unknown integer backend')
    if backend == 'cuda_triton' and torch.device(device).type != 'cuda':
        raise RuntimeError('Integer study requires CUDA/Triton; no floating-point fallback')
    folder = Path(folder)
    trained = json.loads((folder/'training.json').read_text())
    for checkpoint in trained['snapshots']:
        if checkpoint not in spec['integer']['checkpoints']:
            continue
        source = folder/'checkpoints'/f'{checkpoint}.ckpt'
        for bits in spec['integer']['bits']:
            destination = folder/'integer'/checkpoint/f'int{bits}'
            done = destination/'status.json'
            quant_signature = digest({'training': signature, 'checkpoint_sha256': hash_file(source),
                                      'bits': bits, 'integer_recipe': 'v17_dynamic_WA_integer_matrices', 'backend': backend})
            if done.exists():
                status = json.loads(done.read_text())
                if status['signature'] != quant_signature:
                    raise RuntimeError('Integer endpoint provenance mismatch')
                if status['status'] == 'complete':
                    for name, expected in status['artifacts'].items():
                        if hash_file(destination/name) != expected:
                            raise RuntimeError('Changed integer artifact')
                    continue
            print(f"INTEGER | {job['id']} | {checkpoint} | W{bits}A{bits}", flush=True)
            saved = torch.load(source, map_location='cpu', weights_only=True)
            if saved['signature'] != signature:
                raise RuntimeError('Integer parent checkpoint mismatch')
            model = configured_model(job['profile'], job['arm'])
            model.load_state_dict(saved['model_state'], strict=True)
            model = convert(model, bits, backend).to(device)
            evaluate(model, bundle, job['profile'], destination, device, spec)
            buffers = list(model.buffers())
            write_json(destination/'storage.json', {
                'bits': bits, 'execution': (f'W{bits}A{bits}_INT8_MMA_INT32_accumulation'
                                           if backend == 'cuda_triton' else 'CPU_integer_reference_test_only'),
                'native_INT4_instructions': False,
                'resident_buffers_bytes': sum(v.numel()*v.element_size() for v in buffers),
                'duplicate_export_saved': False,
                'regeneration': 'Deterministic integer.convert from pinned FP32 checkpoint',
                'floating_operations': 'scales,biases,residuals,position,softmax,sigmoid,LeakyReLU,focus square'})
            write_json(destination/'kernel_audit.json', {'compiled_integer_kernels': kernel_audit()})
            artifacts = {p.name: hash_file(p) for p in destination.glob('*.json') if p.name != 'status.json'}
            write_json(done, {'status': 'complete', 'signature': quant_signature, 'artifacts': artifacts})
