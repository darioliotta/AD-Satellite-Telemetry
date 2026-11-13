"""Explicit weight-only quantization, including MultiheadAttention parameters.

INT8/INT4 payloads are packed on disk. Forward execution reconstructs FP32
weights; it is a numerical sensitivity experiment, not an integer-kernel speed
benchmark. QAT uses the same fixed parent scales and PyTorch fake-quant STE.
"""
import json
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from tranad_plus.core.common import atomic_destination


def recipes(model, bits):
    if bits not in (4, 8):
        raise ValueError('Only signed symmetric 4/8-bit weights are supported')
    limit = 2**(bits-1)-1
    result = {}
    for name, value in model.named_parameters():
        if value.ndim >= 2:
            largest = float(value.detach().abs().max().cpu())
            if not np.isfinite(largest):
                raise FloatingPointError('Nonfinite weight: '+name)
            scale = float(np.float32(largest/limit)) if largest else 1.0
            result[name] = max(scale, float(np.finfo(np.float32).tiny))
    return result


class WeightFakeQuant(nn.Module):
    def __init__(self, scale, bits):
        super().__init__()
        self.scale, self.limit = float(scale), 2**(bits-1)-1

    def forward(self, weight):
        return torch.fake_quantize_per_tensor_affine(weight, self.scale, 0, -self.limit, self.limit)


def apply_qat(model, scales, bits):
    # named_parameters removes aliases: the shared decoder is handled once.
    for name, scale in scales.items():
        owner_name, attribute = name.rsplit('.', 1)
        parametrize.register_parametrization(model.get_submodule(owner_name), attribute,
                                             WeightFakeQuant(scale, bits))
    return model


def master_copy(model):
    # Removing parametrizations from a deepcopy can mutate a dynamic class also
    # used by the original object (notably MultiheadAttention in PyTorch 2.2).
    # Rebuild a plain model and remap original parameters without touching it.
    from tranad_plus.core.models import build_model
    profile={'variant':model.variant,'features':model.n_feats,'window_sz':model.window_sz,
             'batch_sz':model.batch_sz,'num_layers':model.num_encoder_layers,
             'dim_feedforward':model.transformer_encoder.layers[0].linear1.out_features}
    with torch.random.fork_rng(devices=[]):result=build_model(profile)
    state={}
    for name,value in model.state_dict().items():
        if '.parametrizations.' in name:
            prefix,suffix=name.split('.parametrizations.',1)
            attribute,kind=suffix.split('.',1)
            if kind!='original':continue
            name=prefix+'.'+attribute
        state[name]=value.detach().cpu().clone()
    result.load_state_dict(state,strict=True)
    return result


def pack4(codes):
    values = (codes.reshape(-1).astype(np.int16)+7).astype(np.uint8)
    if len(values) % 2:
        values = np.r_[values, np.uint8(7)]
    return values[::2] | (values[1::2] << 4)


def unpack4(values, count):
    codes = np.empty(len(values)*2, dtype=np.int8)
    codes[::2] = (values & 15).astype(np.int8)-7
    codes[1::2] = (values >> 4).astype(np.int8)-7
    return codes[:count]


def export_model(model, path, profile, format_name, scales=None, provenance=None,*,measure_only=False):
    """Store unique parameters, explicit aliases, buffers and a JSON schema."""
    model = master_copy(model)
    bits = int(format_name[3:]) if format_name.startswith('int') else None
    scales = scales if scales is not None else (recipes(model, bits) if bits else {})
    arrays, entries, aliases, seen = {}, [], {}, {}
    total_parameters = matrix_parameters = quantized_parameters = 0
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            aliases[name] = seen[id(parameter)]
            continue
        seen[id(parameter)] = name
        key = 'p'+str(len(entries))
        value = parameter.detach().float().cpu()
        count = value.numel()
        total_parameters += count
        matrix_parameters += count if value.ndim >= 2 else 0
        item = {'name': name, 'key': key, 'shape': list(value.shape), 'count': count}
        if bits and value.ndim >= 2:
            scale = scales[name]
            limit = 2**(bits-1)-1
            rounded = torch.fake_quantize_per_tensor_affine(value, scale, 0, -limit, limit)
            codes = torch.round(rounded/scale).clamp(-limit, limit).to(torch.int8).numpy()
            arrays[key] = pack4(codes) if bits == 4 else codes
            arrays[key+'_scale'] = np.array(scale, dtype=np.float32)
            item['encoding'] = format_name
            error=(rounded-value).double()
            norm=float(value.double().square().sum())
            item['quantization_error']={'squared_error_sum':float(error.square().sum()),
                         'relative_L2':float(torch.sqrt(error.square().sum()/norm)) if norm else 0.,
                         'max_absolute_error':float(error.abs().max()),
                         'zero_code_fraction':float(np.mean(codes==0)),
                         'clipped_master_fraction':float((value.abs()>limit*scale).float().mean())}
            quantized_parameters += count
        else:
            dtype = np.float16 if format_name == 'fp16' else np.float32
            arrays[key] = value.numpy().astype(dtype)
            item['encoding'] = 'fp16' if format_name == 'fp16' else 'fp32'
        entries.append(item)
    buffers = []
    for i, (name, value) in enumerate(model.named_buffers()):
        key = 'b'+str(i)
        arrays[key] = value.detach().cpu().numpy().astype(np.float16 if format_name == 'fp16' else np.float32)
        buffers.append({'name': name, 'key': key})
    schema = {'schema_version': 1, 'profile': profile, 'format': format_name,
              'parameters': entries, 'aliases': aliases, 'buffers': buffers,
              'provenance': provenance or {}}
    tensor_bytes = sum(a.nbytes for a in arrays.values())
    arrays['schema_json'] = np.array(json.dumps(schema, sort_keys=True))
    # Uncompressed payloads: avoid conflating ZIP entropy compression with dtype savings.
    if measure_only:
        # The exact same NPZ container, without fsync or temporary disk output.
        # Preflight measures bytes; actual model exports remain durable below.
        with io.BytesIO() as stream:
            np.savez(stream, **arrays)
            file_bytes=stream.getbuffer().nbytes
    else:
        with atomic_destination(path, binary=True) as stream:
            np.savez(stream, **arrays)
        file_bytes=Path(path).stat().st_size
    return {'format': format_name, 'unique_parameters': total_parameters,
            'matrix_parameters': matrix_parameters, 'quantized_matrix_parameters': quantized_parameters,
            'matrix_coverage_fraction': quantized_parameters/matrix_parameters if bits else 0.0,
            'tensor_bytes_including_bias_scales_buffers': tensor_bytes,
            'artifact_file_bytes': file_bytes,
            'execution': 'dequantized_fp32_simulation' if bits else format_name,
            'includes_training_optimizer': False}


class HalfForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.half()
        self.output_count, self.variant = model.output_count, model.variant

    def forward(self, x, target):
        values = self.model(x.half(), target.half())
        return tuple(v.float() for v in values) if isinstance(values, tuple) else values.float()


def load_export(path, device):
    from tranad_plus.core.models import build_model
    with np.load(path, allow_pickle=False) as saved:
        schema = json.loads(str(saved['schema_json']))
        model = build_model(schema['profile'])
        state = model.state_dict()
        decoded = {}
        for item in schema['parameters']:
            key, encoding = item['key'], item['encoding']
            if encoding.startswith('int'):
                codes = unpack4(saved[key], item['count']) if encoding == 'int4' else saved[key]
                value = codes.astype(np.float32).reshape(item['shape'])*saved[key+'_scale']
            else:
                value = saved[key].astype(np.float32)
            decoded[item['name']] = torch.from_numpy(value.copy())
        for alias, original in schema['aliases'].items():
            decoded[alias] = decoded[original]
        for name, value in decoded.items():
            state[name] = value
        for item in schema['buffers']:
            state[item['name']] = torch.from_numpy(saved[item['key']].astype(np.float32))
        model.load_state_dict(state, strict=True)
    model.to(device)
    return HalfForward(model).to(device) if schema['format'] == 'fp16' else model
