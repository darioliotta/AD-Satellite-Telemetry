"""Dynamic per-vector W8A8/W4A4 for every projection and QK/AV product."""
import json
import math
import numpy as np
import torch
from torch import nn
from .kernel import int_bmm
from tranad_plus.core.common import atomic_destination
from tranad_plus.core.models import build_model


def quantize(x, bits, dim=-1):
    if bits not in (4, 8):
        raise ValueError('Only 4/8-bit signed integer precision')
    limit = 2**(bits-1)-1
    # Per vector: a sample never uses statistics from other batch samples.
    scale = x.detach().abs().amax(dim=dim, keepdim=True) / limit
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(x / scale).clamp(-limit, limit).to(torch.int8)
    return codes, scale


def pack4(q):
    flat = (q.reshape(-1).to(torch.int16) & 15).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, flat.new_zeros(1)))
    return flat[::2] | (flat[1::2] << 4)


def unpack4(packed, shape):
    flat = torch.stack((packed & 15, packed >> 4), dim=1).reshape(-1).to(torch.int8)
    flat = torch.where(flat >= 8, flat - 16, flat)
    return flat[:math.prod(shape)].reshape(shape)


def product(a, b, bits, backend):
    """Batched matrix product, with FP32 scaling only AFTER INT32 accumulation."""
    qa, sa = quantize(a, bits, -1)
    qb, sb = quantize(b, bits, -2)
    return int_bmm(qa, qb, backend).to(torch.float32) * sa * sb


class IntegerLinear(nn.Module):
    def __init__(self, weight, bias, bits, backend):
        super().__init__()
        self.bits, self.backend = bits, backend
        self.out_features, self.in_features = weight.shape
        q, scale = quantize(weight.detach().float(), bits)
        self.register_buffer('codes', pack4(q) if bits == 4 else q.clone())
        self.register_buffer('scale', scale.clone())
        self.register_buffer('bias', None if bias is None else bias.detach().float().clone())

    def forward(self, x):
        qx, scale = quantize(x.reshape(-1, self.in_features), self.bits)
        qw = unpack4(self.codes, (self.out_features, self.in_features)) if self.bits == 4 else self.codes
        integers = int_bmm(qx.unsqueeze(0), qw.t().unsqueeze(0), self.backend)[0]
        y = integers.float() * scale * self.scale.t()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


class IntegerAttention(nn.Module):
    batch_first = False
    _qkv_same_embed_dim = True
    def __init__(self, source, bits, backend):
        super().__init__()
        if source.batch_first or source.bias_k is not None or source.bias_v is not None or source.add_zero_attn:
            raise ValueError('Unsupported MHA mode; no fallback')
        self.embed_dim, self.num_heads = source.embed_dim, source.num_heads
        self.bits, self.backend = bits, backend
        weights = source.in_proj_weight.detach().chunk(3, dim=0)
        biases = source.in_proj_bias.detach().chunk(3) if source.in_proj_bias is not None else [None]*3
        self.q, self.k, self.v = [IntegerLinear(w, b, bits, backend) for w, b in zip(weights, biases)]
        self.out_proj = IntegerLinear(source.out_proj.weight, source.out_proj.bias, bits, backend)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if key_padding_mask is not None or attn_mask is not None or is_causal:
            raise ValueError('Unsupported attention mask; no fallback')
        t, batch, d = query.shape
        s, head = key.shape[0], d // self.num_heads
        def heads(x, projection):
            return projection(x).reshape(x.shape[0], batch, self.num_heads, head).permute(1, 2, 0, 3).reshape(-1, x.shape[0], head)
        q, k, v = heads(query, self.q), heads(key, self.k), heads(value, self.v)
        logits = product(q, k.transpose(1, 2), self.bits, self.backend) / math.sqrt(head)
        probabilities = torch.softmax(logits, dim=-1)
        out = product(probabilities, v, self.bits, self.backend)
        out = out.reshape(batch, self.num_heads, t, head).permute(2, 0, 1, 3).reshape(t, batch, d)
        weights = probabilities.reshape(batch, self.num_heads, t, s) if need_weights else None
        if need_weights and average_attn_weights:
            weights = weights.mean(1)
        return self.out_proj(out), weights


def convert(model, bits, backend='cuda_triton'):
    """In-place conversion; preserve shared decoder object identities."""
    seen = {}
    def visit(module):
        if id(module) in seen:
            return seen[id(module)]
        if isinstance(module, nn.MultiheadAttention):
            result = IntegerAttention(module, bits, backend)
        elif isinstance(module, nn.Linear):
            result = IntegerLinear(module.weight, module.bias, bits, backend)
        else:
            result = module
            for name, child in list(module._modules.items()):
                if child is not None:
                    module._modules[name] = visit(child)
        seen[id(module)] = result
        return result
    model = visit(model).eval()
    if any(isinstance(m, (nn.Linear, nn.MultiheadAttention)) for m in model.modules()):
        raise RuntimeError('Unconverted matrix operator')
    if any(p.ndim >= 2 for p in model.parameters()):
        raise RuntimeError('Floating-point learned matrix remains')
    return model


def export(model, profile, bits, path):
    entries, arrays, aliases, seen = [], {}, {}, {}
    for name, value in model.named_buffers(remove_duplicate=False):
        if id(value) in seen:
            aliases[name] = seen[id(value)]
            continue
        seen[id(value)] = name
        key = f't{len(entries)}'
        arrays[key] = value.detach().cpu().numpy()
        entries.append({'name': name, 'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype)})
    schema = {'schema_version': 1, 'profile': profile, 'bits': bits, 'tensors': entries, 'aliases': aliases,
              'execution': 'W8A8_INT8_MMA' if bits == 8 else 'W4A4_INT8_MMA',
              'scale_policy': 'weights_per_output_channel; activations_per_vector_dynamic',
              'integer_scope': 'QKV, attention QK and AV, attention output, FFN, reconstruction head',
              'floating_scope': 'scales, biases, residuals, position encoding, LeakyReLU, softmax, sigmoid, focus square',
              'native_INT4_kernel': False}
    arrays['schema'] = np.frombuffer(json.dumps(schema).encode(), dtype=np.uint8)
    with atomic_destination(path, binary=True) as stream:
        np.savez_compressed(stream, **arrays)
    size = sum(a.nbytes for k, a in arrays.items() if k != 'schema')
    return {**schema, 'format': 'int'+str(bits), 'tensor_bytes_including_bias_scales_buffers': size,
            'artifact_file_bytes': path.stat().st_size, 'matrix_coverage_fraction': 1.0,
            'resident_int4_packed_weights': bits == 4,
            'int4_temporary_execution_dtype': 'int8' if bits == 4 else None}


def load(path, device, backend='cuda_triton'):
    with np.load(path, allow_pickle=False) as archive:
        schema = json.loads(archive['schema'].tobytes())
        if schema['schema_version'] != 1:
            raise ValueError('Unknown integer export schema')
        model = convert(build_model(schema['profile']), schema['bits'], backend)
        state = {e['name']: torch.from_numpy(archive[e['key']].copy()) for e in schema['tensors']}
        state.update({a: state[b] for a, b in schema['aliases'].items()})
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()
