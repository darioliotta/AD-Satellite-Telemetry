import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tranad_plus.core.models import build_model
from . import runtime
from .kernel import int_bmm


class NoFloatGemm(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if any(op in str(func) for op in ('aten.mm.', 'aten.bmm.', 'aten.addmm.')):
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            if any(a.is_floating_point() for a in tensors):
                raise AssertionError('Floating-point GEMM: '+str(func))
        return func(*args, **(kwargs or {}))


def profile(variant):
    return dict(variant=variant, features=3, window_sz=5, dim_feedforward=7, batch_sz=4, num_layers=1)


class IntegerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pack_roundtrip_all_codes_and_odd_shapes(self):
        x = torch.arange(-7, 8, dtype=torch.int8)
        torch.testing.assert_close(runtime.unpack4(runtime.pack4(x), x.shape), x, rtol=0, atol=0)
        self.assertEqual(runtime.pack4(x).numel(), 8)

    def test_plan_is_balanced_and_rejects_old_outputs(self):
        from .run import arguments, plan, validate_output
        from tranad_plus.core.bootstrap import PROJECT_ROOT
        args=arguments(['--dry-run']);spec=json.loads(args.plan.read_text());jobs=plan(spec,args)
        self.assertEqual(len(jobs),240);self.assertEqual(len(set(jobs)),240)
        for index,key,count in [(0,'datasets',80),(1,'variants',60),(2,'configurations',60)]:
            for name in spec[key]:self.assertEqual(sum(Path(j).parts[index]==name for j in jobs),count)
        for folder in ('results/v16','results/v13/architectures','data','tranad_plus'):
            with self.assertRaises(ValueError):
                validate_output(arguments(['--output-root',str(PROJECT_ROOT/folder)]))

    def test_real_dataset_dimensions(self):
        from tranad_plus.core.bootstrap import ROOT
        spec=json.loads((ROOT/'configs/baselines.json').read_text())
        for ds in spec['datasets'].values():
            p={**profile('encoder_only'),**ds['configurations']['reference'],'features':ds['features']}
            for variant in ('reference','shared_decoder','single_pass','encoder_only'):
                p['variant']=variant
                for bits in (4,8):
                    model=runtime.convert(build_model(p),bits,'reference_cpu')
                    x=torch.rand(p['window_sz'],2,p['features'])
                    with torch.inference_mode(),NoFloatGemm():out=model(x,x[-1:])
                    for value in out if isinstance(out,tuple) else (out,):
                        self.assertEqual(value.shape,(1,2,p['features']))
                        self.assertTrue(torch.isfinite(value).all())

    def test_compact_evaluation_and_cache(self):
        import numpy as np
        from tranad_plus.core.data import DataBundle
        from tranad_plus.campaigns.precision.evaluation import evaluate
        from tranad_plus.core.bootstrap import PROJECT_ROOT
        p={**profile('encoder_only'),'dataset':'SMAP','configuration':'compact','seed':2026,'pot_q':.01}
        class Dataset(torch.utils.data.Dataset):
            windows=torch.rand(64,5,3)
            def __len__(self):return len(self.windows)
            def __getitem__(self,i):return self.windows[i]
        bundle=DataBundle({'train':Dataset(),'test':Dataset()},
                         {'train':np.zeros(64,dtype=np.int8),'test':np.r_[np.zeros(32,dtype=np.int8),np.ones(32,dtype=np.int8)]},
                         {'synthetic':True})
        settings={'score_batch_size_by_dataset':{'SMAP':16},'latency_warmup':1,'latency_iterations':2,'profile_operators':False}
        model=runtime.convert(build_model(p),4,'reference_cpu')
        with tempfile.TemporaryDirectory() as tmp:
            packed=Path(tmp)/'model.npz';storage=runtime.export(model,p,4,packed)
            folder=Path(tmp)/'job/evaluation/int4'
            with patch.dict('os.environ',{'V16_NO_PROGRESS':'1'}):
                result=evaluate(model,bundle,p,folder,torch.device('cpu'),settings,None,None,PROJECT_ROOT/'data',storage)
                cached=evaluate(model,bundle,p,folder,torch.device('cpu'),settings,None,None,PROJECT_ROOT/'data',storage)
            self.assertEqual(result['oracle'],cached['oracle'])
            self.assertIn('bf',result['oracle']);self.assertIn('pot',result['oracle'])
            self.assertNotIn('bf_dense',result['oracle'])
            self.assertTrue((folder/'latency_samples.npz').exists())
            self.assertFalse((folder/'scores.npz').exists())
            self.assertFalse(result['resources']['not_integer_kernel_latency'])

    def test_exact_integer_accumulation_and_padding_shapes(self):
        a = torch.randint(-127, 128, (3, 13, 55), dtype=torch.int8)
        b = torch.randint(-127, 128, (1, 55, 25), dtype=torch.int8)
        with NoFloatGemm():
            actual = int_bmm(a, b, 'reference_cpu')
        torch.testing.assert_close(actual, torch.bmm(a.long(), b.expand(3, -1, -1).long()).int(), rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            int_bmm(a, b)  # Never select a CPU fallback implicitly.
        with self.assertRaises(TypeError):
            int_bmm(a.float(), b.float(), 'reference_cpu')

    def test_attention_layout_matches_original_without_quantization(self):
        original = torch.nn.MultiheadAttention(6, 3, dropout=0).eval()
        model = runtime.IntegerAttention(original, 8, 'reference_cpu')
        weights = original.in_proj_weight.chunk(3)
        biases = original.in_proj_bias.chunk(3)
        for name, w, b in zip(('q', 'k', 'v'), weights, biases):
            layer = torch.nn.Linear(6, 6)
            layer.weight.data.copy_(w); layer.bias.data.copy_(b)
            setattr(model, name, layer)
        model.out_proj = original.out_proj
        for t, s in ((5, 5), (1, 5)):
            q, k, v = torch.randn(t, 4, 6), torch.randn(s, 4, 6), torch.randn(s, 4, 6)
            with torch.inference_mode(), patch.object(runtime, 'product', side_effect=lambda a,b,*args:torch.bmm(a,b)):
                actual, weights = model(q, k, v)
                expected, reference_weights = original(q, k, v)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(weights, reference_weights, rtol=2e-5, atol=2e-6)

    def test_all_architectures_no_float_gemm_sharing_export_batch_invariance(self):
        for variant in ('reference', 'shared_decoder', 'single_pass', 'encoder_only'):
            for bits in (4, 8):
                with self.subTest(variant=variant, bits=bits), tempfile.TemporaryDirectory() as tmp:
                    torch.manual_seed(2026)
                    p = profile(variant)
                    model = runtime.convert(build_model(p), bits, 'reference_cpu')
                    if variant == 'shared_decoder':
                        self.assertIs(model.transformer_decoder1, model.transformer_decoder2)
                    x = torch.rand(5, 4, 3)
                    with torch.inference_mode(), NoFloatGemm():
                        out = model(x, x[-1:])
                        one = model(x[:, :1], x[-1:, :1])
                    sequence = out if isinstance(out, tuple) else (out,)
                    single = one if isinstance(one, tuple) else (one,)
                    for a, b in zip(sequence, single):
                        self.assertTrue(torch.isfinite(a).all())
                        torch.testing.assert_close(a[:, :1], b, rtol=1e-5, atol=1e-6)
                    path = Path(tmp)/'model.npz'
                    info = runtime.export(model, p, bits, path)
                    loaded = runtime.load(path, 'cpu', 'reference_cpu')
                    with torch.inference_mode(), NoFloatGemm():
                        restored = loaded(x, x[-1:])
                    restored = restored if isinstance(restored, tuple) else (restored,)
                    for a, b in zip(sequence, restored):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertFalse(info['native_INT4_kernel'])

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA/A100 and Triton; preflight runs on server')
    def test_cuda_exact_and_all_architectures(self):
        from .kernel import preflight
        result = preflight('cuda:'+str(torch.cuda.current_device()))
        self.assertTrue(all(c['exact_integer_match'] for c in result['cases']))
        for variant in ('reference', 'shared_decoder', 'single_pass', 'encoder_only'):
            for bits in (4, 8):
                torch.manual_seed(2026)
                original = build_model(profile(variant))
                import copy
                cpu = runtime.convert(copy.deepcopy(original), bits, 'reference_cpu')
                gpu = runtime.convert(original, bits).cuda()
                x = torch.rand(5, 4, 3)
                with torch.inference_mode(), NoFloatGemm():
                    a = cpu(x, x[-1:]); b = gpu(x.cuda(), x[-1:].cuda())
                a = a if isinstance(a, tuple) else (a,)
                b = b if isinstance(b, tuple) else (b,)
                for expected, actual in zip(a, b):
                    torch.testing.assert_close(expected, actual.cpu(), rtol=3e-4, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
