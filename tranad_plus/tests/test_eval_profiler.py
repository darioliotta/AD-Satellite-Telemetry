import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from profile_evaluation_a100 import alpha_scores

def test_alpha_scores_equal_original_combination():
    torch.manual_seed(4)
    z1=torch.randn(1,7,3)
    z2=torch.randn(1,7,3)
    target=torch.randn(1,7,3)
    alphas=torch.linspace(0,1,21)
    actual=alpha_scores(z1,z2,target,alphas=alphas)
    expected=torch.stack([((a*z1+(1-a)*z2-target)**2).mean(dim=(0,2)) for a in alphas])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    assert actual.shape==(21,7)
