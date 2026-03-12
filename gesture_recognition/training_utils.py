from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor
from jaxtyping import Float, Int

def hierarchical_f1(preds: Int[Tensor, "bs"], targs: Int[Tensor, "bs"]):
    """Assumes that 0-7 are target gestures while others are non-target"""
    # Compute binary F1 (Target vs Non-Target)
    y_true_bi, y_pred_bi = targs < 8, preds < 8
    f1_binary = f1_score(
        y_true_bi, y_pred_bi, pos_label=True, zero_division=0, average='binary'
    )

    # Compute macro F1 over all gesture classes
    y_true_mc, y_pred_mc = torch.where(targs<8, targs, 8), torch.where(preds<8, preds, 8)
    f1_macro = f1_score(y_true_mc, y_pred_mc, zero_division=0, average='macro')

    return 0.5*f1_binary + 0.5*f1_macro

class MixUp:
    def __init__(self, n_classes: int, alpha: float=0.4):
        assert alpha > 0
        self.C = n_classes
        self.distrib = torch.distributions.Beta(alpha, alpha)

    def _interp_rot(self, rots1: Tensor, rots2: Tensor, lam: Tensor):
        """Implements shortest-distance nlerp for quaternion sequences"""
        dot = (rots1*rots2).sum(dim=-1, keepdim=True)
        rots2 = torch.where(dot<0, -rots2, rots2)
        rots_mix = torch.lerp(rots1, rots2, weight=lam[:, None, None])
        rots_mix = rots_mix / (rots_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return rots_mix

    def __call__(self, *xs: Float[Tensor, "bs 3 L"], y: Int[Tensor, "bs"]):
        bs = xs[0].size(0)
        device = xs[0].device
        lam = self.distrib.sample((bs,)).to(device)
        lam = torch.where(lam>0.5, lam, 1-lam)
        perm = torch.randperm(bs, device=device)

        xs_mix = [torch.lerp(x, x[perm], weight=lam[:, None, None]) for x in xs]
        y = F.one_hot(y, self.C).to(lam.dtype)
        y_mix = torch.lerp(y, y[perm], weight=lam[:, None])

        return *xs_mix, y_mix

def loss_fn(logits: Float[Tensor, "bs n_classes"], targs: Tensor) -> Tensor:
    """Implements soft-target cross entropy. `targs` can be class idxs or probs"""
    if len(targs.shape) == 1:
        targs = F.one_hot(targs, logits.size(1))
    logp = F.log_softmax(logits, dim=-1)
    return -(targs * logp).sum(dim=-1).mean()

def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, it: int, out: str|Path
):
    obj = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), it=it)
    torch.save(obj, out)

def load_checkpoint(
    src: str|Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj["it"]