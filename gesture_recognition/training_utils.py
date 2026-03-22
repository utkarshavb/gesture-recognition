from pathlib import Path
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor
from jaxtyping import Float, Int

def compute_loss(logits: Float[Tensor, "bs n_classes"], targs: Tensor) -> Tensor:
    """Implements soft-target cross entropy. `targs` can be class idxs or probs"""
    if len(targs.shape) == 1:
        targs = F.one_hot(targs, logits.size(1))
    logp = F.log_softmax(logits, dim=-1)
    return -(targs * logp).sum(dim=-1).mean()

def schedule_lr(
    step: int, lr_max: float, tot_steps: int, init_lr_frac: float=1e-4,
    warmup_frac: float=0.1, final_lr_frac: float=1e-4, warmup_strat: str="linear"
) -> float:
    """
    Implements 1cycle LR schedule; also allows for `warmup_strat="exp"`
    with `warmup_frac=1` to be able to perform the LR range test
    """
    init_lr = lr_max*init_lr_frac
    final_lr = lr_max*final_lr_frac
    last_step = tot_steps-1   # step: [0, tot_steps)
    warmup_steps = int(last_step*warmup_frac)
    
    # Phase 1: Warmup (or LR Range Test if warmup_frac=1.0)
    if step <= warmup_steps:
        if warmup_steps == 0:
            return lr_max
        p = step/warmup_steps
        if warmup_strat == "linear":
            return init_lr + (lr_max-init_lr)*p
        elif warmup_strat == "exp":
            return init_lr * (lr_max/init_lr)**p
        else:
            raise ValueError(f"Unknown warmup strategy: '{warmup_strat}'. Use 'linear' or 'exp'")
            
    # Phase 2: Cooldown (Cosine Annealing)
    else:
        cooldown_steps = last_step-warmup_steps
        p = (step-warmup_steps)/cooldown_steps
        return final_lr + 0.5*(lr_max-final_lr)*(1+math.cos(math.pi*p))

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

    return 0.5*(f1_binary+f1_macro)

def upside_down_aug(
    *imus: Float[Tensor, "bs 3 L"], thms: Float[Tensor, "bs 5 L"],
    tofs: Float[Tensor, "bs L 5 8 8"], p: float=0.0
) -> tuple[Tensor, ...]:
    bs, device = thms.size(0), thms.device
    mask = torch.rand(bs, device=device) < p

    # clone the tensors
    out_imus = tuple(x.clone() for x in imus)
    out_thms = thms.clone()
    out_tofs = tofs.clone()

    for x in out_imus:
        x[mask, :2, :] *= -1
    perm = [0, 3, 4, 1, 2]   # swap 2nd and 3rd with 4th and 5th sensors respectively
    out_thms[mask] = out_thms[mask][:, perm]
    out_tofs[mask] = out_tofs[mask][:, :, perm]
    out_tofs[mask] = out_tofs[mask].flip(-2, -1)   # rotate each tof sensor by 180
    
    return *out_imus, out_thms, out_tofs

class MixUp:
    def __init__(self, n_classes: int, alpha: float=0.4):
        self.C = n_classes
        self.distrib = torch.distributions.Beta(alpha, alpha) if alpha>0 else None

    def _expand_lam(self, lam: Float[Tensor, "bs"], x: Tensor):
        """Exapands dimensions of `lam` to be broadcastable to `x`"""
        shape = [lam.size(0)] + [1]*(x.ndim-1)
        return lam.view(*shape)

    def __call__(self, *xs: Tensor, y: Int[Tensor, "bs"]):
        if self.distrib is None:
            return *xs, y
        
        bs, device = xs[0].size(0), xs[0].device
        lam = self.distrib.sample((bs,)).to(device)
        lam = torch.where(lam>0.5, lam, 1-lam)
        perm = torch.randperm(bs, device=device)

        xs_mix = [torch.lerp(x, x[perm], weight=self._expand_lam(lam, x)) for x in xs]
        y = torch.nn.functional.one_hot(y, self.C).to(lam.dtype)
        y_mix = torch.lerp(y, y[perm], weight=lam[:, None])

        return *xs_mix, y_mix

def modality_dropout(
    thms: Float[Tensor, "bs 5 L"], tofs: Float[Tensor, "bs L 5 8 8"],
    p: float = 0.4, training: bool=True
) -> tuple[Tensor, ...]:
    """Drops proximity sensors with probability `p`"""
    if not training:
        p = 0.0
    bs, device = thms.size(0), thms.device
    drop_mask = torch.rand(bs, device=device) < p
    absent = tofs.isnan().any(dim=(1,2,3,4)) | thms.isnan().any(dim=(1,2))

    drop_mask = drop_mask | absent
    thms[drop_mask] = 0
    tofs[drop_mask] = 0

    return thms, tofs, ~drop_mask

def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, it: int, out: str|Path
):
    obj = dict(
        model=model.state_dict(), model_cfg=model.config,
        optimizer=optimizer.state_dict(), it=it
    )
    torch.save(obj, out)

def load_checkpoint(
    src: str|Path, model_cls: torch.nn.Module, optimizer: torch.optim.Optimizer|None=None
) -> tuple[torch.nn.Module, int]:
    obj = torch.load(src)
    model = model_cls(**obj["model_cfg"])
    model.load_state_dict(obj['model'])
    if optimizer is not None and 'optimizer' in obj:
        optimizer.load_state_dict(obj['optimizer'])
    return model, obj["it"]