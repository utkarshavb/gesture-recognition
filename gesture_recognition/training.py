from tqdm import tqdm
import torch
from collections.abc import Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from gesture_recognition.model import Model
from gesture_recognition.training_utils import (
    compute_loss, hierarchical_f1, MixUp, upside_down_aug, modality_dropout
)

@torch.inference_mode()
def valid_loop(dl: DataLoader, model: Model, device: str|None=None):
    model.eval()
    tot_loss, tot_samples = 0, 0
    all_preds, all_targs = [], []
    for *imus, thms, tofs, y in tqdm(dl, desc="Validating", leave=False):
        *imus, thms, tofs = tuple(x.to(device, non_blocking=True) for x in [*imus,thms,tofs])
        y = y.to(device, non_blocking=True)
        thms, tofs, proximity_mask = modality_dropout(thms, tofs, training=False)
        logits = model(*imus, thms=thms, tofs=tofs, proximity_mask=proximity_mask)
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu())
        all_targs.append(y.cpu())
        
        loss = compute_loss(logits, y)
        samples = y.size(0)
        tot_loss += samples*loss.item()
        tot_samples += samples
    all_preds = torch.cat(all_preds, dim=0)
    all_targs = torch.cat(all_targs, dim=0)
    f1 = hierarchical_f1(all_preds, all_targs)
    tot_loss /= max(1, tot_samples)
    model.train()
    return tot_loss, f1

def train(
    model: Model, train_dl: DataLoader, num_steps: int, optimizer: Optimizer,
    lr_scheduler: Callable, mixup: MixUp, valid_dl: DataLoader|None=None,
    log_fn: Callable|None=None, p_flip: float=0.0, p_proximity_drop: float=0.4,
    device: str="cpu", verbose: bool=True
):
    model.train()
    train_dl_iter = iter(train_dl)
    steps_per_epoch = len(train_dl)
    final_loss, final_f1 = 0, 0

    for step in range(num_steps):
        *imus, thms, tofs, y = next(train_dl_iter)
        *imus, thms, tofs = tuple(x.to(device, non_blocking=True) for x in [*imus,thms,tofs])
        y = y.to(device, non_blocking=True)
        *imus, thms, tofs = upside_down_aug(*imus, thms=thms, tofs=tofs, p=p_flip)
        thms, tofs, proximity_mask = modality_dropout(thms, tofs, p=p_proximity_drop)
        *imus, thms, tofs, y = mixup(*imus, thms, tofs, y=y)
        
        lr = lr_scheduler(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        
        logits = model(*imus, thms=thms, tofs=tofs, proximity_mask=proximity_mask)
        loss = compute_loss(logits, y)
        if not torch.isfinite(loss).item():
            print("loss not finite, terminating training")
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log
        log = {"train/loss":loss.detach().item(), "lr": lr}
        end_of_epoch = (step+1)%steps_per_epoch==0
        if end_of_epoch:
            train_dl_iter = iter(train_dl)
            # TODO: compute and log metric on training set
            if valid_dl is not None:
                valid_loss, valid_f1 = valid_loop(valid_dl, model, device)
                final_loss, final_f1 = valid_loss, valid_f1
                log["valid/f1"] = valid_f1
                log["valid/loss"] = valid_loss
        log_str = " | ".join(f"{k}={v:.4f}" for k,v in log.items())
        if verbose:
            print(f"{step=}/{num_steps} | " + log_str)
        if log_fn is not None:
            log_fn(log, step=step)

    return final_loss, final_f1