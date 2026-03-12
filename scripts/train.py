from pathlib import Path
import argparse, time, os
from tqdm import tqdm
from contextlib import nullcontext
from functools import partial

import numpy as np, torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
import wandb

from gesture_recognition.dataset import N_CLASSES, GestureDataset
from gesture_recognition.model import Model
from gesture_recognition.training_utils import compute_loss, schedule_lr, hierarchical_f1, MixUp, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default=None, help="Name of the wandb run")
parser.add_argument("--wandb-group", type=str, default="test")
parser.add_argument("--sensor-dir", type=str, default="data/processed")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr-range-test", action="store_true", help="Runs LR range test by setting `warmup_frac=1` and `warmup_strat='exp'`")
# model
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--seq-len", type=int, default=96)
parser.add_argument("--d-model", type=int, default=64)
# training horizon
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--bs", type=int, default=128)
# optimization
parser.add_argument("--lr-max", type=float, default=3e-2)
parser.add_argument("--init-lr-frac", type=float, default=1e-2)
parser.add_argument("--final-lr-frac", type=float, default=1e-3)
parser.add_argument("--warmup-frac", type=float, default=1e-1)
parser.add_argument("--momentum", type=float, default=0.95)
# regularization
parser.add_argument("--wd", type=float, default=1e-1)
parser.add_argument("--p", type=float, default=0.0)
parser.add_argument("--mixup-alpha", type=float, default=0.0)
args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------
# device init
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# seed everything
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if device=="cuda":
    torch.cuda.manual_seed_all(seed)

# dataset init
seq_len = args.seq_len
dset = GestureDataset(args.sensor_dir, max_seq_len=seq_len)
mixup = MixUp(N_CLASSES, args.mixup_alpha)
train_collate = partial(dset.collate, device=device, mixup=mixup)
valid_collate = partial(dset.collate, device=device, mixup=None)
train_idxs, valid_idxs = dset.get_split(seed=seed)

#dataloader init
num_workers = os.cpu_count() or 4
bs = args.bs
train_dl = DataLoader(
    Subset(dset, train_idxs), batch_size=bs, collate_fn=train_collate,
    shuffle=True, pin_memory=(device=="cuda"), num_workers=num_workers
)
valid_dl = DataLoader(
    Subset(dset, valid_idxs), batch_size=bs, collate_fn=valid_collate,
    shuffle=False, pin_memory=(device=="cuda"), num_workers=num_workers
)

# training horizon
epochs = args.epochs
num_steps = len(train_dl)*epochs

# model init
num_layers = args.num_layers
d_model = args.d_model
model = Model(num_layers, d_model, N_CLASSES, p=args.p).to(device)

# optimization
lr_max = args.lr_max
warmup_strat = "linear"
betas = (args.momentum, 0.99)
optimizer = AdamW(model.parameters(), weight_decay=args.wd, betas=betas)

if args.lr_range_test:
    args.warmup_frac = 1.0
    warmup_strat = "exp"

# logging init
config = vars(args).copy()
for k in ["run","wandb_group","sensor_dir"]:
    config.pop(k)
run = wandb.init(
    project="gesture_recognition", name=args.run, group=args.wandb_group, config=config
)
ckpt_path = Path(f"models/{run.name}")

# ----------------------------------------------------------------------------------------------------
@torch.inference_mode()
def valid_loop():
    tot_loss, tot_samples = 0, 0
    all_preds, all_targs = [], []
    for *xs, y in tqdm(valid_dl, desc="Validating"):
        logits = model(*xs)
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
    return tot_loss, f1

start = time.time()
train_dl_iter = iter(train_dl)

for step in range(num_steps):
    try:
        *xs, y = next(train_dl_iter)
    except StopIteration:
        # TODO: compute and log metric on training set
        log = {}
        if not args.lr_range_test:
            valid_loss, valid_f1 = valid_loop()
            log["valid/f1"] = valid_f1
            log["valid/loss"] = valid_loss
        wandb.log(log, step=step-1)
        print(" | ".join(f"{k}={v:.4f}" for k,v in log.items()))

        train_dl_iter = iter(train_dl)
        *xs, y = next(train_dl_iter)

    lr = schedule_lr(
        step=step, lr_max=lr_max, tot_steps=num_steps, init_lr_frac=args.init_lr_frac,
        warmup_frac=args.warmup_frac, final_lr_frac=args.final_lr_frac, warmup_strat=warmup_strat
    )
    for g in optimizer.param_groups:
        g["lr"] = lr
    
    logits = model(*xs)
    loss = compute_loss(logits, y)
    if not torch.isfinite(loss).item():
        break
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # log
    log = {"train/loss":loss.detach().item(), "lr": lr}
    log_str = " | ".join(f"{k}={v:.4f}" for k,v in log.items())
    print(f"{step=}/{num_steps} | " + log_str)
    wandb.log(log, step=step)

save_checkpoint(model, optimizer, num_steps, ckpt_path)
tot_time = time.time()-start
print(f"\nRun complete! Total time taken: {tot_time/60:,} minutes")
run.finish()