from pathlib import Path
import argparse, time, os
from tqdm import tqdm
from contextlib import nullcontext

import numpy as np, torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import wandb

from gesture_recognition.dataset import N_CLASSES, GestureDataset
from gesture_recognition.model import Model
from gesture_recognition.training_utils import loss_fn, hierarchical_f1, MixUp, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default=None, help="Name of the wandb run")
parser.add_argument("--wandb-group", type=str, default="test")
parser.add_argument("--sensor-dir", type=str, default="data/processed")
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--seq-len", type=int, default=96)
parser.add_argument("--d-model", type=int, default=64)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--lr-max", type=float, default=3e-2)
parser.add_argument("--wd", type=float, default=1e-1)
parser.add_argument("--momentum", type=float, default=0.95)
parser.add_argument("--p", type=float, default=0.0)
parser.add_argument("--mixup-alpha", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------
print("\nConfiguration:")

# seed everything
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"{seed=}")

# device init
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device=}")

# dataset init
seq_len = args.seq_len
print(f"{seq_len=}")
dset = GestureDataset(args.sensor_dir, max_seq_len=seq_len)
train_idxs, valid_idxs = dset.get_split(seed=seed)

#dataloader init
num_workers = os.cpu_count() or 4
bs = args.bs
print(f"{bs=}, {num_workers=}")
train_dl = DataLoader(
    Subset(dset, train_idxs), batch_size=bs, shuffle=True,
    pin_memory=(device=="cuda"), num_workers=num_workers
)
valid_dl = DataLoader(
    Subset(dset, valid_idxs), batch_size=bs, shuffle=False,
    pin_memory=(device=="cuda"), num_workers=num_workers
)

# model init
num_layers = args.num_layers
d_model = args.d_model
print(f"{num_layers=}, {d_model=}, p={args.p:.4f}")
model = Model(num_layers, d_model, N_CLASSES, p=args.p).to(device)

# optimization
epochs = args.epochs
print(f"{epochs=}")
lr_max, wd, momentum = args.lr_max, args.wd, args.momentum
print(f"{lr_max=:.2f}, {wd=:.3f}, {momentum=:.2f}")
optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=wd)
scheduler = OneCycleLR(
    optimizer, lr_max, epochs=epochs, steps_per_epoch=len(train_dl),
    anneal_strategy="cos", div_factor=100, max_momentum=momentum
)

# extra
mixup = MixUp(N_CLASSES, args.mixup_alpha)
ckpt_dir = Path(f"models/{args.run}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# wandb init
config = {
    "num_layers":num_layers, "d_model":d_model, "seq_len":seq_len,
    "epochs":epochs, "bs":bs, "wd":wd, "lr_max":lr_max, "p":args.p,
    "momentum":momentum, "mixup_alpha":args.mixup_alpha
}
run = wandb.init(
    project="gesture_recognition", name=args.run, group=args.wandb_group, config=config
)

# ----------------------------------------------------------------------------------------------------
print("\nTraining loop:")

def run_loop(train: bool=True):
    if train:
        dl = train_dl
        model.train()
        desc = "Training"
        ctx = nullcontext
    else:
        dl = valid_dl
        model.eval()
        desc = "Validating"
        ctx = torch.inference_mode
    tot_loss = 0
    all_preds, all_targs = [], []

    with ctx():
        for accs, lin_accs, rel_rots, gestures in tqdm(dl, desc=desc, leave=False):
            all_targs.append(gestures.detach().cpu())   # store hard targets for metric computation

            accs = accs.to(device, non_blocking=True)
            lin_accs = lin_accs.to(device, non_blocking=True)
            rel_rots = rel_rots.to(device, non_blocking=True)
            gestures = gestures.to(device, non_blocking=True)
            if train:
                accs, lin_accs, rel_rots, gestures = mixup(accs, lin_accs, rel_rots, y=gestures)

            logits = model(accs, lin_accs, rel_rots)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu())

            loss = loss_fn(logits, gestures)
            tot_loss += loss.detach().item()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step(); scheduler.step()

    all_preds = torch.cat(all_preds, dim=0)
    all_targs = torch.cat(all_targs, dim=0)
    f1 = hierarchical_f1(all_preds, all_targs)
    return tot_loss/len(dl), f1

start = time.time()

for epoch in range(1, epochs+1):
    t0 = time.time()
    lr = optimizer.param_groups[0]["lr"]
    train_loss, train_f1 = run_loop(train=True)
    train_dt = time.time()-t0
    valid_loss, valid_f1 = run_loop(train=False)
    valid_dt = time.time()-t0-train_dt
    
    log = {
        "train/loss":train_loss, "valid/loss":valid_loss, "train/f1":train_f1,
        "valid/f1":valid_f1, "train/dt": train_dt, "valid/dt": valid_dt,
    }
    log_str = " | ".join([f"{k}={v:.4f}" for k,v in log.items()])
    print(f"{epoch=}/{epochs} | " + log_str)
    wandb.log(log)

save_checkpoint(model, optimizer, epochs, ckpt_dir/f"step: {epochs}")
tot_time = time.time()-start
print(f"\nRun complete! Total time taken: {tot_time/60:,} minutes")
run.finish()