from pathlib import Path
import argparse, time, os
from functools import partial

import numpy as np, torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
import wandb

from gesture_recognition.dataset import N_CLASSES, GestureDataset
from gesture_recognition.model import Model
from gesture_recognition.training_utils import schedule_lr, MixUp, save_checkpoint
from gesture_recognition.training import train

parser = argparse.ArgumentParser()
parser.add_argument("--sensor-dir", type=str, default="data/processed")
# run state
parser.add_argument("--run", type=str, default=None, help="Name of the wandb run")
parser.add_argument("--wandb-group", type=str, default="test")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-cv", action="store_true")
parser.add_argument("--lr-range-test", action="store_true", help="Runs LR range test by setting `warmup_frac=1` and `warmup_strat='exp'`")
parser.add_argument("--save-ckpt", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
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
parser.add_argument("--warmup-frac", type=float, default=0.1)
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
splits = dset.get_splits(seed=seed)
if not args.run_cv or args.lr_range_test:
    splits = [next(iter(splits))]

# model and dataloader params
num_layers = args.num_layers
d_model = args.d_model
num_workers = os.cpu_count() or 4
bs = args.bs

# optimization
lr_max = args.lr_max
warmup_strat = "linear"
betas = (args.momentum, 0.99)
mixup = MixUp(N_CLASSES, args.mixup_alpha)

if args.lr_range_test:
    args.warmup_frac = 1.0
    warmup_strat = "exp"
    args.epochs = 5

# user config
config = vars(args).copy()
for k in ["run","wandb_group","sensor_dir"]:
    config.pop(k)

# ----------------------------------------------------------------------------------------------------
for i, (train_idxs, valid_idxs) in enumerate(splits):
    train_dl = DataLoader(
        Subset(dset, train_idxs), batch_size=bs, shuffle=True,
        pin_memory=(device=="cuda"), num_workers=num_workers
    )
    valid_dl = DataLoader(
        Subset(dset, valid_idxs), batch_size=bs, shuffle=False,
        pin_memory=(device=="cuda"), num_workers=num_workers
    )
    steps_per_epoch = len(train_dl)
    num_steps = steps_per_epoch*args.epochs

    model = Model(num_layers, d_model, N_CLASSES, p=args.p).to(device)
    optimizer = AdamW(model.parameters(), weight_decay=args.wd, betas=betas)
    lr_scheduler = partial(
        schedule_lr, lr_max=lr_max, tot_steps=num_steps, init_lr_frac=args.init_lr_frac,
        final_lr_frac=args.final_lr_frac, warmup_frac=args.warmup_frac, warmup_strat=warmup_strat
    )

    run_name = f"{args.run}-fold{i}"
    run = wandb.init(
        project="gesture_recognition", name=run_name, group=args.wandb_group, config=config
    )
    
    print(f"\nStarting run {run_name}")
    start = time.time()
    valid_loss, valid_f1 = train(
        train_dl, valid_dl, model=model, mixup=mixup, optimizer=optimizer, num_steps=num_steps,
        lr_scheduler=lr_scheduler, log_fn=run.log, device=device, verbose=args.verbose
    )
    run.finish()
    tot_time = (time.time()-start)/60
    print(f"Run complete!")
    print(f"{tot_time=:.3f} minutes | {valid_loss=:.3f} | {valid_f1=:.3f}")

    if args.save_ckpt:
        ckpt_path = Path(f"models/{run_name}.tar")
        save_checkpoint(model, optimizer, num_steps, ckpt_path)