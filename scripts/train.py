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
parser.add_argument("--wandb-group", type=str, default="leaderboard")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-cv", action="store_true")
parser.add_argument("--lr-range-test", action="store_true", help="Runs LR range test by setting `warmup_frac=1` and `warmup_strat='exp'`")
parser.add_argument("--save-ckpt", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
# model
parser.add_argument("--num-layers", type=int, default=5)
parser.add_argument("--seq-len", type=int, default=96)
parser.add_argument("--d-model", type=int, default=64)
# training horizon
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--bs", type=int, default=128)
# optimization
parser.add_argument("--lr-max", type=float, default=0.02)
parser.add_argument("--init-lr-frac", type=float, default=1e-2)
parser.add_argument("--final-lr-frac", type=float, default=1e-3)
parser.add_argument("--warmup-frac", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.935)
# regularization
parser.add_argument("--wd", type=float, default=2e-3)
parser.add_argument("--p-dropout", type=float, default=0.09, help="probability for dropout")
parser.add_argument("--p-flip", type=float, default=0.5, help="probability for flipping the sequences")
parser.add_argument("--p-proximity-drop", type=float, default=0.45)
parser.add_argument("--mixup-alpha", type=float, default=-1)
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
        Subset(dset, train_idxs), batch_size=bs, shuffle=True, drop_last=True, 
        pin_memory=(device=="cuda"), num_workers=num_workers
    )
    if not args.lr_range_test:
        valid_dl = DataLoader(
            Subset(dset, valid_idxs), batch_size=bs, shuffle=False, drop_last=False,
            pin_memory=(device=="cuda"), num_workers=num_workers
        )
    else:
        valid_dl = None
    steps_per_epoch = len(train_dl)
    num_steps = steps_per_epoch*args.epochs

    model = Model(num_layers, d_model, N_CLASSES, p=args.p_dropout).to(device)
    optimizer = AdamW(model.parameters(), weight_decay=args.wd, betas=betas)
    lr_scheduler = partial(
        schedule_lr, lr_max=lr_max, tot_steps=num_steps, init_lr_frac=args.init_lr_frac,
        final_lr_frac=args.final_lr_frac, warmup_frac=args.warmup_frac, warmup_strat=warmup_strat
    )

    run_name = f"{args.run}-fold{i}" if args.run_cv else args.run
    run = wandb.init(
        project="gesture_recognition", name=run_name, group=args.wandb_group, config=config
    )
    
    print(f"\nStarting run {run.name}")
    start = time.time()
    valid_loss, valid_f1 = train(
        model, train_dl, num_steps=num_steps, optimizer=optimizer, lr_scheduler=lr_scheduler,
        mixup=mixup, valid_dl=valid_dl, log_fn=run.log, p_flip=args.p_flip, device=device,
        p_proximity_drop=args.p_proximity_drop, verbose=args.verbose,
    )
    run.finish()
    tot_time = (time.time()-start)/60
    print(f"Run complete!")
    print(f"{tot_time=:.3f} minutes | {valid_loss=:.3f} | {valid_f1=:.3f}")

    if args.save_ckpt:
        ckpt_path = Path(f"models/{run.name}.tar")
        save_checkpoint(model, optimizer, num_steps, ckpt_path)