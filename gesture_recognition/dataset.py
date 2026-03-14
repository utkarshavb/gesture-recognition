from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import StratifiedGroupKFold
from einops import rearrange
import numpy.typing as npt
from torch import Tensor
from gesture_recognition.training_utils import MixUp

ACC_COLS = ['acc_x','acc_y','acc_z']
ROT_COLS = ['rot_w','rot_x','rot_y','rot_z']

LBL2ID = {
    # Target gestures
    "Above ear - pull hair": 0,
    "Cheek - pinch skin": 1,
    "Eyebrow - pull hair": 2,
    "Eyelash - pull hair": 3,
    "Forehead - pull hairline": 4,
    "Forehead - scratch": 5,
    "Neck - pinch skin": 6,
    "Neck - scratch": 7,
    # Non-target gestures
    "Drink from bottle/cup": 8,
    "Feel around in tray and pull out an object": 9,
    "Glasses on/off": 10,
    "Pinch knee/leg skin": 11,
    "Pull air toward your face": 12,
    "Scratch knee/leg skin": 13,
    "Text on phone": 14,
    "Wave hello": 15,
    "Write name in air": 16,
    "Write name on leg": 17,
}

N_CLASSES = len(LBL2ID)

def impute_quat(quat: npt.NDArray, dtype):
    quat = np.where(np.isnan(quat), np.array([1,0,0,0], dtype=dtype), quat)
    return quat

def remove_gravity(acc: npt.NDArray, quat: npt.NDArray) -> npt.NDArray:
    """Removes effect of gravity from acceleration"""
    rot = R.from_quat(quat, scalar_first=True)
    gravity_world = np.array([0, 0, 9.81], dtype=acc.dtype)
    gravity_sensor_frame = rot.apply(gravity_world, inverse=True)
    lin_acc = acc - gravity_sensor_frame
    return lin_acc.astype(quat.dtype)

def get_rel_rot(quat: npt.NDArray) -> npt.NDArray:
    rot = R.from_quat(quat, scalar_first=True)
    rel = (rot[:-1].inv() * rot[1:]).as_rotvec()
    pad = np.zeros((1,3), dtype=quat.dtype)
    rel = np.concatenate([pad, rel], axis=0)
    return rel.astype(quat.dtype)

def handedness_flip(acc, lin_acc, rel_rot, hand: int):
    if hand==0:
        acc[:, 0] *= -1   # flips x component
        lin_acc[:, 0] *= -1
        rel_rot[:, 1:] *= -1   # flips y & z components
    return acc, lin_acc, rel_rot

class GestureDataset(Dataset):
    def __init__(self, sensor_dir: str|Path, max_seq_len: int=128):
        if isinstance(sensor_dir, str):
            sensor_dir = Path(sensor_dir)
        self.imus: npt.NDArray = np.load(sensor_dir/"imu.npy", mmap_mode="r")
        self.L = max_seq_len
        indexes = np.load(sensor_dir/"index.npz")
        self.seq_starts = indexes["seq_starts"]
        self.seq_lens = indexes["seq_lens"]
        self.gestures = indexes["gestures"]
        self.subjects = indexes["subjects"]
        self.handedness = indexes["handedness"]

    def __len__(self) -> int:
        return self.seq_starts.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        start = self.seq_starts[idx]
        seq_len = self.seq_lens[idx]
        end = start+seq_len
        if seq_len > self.L:
            start = end-self.L
            seq_len = self.L
        imu = self.imus[start:end, :].copy()

        pad_len = self.L-seq_len
        pre_pad = np.full((pad_len, 7), [0,0,0,1,0,0,0], dtype=imu.dtype)
        imu = np.concatenate([pre_pad, imu], axis=0)

        acc, quat = imu[:, :3], imu[:, 3:]
        quat = impute_quat(quat, dtype=imu.dtype)
        lin_acc = remove_gravity(acc, quat)
        rel_rot = get_rel_rot(quat)
        hand = self.handedness[idx].item()
        acc, lin_acc, rel_rot = handedness_flip(acc, lin_acc, rel_rot, hand)

        gesture = LBL2ID[self.gestures[idx].item()]
        gesture = torch.tensor(gesture, dtype=torch.long)
        acc, lin_acc, rel_rot = [
            rearrange(torch.from_numpy(x), "L c -> c L") for x in (acc, lin_acc, rel_rot)
        ]
        return acc, lin_acc, rel_rot, hand, gesture
    
    def collate(self, batch: list[tuple[Tensor, ...]], device=None, mixup: MixUp|None=None):
        """Collates and moves the batch to device; also does mixup if `mixup_alpha>0`"""
        *xs, y = zip(*batch)
        xs = [
            torch.stack(x).to(device, non_blocking=True) for x in xs
        ]
        y = torch.stack(y).to(device, non_blocking=True)
        if mixup:
            *xs, y = mixup(*xs, y=y)
        return *xs, y

    def get_split(self, num_folds: int=5, seed: int=42):
        """Returns a train/valid split"""
        dummy_x = self.seq_lens.reshape(-1,1)
        splits = StratifiedGroupKFold(
            num_folds, shuffle=True, random_state=seed
        ).split(dummy_x, self.gestures, self.subjects)
        return next(iter(splits))