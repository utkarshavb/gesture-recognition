from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedGroupKFold
from scipy.spatial.transform import Rotation as R
from einops import rearrange

import numpy.typing as npt
from torch import Tensor
from jaxtyping import Float, Int

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
    fill_arr = np.array([1,0,0,0], dtype=dtype)
    quat = np.where(np.isnan(quat), fill_arr, quat)
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

def pad_or_trunc(*xs: Float[npt.NDArray, "seq_len c"], L=128) -> tuple[Float[npt.NDArray, "L c"], ...]:
    seq_len, dtype = xs[0].shape[0], xs[0].dtype
    if seq_len > L:
        xs = tuple(x[-L:,:] for x in xs)
    else:
        pad_len = L-seq_len
        xs = tuple(
            np.concatenate(
                [np.full((pad_len, x.shape[1]), 0, dtype=dtype), x], axis=0
            ) for x in xs
        )
    return xs

def handedness_flip(acc, lin_acc, rel_rot, thm, tof, hand: int) -> tuple[npt.NDArray, ...]:
    if hand==0:
        acc[:, 0] *= -1   # flips x component
        lin_acc[:, 0] *= -1
        rel_rot[:, 1:] *= -1   # flips y & z components
        perm = [0, 1, 4, 3, 2]   # swap the 3rd and 5th sensors
        thm, tof = thm[:, perm], tof[:, perm]
        tof = tof[:,:,:,::-1].copy()   # horizontally flip each tof sensor
    return acc, lin_acc, rel_rot, thm, tof

def extract_features(
    imu: npt.NDArray, thm: npt.NDArray, tof: npt.NDArray, hand: int, L: int
) -> tuple[Tensor, ...]:
    """missing `thm` and `tof` values are intentionally not imputed"""
    acc, quat = imu[:, :3], imu[:, 3:]
    quat = impute_quat(quat, dtype=imu.dtype)
    lin_acc = remove_gravity(acc, quat)
    rel_rot = get_rel_rot(quat)
    imus = (acc, lin_acc, rel_rot)

    *imus, thm, tof = pad_or_trunc(*imus, thm, tof, L=L)
    tof = rearrange(tof, "L (n h w) -> L n h w", w=8, h=8)
    *imus, thm, tof = handedness_flip(*imus, thm=thm, tof=tof, hand=hand)

    imu_tensors = tuple(rearrange(torch.from_numpy(x), "L c -> c L") for x in imus)
    thm_tensor = rearrange(torch.from_numpy(thm), "L c -> c L")
    tof_tensor = torch.from_numpy(tof)
    tof_tensor = torch.where(tof_tensor==-1, 255.0, tof_tensor)/255

    return *imu_tensors, thm_tensor, tof_tensor

class GestureDataset(Dataset):
    def __init__(self, sensor_dir: str|Path, max_seq_len: int=128):
        if isinstance(sensor_dir, str):
            sensor_dir = Path(sensor_dir)
        self.L = max_seq_len

        self.imus: npt.NDArray = np.load(sensor_dir/"imu.npy", mmap_mode="r")
        self.thms: npt.NDArray = np.load(sensor_dir/"thm.npy", mmap_mode="r")
        self.tofs: npt.NDArray = np.load(sensor_dir/"tof.npy", mmap_mode="r")

        meta = np.load(sensor_dir/"metadata.npz")
        self.seq_starts = meta["seq_starts"]
        self.seq_lens = meta["seq_lens"]
        self.gestures = meta["gestures"]
        self.subjects = meta["subjects"]
        self.handedness = meta["handedness"]

    def __len__(self) -> int:
        return self.seq_starts.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        start = self.seq_starts[idx]
        seq_len = self.seq_lens[idx]
        imu = self.imus[start: start+seq_len, :].copy()
        thm = self.thms[start: start+seq_len, :].copy()
        tof = self.tofs[start: start+seq_len, :].copy().astype(imu.dtype)
        gesture = LBL2ID[self.gestures[idx].item()]
        hand = self.handedness[idx].item()

        *imus, thm, tof = extract_features(imu, thm, tof, hand=hand, L=self.L)
        gesture = torch.tensor(gesture, dtype=torch.long)
        return *imus, thm, tof, gesture

    def get_splits(self, num_folds: int=5, seed: int=42):
        dummy_x = self.seq_lens.reshape(-1,1)
        splits = StratifiedGroupKFold(
            num_folds, shuffle=True, random_state=seed
        ).split(dummy_x, self.gestures, self.subjects)
        return splits