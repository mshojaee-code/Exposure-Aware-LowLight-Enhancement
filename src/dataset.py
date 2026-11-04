import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class MultiCameraLOLDataset(Dataset):
    """Paired dataset loader for LOL-v1 and LOL-v2 Real with exposure offset conditioning."""

    def __init__(
            self,
            lol_v1_root: str,
            extra_pairs: list = None,
            split: str = "train",
            patch_size: int = 256,
            mode: str = "multi"
    ):
        super().__init__()
        self.split = split
        self.patch_size = patch_size
        self.mode = mode

        sub_folder = "our485" if split == "train" else "eval15"
        low_dir = os.path.join(lol_v1_root, sub_folder, "low")
        high_dir = os.path.join(lol_v1_root, sub_folder, "high")

        low_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        high_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(low_files) == len(high_files), "mismatch between low and high image counts in LOL-v1"

        self.v1_pairs = []
        means_high_v1 = []
        print(f"Loading {split} split ({len(low_files)} pairs) into memory...")
        for lf, hf in tqdm(zip(low_files, high_files), total=len(low_files)):
            l_img = Image.open(os.path.join(low_dir, lf)).convert("RGB")
            h_img = Image.open(os.path.join(high_dir, hf)).convert("RGB")
            self.v1_pairs.append((l_img, h_img, lf))
            means_high_v1.append(np.mean(np.asarray(h_img, dtype=np.float32)) / 255.0)

        self.ref_exposure = float(np.mean(means_high_v1))
        self.extra_pairs = extra_pairs if (split == "train" and extra_pairs is not None) else []

    def set_mode(self, new_mode: str):
        self.mode = new_mode

    def __len__(self) -> int:
        if self.split == "val":
            return len(self.v1_pairs)
        if self.mode == "finetune_lolv1" or len(self.extra_pairs) == 0:
            return len(self.v1_pairs) * 2
        return 1000

    def _sync_crop_augment(self, low_img: Image.Image, high_img: Image.Image):
        w, h = low_img.size
        if w < self.patch_size or h < self.patch_size:
            low_img = TF.resize(low_img, (max(h, self.patch_size), max(w, self.patch_size)))
            high_img = TF.resize(high_img, (max(h, self.patch_size), max(w, self.patch_size)))

        i, j, ch, cw = T.RandomCrop.get_params(low_img, output_size=(self.patch_size, self.patch_size))
        low_img = TF.crop(low_img, i, j, ch, cw)
        high_img = TF.crop(high_img, i, j, ch, cw)

        if random.random() > 0.5:
            low_img, high_img = TF.hflip(low_img), TF.hflip(high_img)
        if random.random() > 0.5:
            low_img, high_img = TF.vflip(low_img), TF.vflip(high_img)

        low_t = TF.to_tensor(low_img)
        high_t = TF.to_tensor(high_img)

        k = random.randint(0, 3)
        if k > 0:
            low_t = torch.rot90(low_t, k, dims=[1, 2])
            high_t = torch.rot90(high_t, k, dims=[1, 2])

        return low_t, high_t

    def __getitem__(self, idx: int):
        if self.split == "val":
            low_img, high_img, fname = self.v1_pairs[idx]
            low_t, high_t = TF.to_tensor(low_img), TF.to_tensor(high_img)
            exp_cond = torch.tensor([0.0], dtype=torch.float32)
            return {"low": low_t, "high": high_t, "exp_cond": exp_cond, "filename": fname}

        if self.mode == "finetune_lolv1" or len(self.extra_pairs) == 0:
            low_img, high_img, fname = self.v1_pairs[idx % len(self.v1_pairs)]
            low_t, high_t = self._sync_crop_augment(low_img, high_img)
            exp_cond = torch.tensor([0.0], dtype=torch.float32)
            return {"low": low_t, "high": high_t, "exp_cond": exp_cond, "filename": fname}

        if idx < len(self.v1_pairs):
            low_img, high_img, fname = self.v1_pairs[idx]
            low_t, high_t = self._sync_crop_augment(low_img, high_img)
            exp_cond = torch.tensor([0.0], dtype=torch.float32)
        else:
            ext_idx = random.randint(0, len(self.extra_pairs) - 1)
            l_path, h_path = self.extra_pairs[ext_idx]
            low_img = Image.open(l_path).convert("RGB")
            high_img = Image.open(h_path).convert("RGB")
            if low_img.size != high_img.size:
                high_img = TF.resize(high_img, (low_img.size[1], low_img.size[0]))
            low_t, high_t = self._sync_crop_augment(low_img, high_img)
            fname = os.path.basename(l_path)

            # relative brightness offset vs. LOL-v1 reference distribution
            target_mean = high_t.mean().item()
            exp_delta = float(np.clip(target_mean - self.ref_exposure, -0.35, 0.35))
            exp_cond = torch.tensor([exp_delta], dtype=torch.float32)

        return {"low": low_t, "high": high_t, "exp_cond": exp_cond, "filename": fname}


def get_dataloaders(lol_v1_root: str, extra_pairs: list = None, batch_size: int = 16, patch_size: int = 256):
    train_ds = MultiCameraLOLDataset(lol_v1_root, extra_pairs, split="train", patch_size=patch_size, mode="multi")
    val_ds = MultiCameraLOLDataset(lol_v1_root, None, split="val", patch_size=patch_size, mode="multi")

    # automatically adjust num_workers based on os to prevent windows multiprocessing crash
    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
