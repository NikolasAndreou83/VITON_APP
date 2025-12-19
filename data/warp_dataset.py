from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_l(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def resize(img: Image.Image, size_hw: Tuple[int, int], is_mask: bool = False) -> Image.Image:
    h, w = size_hw
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((w, h), resample=resample)


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    return torch.from_numpy(arr) / 255.0


def binarize_mask(t: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (t > thr).float()


class WarpDataset(Dataset):
    """
    Loads items for WarpNet training from a Zalando/VITON-like processed folder.

    Expected under split_dir:
      - pairs.txt
      - agnostic-v3.2/{person}.jpg|png
      - openpose_img/{person_stem}_rendered.png
      - cloth-mask/{cloth}.jpg|png
      - gt_cloth_warped_mask/{person}.jpg|png
    """

    def __init__(self, data_root: str | Path, split: str = "train", height: int = 512, width: int = 384):
        self.data_root = Path(data_root)
        # Allow passing either ".../processed" or ".../processed/train"
        split_dir = self.data_root / split
        if (self.data_root / "pairs.txt").exists():
            split_dir = self.data_root
        self.split_dir = split_dir

        self.h = int(height)
        self.w = int(width)

        self.pairs_path = self.split_dir / "pairs.txt"
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"pairs.txt not found at: {self.pairs_path}")

        self.agn_dir = self.split_dir / "agnostic-v3.2"
        self.pose_img_dir = self.split_dir / "openpose_img"
        self.cloth_mask_dir = self.split_dir / "cloth-mask"
        self.gt_warp_dir = self.split_dir / "gt_cloth_warped_mask"

        for d in [self.agn_dir, self.pose_img_dir, self.cloth_mask_dir, self.gt_warp_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Missing folder: {d}")

        self.pairs: List[Tuple[str, str]] = []
        with open(self.pairs_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p, c = line.split()
                self.pairs.append((p, c))

    def __len__(self) -> int:
        return len(self.pairs)

    def _find_with_ext(self, folder: Path, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
        # also try exact stem if it already has extension
        p = folder / stem
        if p.exists():
            return p
        raise FileNotFoundError(f"File not found for stem={stem} in {folder}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        person_name, cloth_name = self.pairs[idx]
        person_stem = Path(person_name).stem
        cloth_stem = Path(cloth_name).stem

        # agnostic
        agn_path = self._find_with_ext(self.agn_dir, person_stem)

        # pose rendered image
        pose_path = self.pose_img_dir / f"{person_stem}_rendered.png"
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose rendered not found: {pose_path}")

        # cloth mask
        cm_path = self._find_with_ext(self.cloth_mask_dir, cloth_stem)

        # gt warped cloth mask (indexed by person id)
        gt_path = self._find_with_ext(self.gt_warp_dir, person_stem)

        # load + resize
        agn = resize(pil_rgb(agn_path), (self.h, self.w), is_mask=False)
        pose = resize(pil_rgb(pose_path), (self.h, self.w), is_mask=False)

        cm = resize(pil_l(cm_path), (self.h, self.w), is_mask=True)
        gt = resize(pil_l(gt_path), (self.h, self.w), is_mask=True)

        agn_t = to_tensor(agn)     # 3,H,W
        pose_t = to_tensor(pose)   # 3,H,W
        cm_t = to_tensor(cm)       # 1,H,W
        gt_t = to_tensor(gt)       # 1,H,W

        # JPEG masks often have artifacts; binarize both
        cm_t = binarize_mask(cm_t, 0.5)
        gt_t = binarize_mask(gt_t, 0.5)

        return {
            "person_name": person_name,
            "cloth_name": cloth_name,
            "agnostic": agn_t,
            "pose": pose_t,
            "cloth_mask": cm_t,
            "gt_warp_mask": gt_t,
        }
