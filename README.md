VITON_APP

VITON_APP is a custom, commercial-ready reimplementation of the VITON pipeline for virtual try-on, based on Zalando-style datasets.

This repository currently implements Phase 1: WarpNet training, which learns to warp a clothing mask to match the target person pose.

The project is designed to be:
	•	modular
	•	reproducible
	•	scalable from local CPU to GPU VMs
	•	suitable for commercial use

⸻

Project Structure

VITON_APP/
├── data/
│   └── warp_dataset.py        # Dataset loader for WarpNet
├── models/
│   └── warpnet.py             # WarpNet model (flow prediction)
├── utils/
│   └── losses.py              # Loss functions (TV loss)
├── train/
│   └── train_warp.py          # Training script for WarpNet
├── datasets/                  # (NOT committed) training data
├── checkpoints/               # (NOT committed) model checkpoints
├── debug/                     # (NOT committed) debug visualizations
├── requirements.txt
└── README.md


⸻

Environment Setup

Python version
	•	Python 3.10+ recommended

Create a virtual environment (recommended)

python3 -m venv .venv
source .venv/bin/activate

Upgrade pip:

pip install --upgrade pip

Install dependencies

CPU-only (works everywhere):

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow tqdm opencv-python

On a CUDA machine, install the appropriate CUDA-enabled PyTorch build instead.

⸻

Dataset Requirements

The dataset must follow the Zalando / VITON processed structure.

Expected layout:

datasets/chiqado_v1/processed/train/
├── pairs.txt
├── agnostic-v3.2/
├── openpose_img/
├── cloth-mask/
└── gt_cloth_warped_mask/

pairs.txt

Each line contains:

<person_image> <cloth_image>

Example:

10224_00.jpg 03195_00.jpg
12308_00.jpg 07502_00.jpg

Required files per pair

Type	Path
Agnostic person	agnostic-v3.2/{person_id}.jpg
Pose image	openpose_img/{person_id}_rendered.png
Cloth mask	cloth-mask/{cloth_id}.jpg
GT warped mask	gt_cloth_warped_mask/{person_id}.jpg

Masks may be .jpg or .png.
They are binarized automatically in the loader.

⸻

Phase 1 — WarpNet Training

Goal

Train a WarpNet that predicts a dense flow field to warp a clothing mask so it aligns with the target person pose.

Inputs
	•	Agnostic person image
	•	Rendered OpenPose image
	•	Cloth mask

Target
	•	Ground-truth warped cloth mask

Outputs
	•	Dense flow field
	•	Warped cloth mask

⸻

Sanity Check Dataset Loading

Run from the project root:

PYTHONPATH=. python - <<'PY'
from data.warp_dataset import WarpDataset
ds = WarpDataset("datasets/chiqado_v1/processed", split="train", height=256, width=192)
x = ds[0]
print(x["person_name"], x["cloth_name"])
for k in ["agnostic","pose","cloth_mask","gt_warp_mask"]:
    print(k, x[k].shape, x[k].min().item(), x[k].max().item())
PY

Expected:
	•	tensors with correct shapes
	•	masks in [0, 1]

⸻

Train WarpNet

Important: always run with PYTHONPATH=. from the project root.

Quick smoke test (small resolution)

PYTHONPATH=. python train/train_warp.py \
  --data_root datasets/chiqado_v1/processed \
  --split train \
  --out checkpoints/warpnet \
  --debug_dir debug/warpnet \
  --batch_size 2 \
  --epochs 1 \
  --height 256 \
  --width 192 \
  --workers 0 \
  --debug_every 50

Recommended training (Phase 1)

PYTHONPATH=. python train/train_warp.py \
  --data_root datasets/chiqado_v1/processed \
  --split train \
  --out checkpoints/warpnet \
  --debug_dir debug/warpnet \
  --batch_size 2 \
  --epochs 5 \
  --height 512 \
  --width 384 \
  --workers 2 \
  --debug_every 200


⸻

Outputs

Checkpoints

Saved to:

checkpoints/warpnet/
├── warp_epoch1.pth
├── warp_epoch2.pth
└── warp_final.pth

Debug visualizations

Saved to:

debug/warpnet/
└── e{epoch}_s{step}.png

Each debug image grid shows:
	1.	Input cloth mask
	2.	Warped cloth mask (model output)
	3.	Ground-truth warped mask

These are used to visually verify learning progress.

⸻

Resume Training

PYTHONPATH=. python train/train_warp.py \
  --data_root datasets/chiqado_v1/processed \
  --split train \
  --out checkpoints/warpnet \
  --resume checkpoints/warpnet/warp_epoch5.pth \
  --epochs 10 \
  --height 512 \
  --width 384


⸻

Git & Reproducibility

.gitignore

Do not commit data or checkpoints:

.venv/
datasets/
checkpoints/
debug/
__pycache__/
*.pyc

Freeze dependencies

After environment is stable:

pip freeze > requirements.txt


⸻

Current Status

✅ Phase 1 (WarpNet) implemented and training
⏭ Phase 2: RenderNet (final image synthesis)
⏭ Phase 3: Inference pipeline
⏭ Phase 4: Deployment & app integration

⸻

License Note

This repository is intended to support commercial usage.
All third-party components (e.g. pose extraction, parsing models) must comply with their respective licenses.