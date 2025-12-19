from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.warp_dataset import WarpDataset
from models.warpnet import WarpNet, warp_with_flow
from utils.losses import tv_loss


def save_checkpoint(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
        },
        str(path),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="e.g. datasets/chiqado_v1/processed")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="checkpoints/warpnet")
    p.add_argument("--debug_dir", default="debug/warpnet")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--flow_max", type=float, default=0.5)
    p.add_argument("--tv_weight", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=1)          # epochs
    p.add_argument("--debug_every", type=int, default=200)       # steps
    p.add_argument("--resume", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    ds = WarpDataset(args.data_root, split=args.split, height=args.height, width=args.width)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type=="cuda"))

    model = WarpNet(in_ch=7, base=32, flow_max=args.flow_max).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("step", 0))
        print(f"[resume] epoch={start_epoch} step={global_step}")

    out_dir = Path(args.out)
    dbg_dir = Path(args.debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Dataset size: {len(ds)} | Steps/epoch: {len(dl)}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            agn = batch["agnostic"].to(device)       # B,3,H,W
            pose = batch["pose"].to(device)          # B,3,H,W
            cm = batch["cloth_mask"].to(device)      # B,1,H,W
            gt = batch["gt_warp_mask"].to(device)    # B,1,H,W

            x = torch.cat([agn, pose, cm], dim=1)    # B,7,H,W

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                flow = model(x)                      # B,2,H,W
                warped = warp_with_flow(cm, flow)    # B,1,H,W

                # L1 mask loss (stable with binarized targets)
                loss_mask = F.l1_loss(warped, gt)

                loss_tv = tv_loss(flow) * args.tv_weight
                loss = loss_mask + loss_tv

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), mask=float(loss_mask.item()), tv=float(loss_tv.item()))

            if global_step % args.debug_every == 0:
                # Save a debug grid: [cm, warped, gt]
                # clamp to [0,1]
                grid = torch.cat([cm[:4], warped[:4].clamp(0,1), gt[:4]], dim=0)
                save_image(grid, dbg_dir / f"e{epoch}_s{global_step}.png", nrow=4)

        if epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"warp_epoch{epoch}.pth", model, opt, epoch, global_step)

    save_checkpoint(out_dir / "warp_final.pth", model, opt, args.epochs, global_step)
    print(f"Done. Saved final to: {out_dir/'warp_final.pth'}")


if __name__ == "__main__":
    main()
