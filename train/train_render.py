from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.render_dataset import RenderDataset
from models.warpnet import WarpNet, warp_with_flow
from models.rendernet import RenderNet


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


def load_warp_checkpoint(warp_ckpt_path: str, device: torch.device) -> WarpNet:
    ckpt = torch.load(warp_ckpt_path, map_location="cpu")
    warp = WarpNet(in_ch=7, base=32, flow_max=0.5)
    warp.load_state_dict(ckpt["model"])
    warp.to(device)
    warp.eval()
    for p in warp.parameters():
        p.requires_grad = False
    return warp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="e.g. datasets/chiqado_v1/processed")
    p.add_argument("--split", default="train")
    p.add_argument("--warp_ckpt", required=True, help="path to WarpNet checkpoint (.pth)")
    p.add_argument("--out", default="checkpoints/rendernet")
    p.add_argument("--debug_dir", default="debug/rendernet")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--debug_every", type=int, default=200)
    p.add_argument("--resume", type=str, default="")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = (device.type == "cuda")

    ds = RenderDataset(args.data_root, split=args.split, height=args.height, width=args.width)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type=="cuda"))

    warp = load_warp_checkpoint(args.warp_ckpt, device)

    render = RenderNet(in_ch=10, base=48).to(device)
    opt = torch.optim.Adam(render.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        render.load_state_dict(ckpt["model"])
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
        render.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            agn = batch["agnostic"].to(device)     # B,3,H,W
            pose = batch["pose"].to(device)        # B,3,H,W
            cloth = batch["cloth"].to(device)      # B,3,H,W
            cm = batch["cloth_mask"].to(device)    # B,1,H,W
            gt = batch["gt"].to(device)            # B,3,H,W

            # Warp cloth + mask with frozen WarpNet (no gradients)
            with torch.no_grad():
                warp_in = torch.cat([agn, pose, cm], dim=1)  # B,7,H,W
                flow = warp(warp_in)
                warped_mask = warp_with_flow(cm, flow).clamp(0, 1)
                warped_cloth = warp_with_flow(cloth, flow).clamp(0, 1)

            # RenderNet input
            x = torch.cat([agn, pose, warped_cloth, warped_mask], dim=1)  # B,10,H,W

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = render(x)  # B,3,H,W
                # simple stable loss: L1 overall + stronger on clothing region
                l1_all = F.l1_loss(pred, gt)
                # emphasize clothing region (use warped mask)
                l1_cloth = (torch.abs(pred - gt) * warped_mask).mean()
                loss = l1_all + 2.0 * l1_cloth

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), l1=float(l1_all.item()), l1c=float(l1_cloth.item()))

            if global_step % args.debug_every == 0:
                # Save: [agnostic, warped_cloth, pred, gt]
                grid = torch.cat([agn[:2], warped_cloth[:2], pred[:2], gt[:2]], dim=0)
                save_image(grid, dbg_dir / f"e{epoch}_s{global_step}.png", nrow=2)

        if epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"render_epoch{epoch}.pth", render, opt, epoch, global_step)

    save_checkpoint(out_dir / "render_final.pth", render, opt, args.epochs, global_step)
    print(f"Done. Saved final to: {out_dir/'render_final.pth'}")


if __name__ == "__main__":
    main()
