import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from models import BiMambaFreqConfig, BiMambaFreqModel, bimamba_freq_multitask_loss
from preprocess.finmultitime import FinMultiTimeMultiTaskDataset
from utils import EarlyStopping, create_experiment_dir, multitask_finance_metrics, save_json, seed_everything


def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    cls_probs = []
    cls_targets = []
    vol_preds = []
    vol_targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_cls = batch["y_cls"].to(device)
            y_vol = batch["y_vol"].to(device)
            outputs = model(x)
            losses = bimamba_freq_multitask_loss(outputs, y_cls, y_vol, alpha=1.0, beta=1.0, gamma=0.01)
            batch_size = x.size(0)
            total_loss += losses["loss"].item() * batch_size
            total_items += batch_size
            cls_probs.append(outputs["classification"].detach().cpu())
            cls_targets.append(y_cls.detach().cpu())
            vol_preds.append(outputs["volatility"].detach().cpu())
            vol_targets.append(y_vol.detach().cpu())
    metrics = multitask_finance_metrics(
        torch.cat(cls_probs, dim=0),
        torch.cat(cls_targets, dim=0),
        torch.cat(vol_preds, dim=0),
        torch.cat(vol_targets, dim=0),
    )
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def train_financial_multitask(args) -> Path:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_ds = FinMultiTimeMultiTaskDataset(
        args.processed_root,
        split="train",
        with_time_features=False,
        use_multiscale_ohlcv=True,
        week_length=5,
        month_length=21,
        vol_window=30,
    )
    val_ds = FinMultiTimeMultiTaskDataset(
        args.processed_root,
        split="val",
        with_time_features=False,
        use_multiscale_ohlcv=True,
        week_length=5,
        month_length=21,
        vol_window=30,
    )
    test_ds = FinMultiTimeMultiTaskDataset(
        args.processed_root,
        split="test",
        with_time_features=False,
        use_multiscale_ohlcv=True,
        week_length=5,
        month_length=21,
        vol_window=30,
    )

    if args.expected_market and train_ds.market != args.expected_market:
        raise ValueError(f"Expected market {args.expected_market}, but got {train_ds.market}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = train_ds[0]
    input_dim = int(sample["x"].shape[-1])
    model = BiMambaFreqModel(
        BiMambaFreqConfig(
            input_dim=input_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
            num_classes=1,
            d_state=args.d_state,
            low_rank=args.low_rank,
            freq_group_dt=(0.1, 1.0, 10.0),
            dropout=args.dropout,
            pscan=not args.disable_pscan,
            pool="last",
        )
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        scheduler_is_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        scheduler_is_plateau = True
    stopper = EarlyStopping(patience=args.patience, mode="max")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    task_name = f"finmultitime/{train_ds.market}"
    exp_dir = create_experiment_dir(args.experiments_root, task_name)
    save_json(exp_dir / "config.json", {**vars(args), "market": train_ds.market, "market_type": train_ds.market_type})
    best_path = exp_dir / "best.pt"
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y_cls = batch["y_cls"].to(device)
            y_vol = batch["y_vol"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(x)
                losses = bimamba_freq_multitask_loss(outputs, y_cls, y_vol, alpha=1.0, beta=1.0, gamma=0.01)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            batch_size = x.size(0)
            running_loss += losses["loss"].item() * batch_size
            running_items += batch_size

        train_loss = running_loss / max(running_items, 1)
        val_metrics = _evaluate(model, val_loader, device)
        score = val_metrics["combined_score"]
        if scheduler_is_plateau:
            scheduler.step(score)
        else:
            scheduler.step()
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), **{f"val_{k}": float(v) for k, v in val_metrics.items()}})

        improved = stopper.step(score)
        if improved:
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "market": train_ds.market,
                    "market_type": train_ds.market_type,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        if stopper.should_stop:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = _evaluate(model, test_loader, device)
    save_json(exp_dir / "history.json", {"epochs": history, "best_val": checkpoint["val_metrics"], "test": test_metrics})
    return exp_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_finmultitime")
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--expected-market", type=str, default=None)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--low-rank", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="cosine")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--disable-pscan", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    exp_dir = train_financial_multitask(args)
    print(f"saved experiment to {exp_dir}")


if __name__ == "__main__":
    main()

