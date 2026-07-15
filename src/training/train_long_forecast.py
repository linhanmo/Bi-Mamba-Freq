import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from models import BiMambaFreqForecastConfig, BiMambaFreqForecastModel
from preprocess.datasets import TimeSeriesWindowDataset, preprocess_csv_dataset, resolve_dataset_defaults
from utils import EarlyStopping, create_experiment_dir, forecast_metrics, save_json, seed_everything, to_jsonable


def _get_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _find_series_file(dataset_dir: Path) -> Optional[Path]:
    for name in ("series.npz", "series.pkl"):
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(0, min(4, cpu_count - 1))


def _loader_kwargs(device: torch.device, num_workers: int) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _processed_data_matches(meta_path: Path, args) -> bool:
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    defaults = resolve_dataset_defaults(
        dataset_name=args.dataset_name,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        time_features=args.time_features,
        normalize=args.normalize,
    )
    return (
        int(meta.get("seq_len", -1)) == int(args.seq_len)
        and int(meta.get("pred_len", -1)) == int(args.pred_len)
        and float(meta.get("train_ratio", -1.0)) == float(defaults["train_ratio"])
        and float(meta.get("val_ratio", -1.0)) == float(defaults["val_ratio"])
        and meta.get("time_features") == defaults["time_features"]
        and meta.get("split_scheme") == defaults["split_scheme"]
        and meta.get("normalize") == defaults["normalize"]
    )


def _prepare_dataset(args) -> Path:
    dataset_dir = args.processed_root / args.dataset_name
    series_file = _find_series_file(dataset_dir)
    meta_path = dataset_dir / "meta.json"
    if series_file is not None and _processed_data_matches(meta_path, args):
        return series_file

    raw_csv = args.raw_root / f"{args.dataset_name}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw csv: {raw_csv}")
    defaults = resolve_dataset_defaults(
        dataset_name=args.dataset_name,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        time_features=args.time_features,
        normalize=args.normalize,
    )
    preprocess_csv_dataset(
        csv_path=raw_csv,
        out_dir=dataset_dir,
        seq_len=args.seq_len,
        pred_len=defaults["pred_len"],
        stride=args.stride,
        train_ratio=defaults["train_ratio"],
        val_ratio=defaults["val_ratio"],
        time_features=defaults["time_features"],
        output_format="full",
        dataset_name=args.dataset_name,
        normalize=defaults["normalize"],
    )
    series_file = _find_series_file(dataset_dir)
    if series_file is None:
        raise FileNotFoundError(f"Failed to create processed series file under {dataset_dir}")
    return series_file


def _evaluate(model, loader, device, amp_enabled: bool, desc: Optional[str] = None):
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds = []
    targets = []
    criterion = torch.nn.MSELoss()
    tqdm = _get_tqdm()
    iterator = loader
    if tqdm is not None and desc:
        iterator = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in iterator:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
                outputs = model(x)
                pred = outputs["prediction"]
                loss = criterion(pred, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = forecast_metrics(pred_all, target_all)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def train_long_forecast(args) -> Path:
    seed_everything(args.seed)
    if args.num_workers is None:
        args.num_workers = _default_num_workers()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    series_file = _prepare_dataset(args)
    train_ds = TimeSeriesWindowDataset(series_file, split="train", seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride)
    val_ds = TimeSeriesWindowDataset(series_file, split="val", seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride)
    test_ds = TimeSeriesWindowDataset(series_file, split="test", seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride)

    loader_kwargs = _loader_kwargs(device, args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    sample_x, sample_y = train_ds[0][0], train_ds[0][1]
    if args.low_rank is None:
        args.low_rank = max(1, int(round(float(args.d_model) * float(args.rank_ratio))))
    model = BiMambaFreqForecastModel(
        BiMambaFreqForecastConfig(
            input_dim=sample_x.shape[-1],
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            target_dim=sample_y.shape[-1],
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            low_rank=args.low_rank,
            dropout=args.dropout,
            pscan=not args.disable_pscan,
        )
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    stopper = EarlyStopping(patience=args.patience, mode="min")
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    tqdm = _get_tqdm()

    exp_dir = create_experiment_dir(args.experiments_root, f"long_forecast/{args.dataset_name}_h{args.pred_len}")
    save_json(exp_dir / "config.json", {**vars(args), "loss": "mse"})
    history: List[Dict[str, float]] = []
    best_path = exp_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        train_iter = train_loader
        if tqdm is not None:
            train_iter = tqdm(train_loader, desc=f"Train {args.dataset_name} h{args.pred_len} e{epoch}", leave=False)
        for batch in train_iter:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(x)
                loss = criterion(outputs["prediction"], y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            running_items += batch_size
            if tqdm is not None:
                train_iter.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_items, 1)
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            amp_enabled=args.amp,
            desc=f"Val {args.dataset_name} h{args.pred_len} e{epoch}",
        )
        scheduler.step(val_metrics["loss"])
        row = {"epoch": float(epoch), "train_loss": float(train_loss), **{f"val_{k}": float(v) for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"[{args.dataset_name} h{args.pred_len}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f}"
        )

        improved = stopper.step(val_metrics["loss"])
        if improved:
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": to_jsonable(vars(args)), "val_metrics": to_jsonable(val_metrics)}, best_path)
        if stopper.should_stop:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = _evaluate(model, test_loader, device, amp_enabled=args.amp, desc=f"Test {args.dataset_name} h{args.pred_len}")
    save_json(exp_dir / "history.json", {"epochs": history, "best_val": checkpoint["val_metrics"], "test": test_metrics})
    return exp_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_long_forecast")
    p.add_argument("--dataset-name", type=str, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("datasets"))
    p.add_argument("--processed-root", type=Path, default=Path("data"))
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--low-rank", type=int, default=None)
    p.add_argument("--rank-ratio", type=float, default=0.5)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-features", choices=["none", "simple"], default=None)
    p.add_argument("--normalize", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--disable-pscan", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    exp_dir = train_long_forecast(args)
    print(f"saved experiment to {exp_dir}")


if __name__ == "__main__":
    main()
