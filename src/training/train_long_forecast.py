import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from models import BiMambaFreqForecastConfig, BiMambaFreqForecastModel
from preprocess.datasets import TimeSeriesWindowDataset, preprocess_csv_dataset
from utils import EarlyStopping, create_experiment_dir, forecast_metrics, save_json, seed_everything


def _find_series_file(dataset_dir: Path) -> Optional[Path]:
    for name in ("series.npz", "series.pkl"):
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def _prepare_dataset(args) -> Path:
    dataset_dir = args.processed_root / args.dataset_name
    series_file = _find_series_file(dataset_dir)
    if series_file is not None:
        return series_file

    raw_csv = args.raw_root / f"{args.dataset_name}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw csv: {raw_csv}")
    preprocess_csv_dataset(
        csv_path=raw_csv,
        out_dir=dataset_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        time_features="simple",
        output_format="full",
    )
    series_file = _find_series_file(dataset_dir)
    if series_file is None:
        raise FileNotFoundError(f"Failed to create processed series file under {dataset_dir}")
    return series_file


def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds = []
    targets = []
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    series_file = _prepare_dataset(args)
    train_ds = TimeSeriesWindowDataset(series_file, split="train", seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride)
    val_ds = TimeSeriesWindowDataset(series_file, split="val", seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride)
    test_ds = TimeSeriesWindowDataset(series_file, split="test", seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample_x, sample_y = train_ds[0][0], train_ds[0][1]
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    stopper = EarlyStopping(patience=args.patience, mode="min")
    criterion = torch.nn.MSELoss()

    exp_dir = create_experiment_dir(args.experiments_root, f"long_forecast/{args.dataset_name}_h{args.pred_len}")
    save_json(exp_dir / "config.json", vars(args))
    history: List[Dict[str, float]] = []
    best_path = exp_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs["prediction"], y)
            loss.backward()
            optimizer.step()
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            running_items += batch_size

        train_loss = running_loss / max(running_items, 1)
        val_metrics = _evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        row = {"epoch": float(epoch), "train_loss": float(train_loss), **{f"val_{k}": float(v) for k, v in val_metrics.items()}}
        history.append(row)

        improved = stopper.step(val_metrics["loss"])
        if improved:
            torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics}, best_path)
        if stopper.should_stop:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = _evaluate(model, test_loader, device)
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
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--low-rank", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--disable-pscan", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    exp_dir = train_long_forecast(args)
    print(f"saved experiment to {exp_dir}")


if __name__ == "__main__":
    main()

