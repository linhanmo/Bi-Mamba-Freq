import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from models import BiMambaFreqConfig, BiMambaFreqModel, bimamba_freq_multitask_loss
from preprocess.finmultitime import FinMultiTimeMultiTaskDataset
from utils import EarlyStopping, create_experiment_dir, multitask_finance_metrics, save_json, seed_everything, to_jsonable


def _get_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    except Exception:
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


def _log_message(message: str, log_file=None) -> None:
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def _amp_autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _grad_scaler(device: torch.device, enabled: bool):
    scaler_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def _best_f1_threshold(cls_prob: torch.Tensor, cls_true: torch.Tensor, num_thresholds: int = 81) -> Dict[str, float]:
    probs = cls_prob.detach().float().view(-1).cpu()
    targets = cls_true.detach().float().view(-1).cpu()
    if probs.numel() == 0:
        return {
            "best_f1": 0.0,
            "best_threshold": 0.5,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_pred_positive_rate": 0.0,
        }
    thresholds = torch.linspace(0.05, 0.95, steps=max(2, int(num_thresholds)))
    pred = probs.unsqueeze(0) >= thresholds.unsqueeze(1)
    true_pos = targets.unsqueeze(0) >= 0.5
    tp = (pred & true_pos).sum(dim=1).float()
    fp = (pred & (~true_pos)).sum(dim=1).float()
    fn = ((~pred) & true_pos).sum(dim=1).float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    best_idx = int(torch.argmax(f1).item())
    best_threshold = float(thresholds[best_idx].item())
    best_pred_pos = float(pred[best_idx].float().mean().item())
    return {
        "best_f1": float(f1[best_idx].item()),
        "best_threshold": best_threshold,
        "best_precision": float(precision[best_idx].item()),
        "best_recall": float(recall[best_idx].item()),
        "best_pred_positive_rate": best_pred_pos,
    }


def _validate_processed_manifest(dataset: FinMultiTimeMultiTaskDataset) -> None:
    feature_names = dataset.manifest.get("feature_names", [])
    expected_prefix = [
        "day_open",
        "day_high",
        "day_low",
        "day_close",
        "day_volume",
        "week_open",
        "week_high",
        "week_low",
        "week_close",
        "week_volume",
        "month_open",
        "month_high",
        "month_low",
        "month_close",
        "month_volume",
    ]
    if feature_names != expected_prefix:
        raise ValueError(
            "Processed FinMultitime features are stale or incompatible. "
            "Please regenerate the market dataset with the updated preprocess_finmultitime pipeline."
        )


def _summarize_targets(loader, device, split_name: str) -> Dict[str, float]:
    total_items = 0
    pos_sum = 0.0
    vol_sum = 0.0
    vol_sq_sum = 0.0
    vol_min = None
    vol_max = None
    for batch in loader:
        y_cls = batch["y_cls"]
        y_vol = batch["y_vol"]
        total_items += int(y_cls.numel())
        pos_sum += float(y_cls.sum().item())
        vol_sum += float(y_vol.sum().item())
        vol_sq_sum += float((y_vol * y_vol).sum().item())
        batch_min = float(y_vol.min().item())
        batch_max = float(y_vol.max().item())
        vol_min = batch_min if vol_min is None else min(vol_min, batch_min)
        vol_max = batch_max if vol_max is None else max(vol_max, batch_max)
    if total_items == 0:
        return {
            "split": split_name,
            "count": 0.0,
            "positive_rate": 0.0,
            "vol_mean": 0.0,
            "vol_std": 0.0,
            "vol_min": 0.0,
            "vol_max": 0.0,
        }
    vol_mean = vol_sum / total_items
    vol_var = max(vol_sq_sum / total_items - vol_mean * vol_mean, 0.0)
    return {
        "split": split_name,
        "count": float(total_items),
        "positive_rate": pos_sum / total_items,
        "vol_mean": vol_mean,
        "vol_std": vol_var**0.5,
        "vol_min": float(vol_min if vol_min is not None else 0.0),
        "vol_max": float(vol_max if vol_max is not None else 0.0),
    }


def _empty_target_stats(dataset: FinMultiTimeMultiTaskDataset, split_name: str) -> Dict[str, float]:
    return {
        "split": split_name,
        "count": float(dataset.manifest.get("split_window_counts", {}).get(split_name, len(dataset))),
        "positive_rate": float("nan"),
        "vol_mean": float("nan"),
        "vol_std": float("nan"),
        "vol_min": float("nan"),
        "vol_max": float("nan"),
    }


def _evaluate(model, loader, device, amp: bool = False, desc: Optional[str] = None):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vol_loss = 0.0
    total_consistency_loss = 0.0
    total_items = 0
    cls_probs = []
    cls_targets = []
    vol_preds = []
    vol_targets = []
    tqdm = _get_tqdm()
    iterator = loader
    if tqdm is not None and desc:
        iterator = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in iterator:
            x = batch["x"].to(device, non_blocking=True)
            y_cls = batch["y_cls"].to(device, non_blocking=True)
            y_vol = batch["y_vol"].to(device, non_blocking=True)
            with _amp_autocast(device, enabled=amp):
                outputs = model(x)
                losses = bimamba_freq_multitask_loss(outputs, y_cls, y_vol, alpha=1.0, beta=1.0, gamma=0.01)
            batch_size = x.size(0)
            total_loss += losses["loss"].item() * batch_size
            total_cls_loss += losses["cls_loss"].item() * batch_size
            total_vol_loss += losses["vol_loss"].item() * batch_size
            total_consistency_loss += losses["consistency_loss"].item() * batch_size
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
    threshold_scan = _best_f1_threshold(torch.cat(cls_probs, dim=0), torch.cat(cls_targets, dim=0))
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["cls_loss"] = total_cls_loss / max(total_items, 1)
    metrics["vol_loss"] = total_vol_loss / max(total_items, 1)
    metrics["consistency_loss"] = total_consistency_loss / max(total_items, 1)
    metrics["target_positive_rate"] = float(torch.cat(cls_targets, dim=0).float().mean().item())
    pred_probs = torch.cat(cls_probs, dim=0).float()
    metrics["pred_positive_rate"] = float((pred_probs >= 0.5).float().mean().item())
    metrics["prob_mean"] = float(pred_probs.mean().item())
    metrics["prob_min"] = float(pred_probs.min().item())
    metrics["prob_max"] = float(pred_probs.max().item())
    metrics["best_f1"] = float(threshold_scan["best_f1"])
    metrics["best_f1_threshold"] = float(threshold_scan["best_threshold"])
    metrics["best_precision"] = float(threshold_scan["best_precision"])
    metrics["best_recall"] = float(threshold_scan["best_recall"])
    metrics["best_pred_positive_rate"] = float(threshold_scan["best_pred_positive_rate"])
    return metrics


def train_financial_multitask(args) -> Path:
    seed_everything(args.seed)
    if args.num_workers is None:
        args.num_workers = _default_num_workers()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

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
    _validate_processed_manifest(train_ds)

    if args.expected_market and train_ds.market != args.expected_market:
        raise ValueError(f"Expected market {args.expected_market}, but got {train_ds.market}")

    loader_kwargs = _loader_kwargs(device, args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    task_name = f"finmultitime/{train_ds.market}"
    exp_dir = create_experiment_dir(args.experiments_root, task_name)
    log_path = exp_dir / "train.log"
    monitor_metric = "auc_roc"

    with log_path.open("w", encoding="utf-8") as log_file:
        _log_message(f"[{train_ds.market}] train log: {log_path}", log_file)
        if args.summarize_targets:
            train_target_stats = _summarize_targets(train_loader, device, "train")
            val_target_stats = _summarize_targets(val_loader, device, "val")
            test_target_stats = _summarize_targets(test_loader, device, "test")
            for stats in (train_target_stats, val_target_stats, test_target_stats):
                _log_message(
                    f"[{train_ds.market}] {stats['split']} targets: count={int(stats['count'])} "
                    f"positive_rate={stats['positive_rate']:.4f} "
                    f"vol_mean={stats['vol_mean']:.4f} vol_std={stats['vol_std']:.4f} "
                    f"vol_min={stats['vol_min']:.4f} vol_max={stats['vol_max']:.4f}",
                    log_file,
                )
        else:
            train_target_stats = _empty_target_stats(train_ds, "train")
            val_target_stats = _empty_target_stats(val_ds, "val")
            test_target_stats = _empty_target_stats(test_ds, "test")
            _log_message(
                f"[{train_ds.market}] skipping target summary scan for faster startup "
                f"(windows train={int(train_target_stats['count'])}, val={int(val_target_stats['count'])}, test={int(test_target_stats['count'])})",
                log_file,
            )

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
        scaler = _grad_scaler(device, enabled=args.amp)
        tqdm = _get_tqdm()
        grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))

        save_json(
            exp_dir / "config.json",
            {**vars(args), "market": train_ds.market, "market_type": train_ds.market_type, "monitor_metric": monitor_metric},
        )
        best_path = exp_dir / "best.pt"
        history: List[Dict[str, float]] = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_items = 0
            train_iter = train_loader
            if tqdm is not None:
                train_iter = tqdm(train_loader, desc=f"Train {train_ds.market} e{epoch}", leave=False)
            optimizer.zero_grad(set_to_none=True)
            last_step = 0
            for step, batch in enumerate(train_iter, start=1):
                x = batch["x"].to(device, non_blocking=True)
                y_cls = batch["y_cls"].to(device, non_blocking=True)
                y_vol = batch["y_vol"].to(device, non_blocking=True)
                with _amp_autocast(device, enabled=args.amp):
                    outputs = model(x)
                    losses = bimamba_freq_multitask_loss(outputs, y_cls, y_vol, alpha=1.0, beta=1.0, gamma=0.01)
                scaler.scale(losses["loss"] / grad_accum_steps).backward()
                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                batch_size = x.size(0)
                running_loss += losses["loss"].item() * batch_size
                running_items += batch_size
                if tqdm is not None:
                    train_iter.set_postfix(loss=f"{losses['loss'].item():.4f}")
                last_step = step

            if last_step % grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss = running_loss / max(running_items, 1)
            val_metrics = _evaluate(model, val_loader, device, amp=args.amp, desc=f"Val {train_ds.market} e{epoch}")
            score = val_metrics[monitor_metric]
            if scheduler_is_plateau:
                scheduler.step(score)
            else:
                scheduler.step()
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_monitor_metric": monitor_metric,
                    "val_monitor_score": float(score),
                    **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                }
            )
            _log_message(
                f"[{train_ds.market}] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
                f"cls_loss={val_metrics['cls_loss']:.6f} vol_loss={val_metrics['vol_loss']:.6f} "
                f"cons={val_metrics['consistency_loss']:.6f} "
                f"f1@0.5={val_metrics['f1']:.4f} auc={val_metrics['auc_roc']:.4f} "
                f"p={val_metrics['precision']:.4f} r={val_metrics['recall']:.4f} "
                f"best_f1={val_metrics['best_f1']:.4f}@{val_metrics['best_f1_threshold']:.2f} "
                f"best_p={val_metrics['best_precision']:.4f} best_r={val_metrics['best_recall']:.4f} "
                f"rmse={val_metrics['rmse']:.4f} combined={val_metrics['combined_score']:.4f} "
                f"monitor_{monitor_metric}={score:.4f} "
                f"target_pos={val_metrics['target_positive_rate']:.4f} pred_pos={val_metrics['pred_positive_rate']:.4f} "
                f"prob_mean={val_metrics['prob_mean']:.4f} prob_range=[{val_metrics['prob_min']:.4f},{val_metrics['prob_max']:.4f}]",
                log_file,
            )

            improved = stopper.step(score)
            if improved:
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": to_jsonable(vars(args)),
                        "market": train_ds.market,
                        "market_type": train_ds.market_type,
                        "val_metrics": to_jsonable(val_metrics),
                    },
                    best_path,
                )
            if stopper.should_stop:
                break

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_metrics = _evaluate(model, test_loader, device, amp=args.amp, desc=f"Test {train_ds.market}")
        save_json(
            exp_dir / "history.json",
            {
                "target_stats": {
                    "train": train_target_stats,
                    "val": val_target_stats,
                    "test": test_target_stats,
                },
                "epochs": history,
                "best_val": checkpoint["val_metrics"],
                "test": test_metrics,
            },
        )
        _log_message(f"[{train_ds.market}] saved history to {exp_dir / 'history.json'}", log_file)
    return exp_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_finmultitime")
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--expected-market", type=str, default=None)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--low-rank", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="cosine")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summarize-targets", action="store_true")
    amp_group = p.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--disable-pscan", action="store_true", default=False)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    exp_dir = train_financial_multitask(args)
    print(f"saved experiment to {exp_dir}")
    log_path = exp_dir / "train.log"
    if log_path.exists():
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"saved experiment to {exp_dir}\n")


if __name__ == "__main__":
    main()
