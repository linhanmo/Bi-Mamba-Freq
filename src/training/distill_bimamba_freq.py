import argparse
import json
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
MAMBA_SRC_DIR = SRC_DIR / "mamba"
for p in (SRC_DIR, MAMBA_SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import torch

from models import BiMambaFreqConfig, BiMambaFreqModel
from training.train_bimamba_freq import (
    _infer_feature_metadata,
    _resolve_sources,
    load_checkpoint_to_model,
    parse_device,
    resume_training_state,
    run_epoch,
    save_interpret_snapshot,
    seed_everything,
)


def _maybe_parse_band_weights(text: str) -> List[float] | None:
    text = text.strip()
    if not text:
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_message(log_path: Path, message: str) -> None:
    line = f"[{_now_text()}] {message}"
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_jsonl(jsonl_path: Path, record: Dict[str, object]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_model_from_checkpoint(checkpoint_path: Path, input_dim: int, device: torch.device) -> BiMambaFreqModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_config" not in ckpt:
        raise ValueError(f"checkpoint 缺少 model_config: {checkpoint_path}")
    model_config = dict(ckpt["model_config"])
    model_config["input_dim"] = input_dim
    config = BiMambaFreqConfig(**model_config)
    model = BiMambaFreqModel(config).to(device)
    load_checkpoint_to_model(model=model, checkpoint_path=checkpoint_path, device=device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 post-training distillation for BiMambaFreqModel.")
    parser.add_argument("--data", type=str, required=True, help="Path to shards manifest json or merged npz.")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--reg_loss_weight", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_interpret", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--distill_max_lambda", type=float, default=1.0)
    parser.add_argument("--distill_min_lambda", type=float, default=0.0)
    parser.add_argument("--distill_kl_weight", type=float, default=1.0)
    parser.add_argument("--distill_freq_weight", type=float, default=1.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--distill_wavelet_levels", type=int, default=3)
    parser.add_argument("--distill_band_weights", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = parse_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "distill.log"
    metrics_jsonl_path = output_dir / "distill_metrics.jsonl"
    log_path.write_text("", encoding="utf-8")
    metrics_jsonl_path.write_text("", encoding="utf-8")

    source_files, manifest = _resolve_sources(Path(args.data))
    feature_cols, _ = _infer_feature_metadata(source_files, manifest)
    input_dim = len(feature_cols)

    teacher_model = build_model_from_checkpoint(Path(args.teacher_checkpoint), input_dim=input_dim, device=device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    student_model = build_model_from_checkpoint(Path(args.student_checkpoint), input_dim=input_dim, device=device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    train_history: List[Dict[str, float]] = []
    best_val = float("inf")
    start_epoch = 0
    stale_epochs = 0
    band_weights = _maybe_parse_band_weights(args.distill_band_weights)
    best_path = output_dir / "best_distilled_model.pt"

    if args.resume_checkpoint:
        start_epoch, best_val, train_history = resume_training_state(
            checkpoint_path=Path(args.resume_checkpoint),
            model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"[Resume Distill] start_epoch={start_epoch} best_val={best_val:.6f} from {args.resume_checkpoint}")

    run_config = {
        "stage": "post_training_distillation",
        "args": vars(args),
        "feature_cols": feature_cols,
        "device": str(device),
        "student_model_config": asdict(student_model.config),
        "teacher_model_config": asdict(teacher_model.config),
    }
    (output_dir / "distill_run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_message(log_path, f"distill start data={args.data} output_dir={output_dir}")
    _log_message(log_path, f"teacher={args.teacher_checkpoint} student={args.student_checkpoint} device={device}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=student_model,
            source_files=source_files,
            split="train",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=optimizer,
            reg_loss_weight=args.reg_loss_weight,
            max_grad_norm=args.max_grad_norm,
            teacher_model=teacher_model,
            distill_max_lambda=args.distill_max_lambda,
            distill_min_lambda=args.distill_min_lambda,
            epoch=epoch,
            total_epochs=args.epochs,
            kl_weight=args.distill_kl_weight,
            freq_weight=args.distill_freq_weight,
            wavelet_levels=args.distill_wavelet_levels,
            distill_temperature=args.distill_temperature,
            band_weights=band_weights,
        )
        val_metrics = run_epoch(
            model=student_model,
            source_files=source_files,
            split="val",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=None,
            reg_loss_weight=args.reg_loss_weight,
            max_grad_norm=args.max_grad_norm,
            teacher_model=teacher_model,
            distill_max_lambda=args.distill_max_lambda,
            distill_min_lambda=args.distill_min_lambda,
            epoch=epoch,
            total_epochs=args.epochs,
            kl_weight=args.distill_kl_weight,
            freq_weight=args.distill_freq_weight,
            wavelet_levels=args.distill_wavelet_levels,
            distill_temperature=args.distill_temperature,
            band_weights=band_weights,
        )
        scheduler.step()

        record = {"epoch": float(epoch + 1)}
        for k, v in train_metrics.items():
            record[f"train_{k}"] = float(v)
        for k, v in val_metrics.items():
            record[f"val_{k}"] = float(v)
        train_history.append(record)
        _append_jsonl(metrics_jsonl_path, record)

        checkpoint = {
            "stage": "post_training_distillation",
            "epoch": epoch + 1,
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": asdict(student_model.config),
            "feature_cols": feature_cols,
            "history": train_history,
            "teacher_checkpoint": args.teacher_checkpoint,
            "student_init_checkpoint": args.student_checkpoint,
        }

        if (epoch + 1) % max(1, args.save_every) == 0:
            torch.save(checkpoint, output_dir / f"distill_checkpoint_epoch_{epoch + 1:03d}.pt")

        val_score = float(val_metrics["loss"])
        if val_score < best_val:
            best_val = val_score
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            if bool(int(args.save_interpret)):
                save_interpret_snapshot(
                    model=student_model,
                    source_files=source_files,
                    device=device,
                    output_path=output_dir / "distill_interpret_snapshot_val.npz",
                    split="val",
                    feature_cols=feature_cols,
                )
        else:
            stale_epochs += 1

        _log_message(
            log_path,
            f"[Distill {epoch + 1:03d}/{args.epochs:03d}] "
            f"train_f1={train_metrics['f1_score']:.4f} train_acc={train_metrics['acc']:.4f} train_mcc={train_metrics['mcc']:.4f} "
            f"val_f1={val_metrics['f1_score']:.4f} val_acc={val_metrics['acc']:.4f} val_mcc={val_metrics['mcc']:.4f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f} distill={train_metrics['distill']:.6f}"
        )

        if args.early_stopping_patience > 0 and stale_epochs >= int(args.early_stopping_patience):
            _log_message(log_path, f"[EarlyStop Distill] no improvement for {stale_epochs} epochs, stopping at epoch {epoch + 1}")
            break

    final_test = run_epoch(
        model=student_model,
        source_files=source_files,
        split="test",
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=None,
        reg_loss_weight=args.reg_loss_weight,
        max_grad_norm=args.max_grad_norm,
        teacher_model=teacher_model,
        distill_max_lambda=args.distill_max_lambda,
        distill_min_lambda=args.distill_min_lambda,
        epoch=max(args.epochs - 1, 0),
        total_epochs=args.epochs,
        kl_weight=args.distill_kl_weight,
        freq_weight=args.distill_freq_weight,
        wavelet_levels=args.distill_wavelet_levels,
        distill_temperature=args.distill_temperature,
        band_weights=band_weights,
    )

    summary = {
        "stage": "post_training_distillation",
        "best_val_loss": best_val,
        "teacher_checkpoint": str(Path(args.teacher_checkpoint)),
        "student_init_checkpoint": str(Path(args.student_checkpoint)),
        "best_checkpoint": str(best_path),
        "test": final_test,
        "history": train_history,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    _log_message(log_path, f"distill done best_checkpoint={best_path}")
    _log_message(log_path, text)
    (output_dir / "distill_summary.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
