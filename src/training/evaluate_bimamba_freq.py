import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
MAMBA_SRC_DIR = SRC_DIR / "mamba"
for p in (SRC_DIR, MAMBA_SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import numpy as np
import torch

from models import BiMambaFreqConfig, BiMambaFreqModel
from training.train_bimamba_freq import (
    _infer_feature_metadata,
    _make_loader_for_split,
    _resolve_sources,
    parse_device,
    run_epoch,
    save_interpret_snapshot,
)


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, input_dim: int) -> BiMambaFreqModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("checkpoint 必须包含 model_state")
    model_config = ckpt.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少 model_config")
    model_config = dict(model_config)
    model_config["input_dim"] = input_dim
    config = BiMambaFreqConfig(**model_config)
    model = BiMambaFreqModel(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def collect_predictions(
    model: BiMambaFreqModel,
    source_files: Sequence[Path],
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int = 0,
) -> Dict[str, np.ndarray]:
    logits_list: List[np.ndarray] = []
    vol_list: List[np.ndarray] = []
    y_cls_list: List[np.ndarray] = []
    y_reg_list: List[np.ndarray] = []
    pred_cls_list: List[np.ndarray] = []

    seen_batches = 0
    for src in source_files:
        loader = _make_loader_for_split(npz_path=src, split=split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if loader is None:
            continue
        for X, y_cls, y_reg in loader:
            X = X.to(device)
            with torch.no_grad():
                outputs = model(X, return_interpret=False)
            logits = outputs["logits"].cpu().numpy()
            volatility = outputs["volatility"].cpu().numpy()
            pred_cls = logits.argmax(axis=-1)
            logits_list.append(logits)
            vol_list.append(volatility)
            y_cls_list.append(y_cls.numpy())
            y_reg_list.append(y_reg.numpy())
            pred_cls_list.append(pred_cls)
            seen_batches += 1
            if max_batches > 0 and seen_batches >= max_batches:
                break
        if max_batches > 0 and seen_batches >= max_batches:
            break

    if not logits_list:
        return {}
    return {
        "logits": np.concatenate(logits_list, axis=0),
        "volatility": np.concatenate(vol_list, axis=0),
        "y_cls": np.concatenate(y_cls_list, axis=0),
        "y_reg": np.concatenate(y_reg_list, axis=0),
        "pred_cls": np.concatenate(pred_cls_list, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BiMambaFreqModel checkpoints.")
    parser.add_argument("--data", type=str, required=True, help="Path to shards manifest json or merged npz.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_predictions", type=int, default=1)
    parser.add_argument("--save_interpret", type=int, default=1)
    parser.add_argument("--max_prediction_batches", type=int, default=0)
    args = parser.parse_args()

    device = parse_device(args.device)
    source_files, manifest = _resolve_sources(Path(args.data))
    feature_cols, _ = _infer_feature_metadata(source_files, manifest)
    model = _load_model_from_checkpoint(Path(args.checkpoint), device=device, input_dim=len(feature_cols))

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    summary: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "feature_cols": feature_cols,
        "results": {},
    }

    output_dir: Optional[Path] = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        metrics = run_epoch(
            model=model,
            source_files=source_files,
            split=split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=None,
            reg_loss_weight=1.0,
            max_grad_norm=0.0,
            teacher_model=None,
            distill_max_lambda=0.0,
            distill_min_lambda=0.0,
            epoch=0,
            total_epochs=1,
            kl_weight=0.0,
            freq_weight=0.0,
            wavelet_levels=1,
            distill_temperature=1.0,
            band_weights=None,
        )
        summary["results"][split] = metrics

        if output_dir is not None and bool(int(args.save_predictions)):
            pred = collect_predictions(
                model=model,
                source_files=source_files,
                split=split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_batches=int(args.max_prediction_batches),
            )
            if pred:
                np.savez_compressed(
                    str(output_dir / f"{split}_predictions.npz"),
                    **pred,
                    feature_cols=np.asarray(feature_cols, dtype=object),
                )

        if output_dir is not None and bool(int(args.save_interpret)):
            save_interpret_snapshot(
                model=model,
                source_files=source_files,
                device=device,
                output_path=output_dir / f"{split}_interpret_snapshot.npz",
                split=split,
                feature_cols=feature_cols,
            )

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if output_dir is not None:
        (output_dir / "evaluation_summary.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
