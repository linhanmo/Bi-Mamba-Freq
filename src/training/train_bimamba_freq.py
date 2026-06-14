import argparse
import json
import math
import random
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
MAMBA_SRC_DIR = SRC_DIR / "mamba"
for p in (SRC_DIR, MAMBA_SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import BiMambaFreqConfig, BiMambaFreqModel
from training.distillation import (
    bimambafreq_total_loss,
    freq_alignment_loss,
    soft_target_kl_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


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


def _load_npz_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _resolve_sources(data_path: Path) -> Tuple[List[Path], Optional[Dict[str, object]]]:
    if data_path.suffix == ".json":
        manifest = json.loads(data_path.read_text(encoding="utf-8"))
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"manifest 中没有有效的 shards: {data_path}")
        base_dir = data_path.parent
        files = [base_dir / str(item["file"]) for item in shards if isinstance(item, dict) and "file" in item]
        return files, manifest
    if data_path.suffix == ".npz":
        return [data_path], None
    raise ValueError(f"只支持 manifest(.json) 或单文件 npz: {data_path}")


def _infer_feature_metadata(files: Sequence[Path], manifest: Optional[Dict[str, object]]) -> Tuple[List[str], List[str]]:
    if manifest is not None:
        feature_cols = manifest.get("feature_cols")
        binary_cols = manifest.get("binary_feature_cols")
        if isinstance(feature_cols, list):
            return [str(x) for x in feature_cols], [str(x) for x in binary_cols] if isinstance(binary_cols, list) else []

    arrays = _load_npz_arrays(files[0])
    feature_cols = arrays.get("feature_cols")
    if feature_cols is None:
        raise ValueError("npz 中缺少 feature_cols")
    return [str(x) for x in feature_cols.tolist()], []


def _infer_num_classes(files: Sequence[Path], manifest: Optional[Dict[str, object]]) -> int:
    if manifest is not None:
        num_classes = manifest.get("num_classes")
        if isinstance(num_classes, int) and num_classes >= 2:
            return int(num_classes)

    arrays = _load_npz_arrays(files[0])
    num_classes_arr = arrays.get("num_classes")
    if num_classes_arr is not None:
        flat = np.asarray(num_classes_arr).reshape(-1)
        if flat.size > 0:
            return max(int(flat[0]), 2)
    candidates: List[np.ndarray] = []
    for key in ("y_train_cls", "y_val_cls", "y_test_cls"):
        arr = arrays.get(key)
        if arr is not None and len(arr) > 0:
            candidates.append(np.asarray(arr))
    if not candidates:
        return 3
    values = np.concatenate([x.reshape(-1) for x in candidates], axis=0)
    uniq = np.unique(values)
    return max(int(len(uniq)), 2)


class SequenceArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray):
        if len(X) != len(y_cls) or len(X) != len(y_reg):
            raise ValueError("X/y 长度不一致")
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y_cls = torch.from_numpy(np.asarray(y_cls, dtype=np.int64))
        self.y_reg = torch.from_numpy(np.asarray(y_reg, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]


def _normalize_class_labels(y_cls: np.ndarray) -> np.ndarray:
    values = np.asarray(y_cls, dtype=np.int64)
    if values.size == 0:
        return values
    uniq = np.unique(values)
    if uniq.min() >= 0 and np.array_equal(uniq, np.arange(len(uniq), dtype=uniq.dtype)):
        return values
    mapping = {int(v): i for i, v in enumerate(sorted(int(v) for v in uniq.tolist()))}
    out = np.vectorize(lambda x: mapping[int(x)], otypes=[np.int64])(values)
    return np.asarray(out, dtype=np.int64)


def _make_loader_for_split(npz_path: Path, split: str, batch_size: int, shuffle: bool, num_workers: int) -> Optional[DataLoader]:
    arrays = _load_npz_arrays(npz_path)
    X = arrays.get(f"X_{split}")
    y_cls = arrays.get(f"y_{split}_cls")
    y_reg = arrays.get(f"y_{split}_reg")
    if X is None or y_cls is None or y_reg is None or len(X) == 0:
        return None
    y_cls_norm = _normalize_class_labels(y_cls)
    uniq = np.unique(y_cls_norm)
    if uniq.size > 0 and (uniq.min() < 0 or uniq.max() >= len(uniq)):
        raise ValueError(
            f"{npz_path.name}:{split} 标签归一化失败，unique={uniq.tolist()}"
        )
    dataset = SequenceArrayDataset(X=X, y_cls=y_cls_norm, y_reg=y_reg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def _match_hidden_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.size(-1) == target_dim:
        return x
    flat = x.reshape(-1, 1, x.size(-1))
    pooled = F.adaptive_avg_pool1d(flat, target_dim)
    return pooled.reshape(*x.shape[:-1], target_dim)


def _extract_distill_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    kl_weight: float,
    freq_weight: float,
    wavelet_levels: int,
    distill_temperature: float,
    band_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    s_interp = student_out["interpret"]
    t_interp = teacher_out["interpret"]
    kl_loss = soft_target_kl_loss(
        student_logits=student_out["logits"],
        teacher_logits=teacher_out["logits"],
        temperature=distill_temperature,
    )

    s_hf = s_interp["h_fwd"]
    s_hb = s_interp["h_bwd"]
    t_hf = _match_hidden_dim(t_interp["h_fwd"], s_hf.size(-1))
    t_hb = _match_hidden_dim(t_interp["h_bwd"], s_hb.size(-1))

    s_bi = torch.cat([s_hf, s_hb], dim=-1)
    t_bi = torch.cat([t_hf, t_hb], dim=-1)
    freq_loss = freq_alignment_loss(s_bi, t_bi, band_weights=band_weights, levels=wavelet_levels)
    return kl_weight * kl_loss + freq_weight * freq_loss


def _update_confusion_matrix(conf_mat: Optional[torch.Tensor], y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    if conf_mat is None:
        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    encoded = y_true.to(torch.int64) * num_classes + y_pred.to(torch.int64)
    binc = torch.bincount(encoded.cpu(), minlength=num_classes * num_classes).to(torch.float64)
    conf_mat += binc.reshape(num_classes, num_classes)
    return conf_mat


def _classification_metrics_from_confusion(conf_mat: torch.Tensor) -> Dict[str, float]:
    total = float(conf_mat.sum().item())
    if total <= 0:
        return {"acc": 0.0, "f1_score": 0.0, "mcc": 0.0}

    diag = conf_mat.diag()
    row_sum = conf_mat.sum(dim=1)
    col_sum = conf_mat.sum(dim=0)

    acc = float(diag.sum().item() / total)

    f1_list: List[float] = []
    for i in range(conf_mat.size(0)):
        tp = float(diag[i].item())
        fp = float(col_sum[i].item() - tp)
        fn = float(row_sum[i].item() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)
    f1_score = float(sum(f1_list) / len(f1_list)) if f1_list else 0.0

    c = float(diag.sum().item())
    s = total
    cov_ytyp = c * s - float(torch.dot(row_sum, col_sum).item())
    cov_ytyt = s * s - float(torch.dot(row_sum, row_sum).item())
    cov_ypyp = s * s - float(torch.dot(col_sum, col_sum).item())
    denom = math.sqrt(max(cov_ytyt, 0.0) * max(cov_ypyp, 0.0))
    mcc = float(cov_ytyp / denom) if denom > 0 else 0.0

    return {"acc": acc, "f1_score": f1_score, "mcc": mcc}


def run_epoch(
    model: BiMambaFreqModel,
    source_files: Sequence[Path],
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    optimizer: Optional[torch.optim.Optimizer],
    reg_loss_weight: float,
    max_grad_norm: float,
    teacher_model: Optional[BiMambaFreqModel],
    distill_max_lambda: float,
    distill_min_lambda: float,
    epoch: int,
    total_epochs: int,
    kl_weight: float,
    freq_weight: float,
    wavelet_levels: int,
    distill_temperature: float,
    band_weights: Optional[List[float]],
) -> Dict[str, float]:
    is_train = optimizer is not None and split == "train"
    model.train(is_train)
    if teacher_model is not None:
        teacher_model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_reg = 0.0
    total_distill = 0.0
    total_samples = 0
    total_abs = 0.0
    total_sq = 0.0
    conf_mat: Optional[torch.Tensor] = None

    for src in source_files:
        loader = _make_loader_for_split(
            npz_path=src,
            split=split,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
        )
        if loader is None:
            continue

        for X, y_cls, y_reg in loader:
            X = X.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)

            need_interpret = teacher_model is not None
            outputs = model(X, return_interpret=need_interpret)
            logits = outputs["logits"]
            y_reg_pred = outputs["volatility"].squeeze(-1)

            cls_loss = F.cross_entropy(logits, y_cls)
            reg_loss = F.mse_loss(y_reg_pred, y_reg)
            task_loss = cls_loss + reg_loss_weight * reg_loss

            distill_loss_value = torch.tensor(0.0, device=device)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(X, return_interpret=True)
                distill_loss_value = _extract_distill_loss(
                    student_out=outputs,
                    teacher_out=teacher_outputs,
                    kl_weight=kl_weight,
                    freq_weight=freq_weight,
                    wavelet_levels=wavelet_levels,
                    distill_temperature=distill_temperature,
                    band_weights=band_weights,
                )
                loss, _ = bimambafreq_total_loss(
                    task_loss=task_loss,
                    distill_loss=distill_loss_value,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    max_lambda=distill_max_lambda,
                    min_lambda=distill_min_lambda,
                )
            else:
                loss = task_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            batch_size_now = int(X.shape[0])
            total_samples += batch_size_now
            total_loss += float(loss.item()) * batch_size_now
            total_ce += float(cls_loss.item()) * batch_size_now
            total_reg += float(reg_loss.item()) * batch_size_now
            total_distill += float(distill_loss_value.item()) * batch_size_now
            cls_pred = logits.argmax(dim=-1)
            conf_mat = _update_confusion_matrix(conf_mat, y_true=y_cls, y_pred=cls_pred, num_classes=int(logits.size(-1)))
            total_abs += float(torch.abs(y_reg_pred - y_reg).sum().item())
            total_sq += float(torch.square(y_reg_pred - y_reg).sum().item())

    denom = max(total_samples, 1)
    cls_metrics = _classification_metrics_from_confusion(
        conf_mat if conf_mat is not None else torch.zeros((model.config.num_classes, model.config.num_classes), dtype=torch.float64)
    )
    return {
        "loss": total_loss / denom,
        "ce": total_ce / denom,
        "reg_mse": total_reg / denom,
        "distill": total_distill / denom,
        "f1_score": cls_metrics["f1_score"],
        "acc": cls_metrics["acc"],
        "mcc": cls_metrics["mcc"],
        "rmse": math.sqrt(total_sq / denom),
        "mae": total_abs / denom,
        "samples": float(total_samples),
    }


def save_interpret_snapshot(
    model: BiMambaFreqModel,
    source_files: Sequence[Path],
    device: torch.device,
    output_path: Path,
    split: str = "val",
    feature_cols: Optional[Sequence[str]] = None,
) -> None:
    for src in source_files:
        loader = _make_loader_for_split(npz_path=src, split=split, batch_size=1, shuffle=False, num_workers=0)
        if loader is None:
            continue
        X, y_cls, y_reg = next(iter(loader))
        X = X.to(device)
        with torch.no_grad():
            outputs = model(X, return_interpret=True)
        interp = outputs["interpret"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_path),
            x=X.cpu().numpy(),
            y_cls=y_cls.numpy(),
            y_reg=y_reg.numpy(),
            feature_cols=np.asarray(list(feature_cols), dtype=object) if feature_cols is not None else np.asarray([], dtype=object),
            logits=outputs["logits"].cpu().numpy(),
            volatility=outputs["volatility"].cpu().numpy(),
            pooled_features=outputs["features"].cpu().numpy(),
            delta_fwd=interp["delta_fwd"].cpu().numpy(),
            delta_bwd=interp["delta_bwd"].cpu().numpy(),
            bi_divergence=interp["bi_divergence"].cpu().numpy(),
            h_fwd=interp["h_fwd"].cpu().numpy(),
            h_bwd=interp["h_bwd"].cpu().numpy(),
            **{f"group_states_fwd_{k}": v.cpu().numpy() for k, v in interp["group_states_fwd"].items()},
            **{f"group_states_bwd_{k}": v.cpu().numpy() for k, v in interp["group_states_bwd"].items()},
        )
        return


def build_model_from_args(args: argparse.Namespace, input_dim: int, num_classes: int, teacher: bool = False) -> BiMambaFreqModel:
    d_model = args.teacher_d_model if teacher and args.teacher_d_model > 0 else args.d_model
    n_layers = args.teacher_n_layers if teacher and args.teacher_n_layers > 0 else args.n_layers
    d_state = args.teacher_d_state if teacher and args.teacher_d_state > 0 else args.d_state
    low_rank = args.teacher_low_rank if teacher and args.teacher_low_rank > 0 else args.low_rank
    if teacher and low_rank <= 0:
        low_rank = d_model

    cfg = BiMambaFreqConfig(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers,
        num_classes=num_classes,
        dt_rank=args.dt_rank if args.dt_rank != "auto" else "auto",
        d_state=d_state,
        expand_factor=args.expand_factor,
        d_conv=args.d_conv,
        freq_groups=args.freq_groups,
        freq_group_dt=tuple(float(x) for x in args.freq_group_dt.split(",")),
        low_rank=low_rank,
        bidirectional_merge=args.bidirectional_merge,
        pool=args.pool,
        dropout=args.dropout,
        pscan=bool(int(args.pscan)),
    )
    return BiMambaFreqModel(cfg)


def load_checkpoint_to_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}


def load_checkpoint_compatible(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError("checkpoint state 必须是参数字典")
    current = model.state_dict()
    loadable = {}
    skipped: List[str] = []
    for k, v in state.items():
        if k in current and current[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)
    current.update(loadable)
    model.load_state_dict(current, strict=True)
    print(f"[InitCheckpoint] loaded={len(loadable)} skipped={len(skipped)} from {checkpoint_path}")
    if skipped:
        print(f"[InitCheckpoint] skipped keys: {', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}")
    return ckpt if isinstance(ckpt, dict) else {}


def resume_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
) -> Tuple[int, float, List[Dict[str, float]]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("resume checkpoint 必须是训练时保存的 dict 格式")
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt.get("epoch", 0))
    history = ckpt.get("history", [])
    best_val = float("inf")
    if isinstance(history, list) and history:
        candidates = [float(x.get("val_loss", float("inf"))) for x in history if isinstance(x, dict)]
        if candidates:
            best_val = min(candidates)
    return start_epoch, best_val, history if isinstance(history, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 task training for BiMambaFreqModel on sharded or merged npz time-series datasets.")
    parser.add_argument("--data", type=str, required=True, help="Path to shards manifest json or merged npz.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--reg_loss_weight", type=float, default=0.0)
    parser.add_argument("--save_interpret", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand_factor", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--dt_rank", type=str, default="auto")
    parser.add_argument("--freq_groups", type=int, default=3)
    parser.add_argument("--freq_group_dt", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--low_rank", type=int, default=32)
    parser.add_argument("--bidirectional_merge", type=str, default="concat", choices=["concat", "sum"])
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pscan", type=int, default=1)

    args = parser.parse_args()

    seed_everything(args.seed)
    device = parse_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_jsonl_path = output_dir / "metrics.jsonl"
    log_path.write_text("", encoding="utf-8")
    metrics_jsonl_path.write_text("", encoding="utf-8")

    source_files, manifest = _resolve_sources(Path(args.data))
    feature_cols, _ = _infer_feature_metadata(source_files, manifest)
    input_dim = len(feature_cols)
    num_classes = _infer_num_classes(source_files, manifest)

    model = build_model_from_args(args=args, input_dim=input_dim, num_classes=num_classes, teacher=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)

    train_history: List[Dict[str, float]] = []
    best_val = math.inf
    start_epoch = 0
    stale_epochs = 0
    best_path = output_dir / "best_model.pt"

    if args.init_checkpoint:
        load_checkpoint_compatible(
            model=model,
            checkpoint_path=Path(args.init_checkpoint),
            device=device,
        )

    if args.resume_checkpoint:
        start_epoch, best_val, train_history = resume_training_state(
            checkpoint_path=Path(args.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"[Resume] start_epoch={start_epoch} best_val={best_val:.6f} from {args.resume_checkpoint}")

    run_config = {
        "args": vars(args),
        "feature_cols": feature_cols,
        "device": str(device),
        "model_config": asdict(model.config),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_message(log_path, f"train start data={args.data} output_dir={output_dir}")
    _log_message(log_path, f"device={device} num_sources={len(source_files)} input_dim={input_dim} num_classes={num_classes}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=model,
            source_files=source_files,
            split="train",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=optimizer,
            reg_loss_weight=args.reg_loss_weight,
            max_grad_norm=args.max_grad_norm,
            teacher_model=None,
            distill_max_lambda=0.0,
            distill_min_lambda=0.0,
            epoch=epoch,
            total_epochs=args.epochs,
            kl_weight=0.0,
            freq_weight=0.0,
            wavelet_levels=1,
            distill_temperature=1.0,
            band_weights=None,
        )
        val_metrics = run_epoch(
            model=model,
            source_files=source_files,
            split="val",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            optimizer=None,
            reg_loss_weight=args.reg_loss_weight,
            max_grad_norm=args.max_grad_norm,
            teacher_model=None,
            distill_max_lambda=0.0,
            distill_min_lambda=0.0,
            epoch=epoch,
            total_epochs=args.epochs,
            kl_weight=0.0,
            freq_weight=0.0,
            wavelet_levels=1,
            distill_temperature=1.0,
            band_weights=None,
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
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": asdict(model.config),
            "feature_cols": feature_cols,
            "history": train_history,
        }
        if (epoch + 1) % max(1, args.save_every) == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")

        val_score = float(val_metrics["loss"])
        if val_score < best_val:
            best_val = val_score
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            if bool(int(args.save_interpret)):
                save_interpret_snapshot(
                    model=model,
                    source_files=source_files,
                    device=device,
                    output_path=output_dir / "interpret_snapshot_val.npz",
                    split="val",
                    feature_cols=feature_cols,
                )
        else:
            stale_epochs += 1

        _log_message(
            log_path,
            f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.6f} train_f1={train_metrics['f1_score']:.4f} train_acc={train_metrics['acc']:.4f} train_mcc={train_metrics['mcc']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_f1={val_metrics['f1_score']:.4f} val_acc={val_metrics['acc']:.4f} val_mcc={val_metrics['mcc']:.4f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
        )

        if args.early_stopping_patience > 0 and stale_epochs >= int(args.early_stopping_patience):
            _log_message(log_path, f"[EarlyStop] no improvement for {stale_epochs} epochs, stopping at epoch {epoch + 1}")
            break

    final_test = run_epoch(
        model=model,
        source_files=source_files,
        split="test",
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=None,
        reg_loss_weight=args.reg_loss_weight,
        max_grad_norm=args.max_grad_norm,
        teacher_model=None,
        distill_max_lambda=0.0,
        distill_min_lambda=0.0,
        epoch=max(args.epochs - 1, 0),
        total_epochs=args.epochs,
        kl_weight=0.0,
        freq_weight=0.0,
        wavelet_levels=1,
        distill_temperature=1.0,
        band_weights=None,
    )

    summary = {
        "best_val_loss": best_val,
        "test": final_test,
        "best_checkpoint": str(best_path),
        "history": train_history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_message(log_path, f"train done best_checkpoint={best_path}")
    _log_message(log_path, json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
