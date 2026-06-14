import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_npz(fp: Path) -> Dict[str, np.ndarray]:
    with np.load(fp, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _signed_std_labels(target: np.ndarray, horizon: int, vol_window: int, vol_k: float) -> Tuple[np.ndarray, np.ndarray]:
    diff_1 = np.diff(target, prepend=np.nan)
    rolling_vol = np.full_like(target, np.nan, dtype=np.float32)
    for i in range(vol_window - 1, len(target)):
        rolling_vol[i] = float(np.nanstd(diff_1[i - vol_window + 1 : i + 1], ddof=0))

    future_delta = np.full_like(target, np.nan, dtype=np.float32)
    future_vol = np.full_like(target, np.nan, dtype=np.float32)
    for i in range(len(target) - horizon):
        future_delta[i] = float(target[i + horizon] - target[i])
        future_slice = np.diff(target[i : i + horizon + 1])
        future_vol[i] = float(np.sqrt(np.mean(np.square(future_slice)))) if len(future_slice) > 0 else np.nan

    labels = np.zeros_like(target, dtype=np.int64)
    threshold = rolling_vol * float(vol_k)
    labels[future_delta > threshold] = 1
    labels[future_delta < -threshold] = -1
    return labels, future_vol


def _create_sequences(data: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys_cls: List[int] = []
    ys_reg: List[float] = []
    for i in range(len(data) - seq_len + 1):
        tgt_idx = i + seq_len - 1
        if np.isnan(y_reg[tgt_idx]):
            continue
        xs.append(data[i : i + seq_len])
        ys_cls.append(int(y_cls[tgt_idx]))
        ys_reg.append(float(y_reg[tgt_idx]))
    if not xs:
        feat_dim = data.shape[1] if data.ndim == 2 else 0
        return (
            np.zeros((0, seq_len, feat_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys_cls, dtype=np.int64),
        np.asarray(ys_reg, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one FRED-MD factors npz into a standard stage-1 training npz.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--feature_source", type=str, default="Fhat", choices=["Fhat", "x2", "data"])
    parser.add_argument("--target_index", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--vol_window", type=int, default=12)
    parser.add_argument("--vol_k", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    arrays = _load_npz(Path(args.input_npz))
    feature_mat = np.asarray(arrays[args.feature_source], dtype=np.float32)
    if feature_mat.ndim != 2:
        raise ValueError(f"{args.feature_source} 需要是二维矩阵")
    target_index = int(args.target_index)
    if target_index < 0 or target_index >= feature_mat.shape[1]:
        raise ValueError("target_index 超出范围")

    labels_cls, labels_reg = _signed_std_labels(
        target=feature_mat[:, target_index].astype(np.float32),
        horizon=int(args.horizon),
        vol_window=int(args.vol_window),
        vol_k=float(args.vol_k),
    )

    n = feature_mat.shape[0]
    train_end = int(n * float(args.train_ratio))
    val_end = int(n * float(args.train_ratio + args.val_ratio))

    train_raw = feature_mat[:train_end]
    val_raw = feature_mat[train_end:val_end]
    test_raw = feature_mat[val_end:]
    y_train_raw = labels_cls[:train_end]
    y_val_raw = labels_cls[train_end:val_end]
    y_test_raw = labels_cls[val_end:]
    r_train_raw = labels_reg[:train_end]
    r_val_raw = labels_reg[train_end:val_end]
    r_test_raw = labels_reg[val_end:]

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    train_raw = (train_raw - mean) / std
    val_raw = (val_raw - mean) / std
    test_raw = (test_raw - mean) / std

    X_train, y_train_cls, y_train_reg = _create_sequences(train_raw, y_train_raw, r_train_raw, seq_len=int(args.seq_len))
    X_val, y_val_cls, y_val_reg = _create_sequences(val_raw, y_val_raw, r_val_raw, seq_len=int(args.seq_len))
    X_test, y_test_cls, y_test_reg = _create_sequences(test_raw, y_test_raw, r_test_raw, seq_len=int(args.seq_len))

    feature_cols = np.asarray([f"{args.feature_source}_{i:03d}" for i in range(feature_mat.shape[1])], dtype=object)
    meta = {
        "dataset": "fred_md",
        "source_npz": str(args.input_npz),
        "feature_source": args.feature_source,
        "target_index": target_index,
        "seq_len": int(args.seq_len),
        "horizon": int(args.horizon),
        "num_classes": 3,
        "label_rule": "future_delta vs rolling_std(diff) * vol_k",
    }

    out_fp = Path(args.output_npz)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_fp),
        X_train=X_train,
        y_train_cls=y_train_cls,
        y_train_reg=y_train_reg,
        X_val=X_val,
        y_val_cls=y_val_cls,
        y_val_reg=y_val_reg,
        X_test=X_test,
        y_test_cls=y_test_cls,
        y_test_reg=y_test_reg,
        feature_cols=feature_cols,
        num_classes=np.asarray([3], dtype=np.int64),
        metadata=np.asarray([json.dumps(meta, ensure_ascii=False)], dtype=object),
    )
    print(f"[FRED-MD Stage1] saved -> {out_fp}")
    print(f"[FRED-MD Stage1] train={len(X_train)} val={len(X_val)} test={len(X_test)}")


if __name__ == "__main__":
    main()
