import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_npz(fp: Path) -> Dict[str, np.ndarray]:
    with np.load(fp, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _collect_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted([p for p in split_dir.glob("*.npz") if p.is_file()])
    xs: List[np.ndarray] = []
    y_cls_list: List[np.ndarray] = []
    y_reg_list: List[np.ndarray] = []

    for fp in files:
        arrays = _load_npz(fp)
        prices = np.asarray(arrays["prices"], dtype=np.float32)
        ys = np.asarray(arrays["ys"], dtype=np.float32)
        main_mv = np.asarray(arrays["main_mv_percent"], dtype=np.float32)
        if prices.ndim != 3 or ys.ndim != 3 or len(prices) == 0:
            continue
        cls = ys[:, -1, :].argmax(axis=-1).astype(np.int64)
        reg = np.abs(main_mv).astype(np.float32)
        xs.append(prices)
        y_cls_list.append(cls)
        y_reg_list.append(reg)

    if not xs:
        return (
            np.zeros((0, 5, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(y_cls_list, axis=0).astype(np.int64),
        np.concatenate(y_reg_list, axis=0).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert StockNet per-ticker split npz files into a unified stage-1 training npz.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory like data/stocknet_nasdaq100")
    parser.add_argument("--output_npz", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train_cls, y_train_reg = _collect_split(input_dir / "train")
    X_val, y_val_cls, y_val_reg = _collect_split(input_dir / "dev")
    X_test, y_test_cls, y_test_reg = _collect_split(input_dir / "test")

    feature_cols = np.asarray(["high", "low", "close"], dtype=object)
    meta = {
        "dataset": "stocknet_nasdaq100",
        "num_classes": 2,
        "feature_cols": feature_cols.tolist(),
        "regression_target": "abs_main_mv_percent",
    }
    np.savez_compressed(
        str(output_npz),
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
        num_classes=np.asarray([2], dtype=np.int64),
        metadata=np.asarray([json.dumps(meta, ensure_ascii=False)], dtype=object),
    )
    print(f"[StockNet Stage1] saved -> {output_npz}")
    print(
        f"[StockNet Stage1] train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"input_shape={tuple(X_train.shape[1:]) if len(X_train) else (0, 0)}"
    )


if __name__ == "__main__":
    main()
