"""
拐点标注（Turning Points Annotation）

实现：基于 ZigZag 思路的“确认式”拐点标注（峰/谷），支持固定阈值或基于滚动波动率的动态阈值。
默认配置面向鲁棒性测试：动态阈值（滚动波动率 * k），并默认对 CSI300 与 S&P500 输出 CSV+NPZ（不会影响原有预处理 npz）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZigZagConfig:
    threshold: float = 0.02
    dynamic_threshold: bool = True
    vol_window: int = 20
    vol_k: float = 2.0
    min_threshold: float = 0.002


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(np.float32)


def _compute_thresholds(
    prices: np.ndarray,
    cfg: ZigZagConfig,
) -> np.ndarray:
    if not cfg.dynamic_threshold:
        return np.full_like(prices, fill_value=float(cfg.threshold), dtype=np.float32)

    px = pd.Series(prices.astype(np.float64))
    ret = px.pct_change()
    vol = ret.rolling(cfg.vol_window).std(ddof=0)
    thr = (vol * float(cfg.vol_k)).astype(np.float32)
    thr = thr.fillna(float(cfg.threshold))
    thr = thr.clip(lower=float(cfg.min_threshold))
    return thr.to_numpy(dtype=np.float32)


def zigzag_turning_points(
    prices: np.ndarray,
    thresholds: np.ndarray,
    include_ends: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    n = int(prices.shape[0])
    tp_type = np.zeros((n,), dtype=np.int8)
    pivots: List[int] = []
    if n < 3:
        return tp_type, pivots

    start_price = float(prices[0])
    high_price = start_price
    low_price = start_price
    high_idx = 0
    low_idx = 0
    trend = 0

    for i in range(1, n):
        p = float(prices[i])
        thr = float(thresholds[i])

        if p > high_price:
            high_price = p
            high_idx = i
        if p < low_price:
            low_price = p
            low_idx = i

        if trend == 0:
            up_move = (p - low_price) / max(low_price, 1e-12)
            down_move = (p - high_price) / max(high_price, 1e-12)
            if up_move >= thr:
                tp_type[low_idx] = -1
                pivots.append(low_idx)
                trend = 1
                high_price = p
                high_idx = i
                continue
            if down_move <= -thr:
                tp_type[high_idx] = 1
                pivots.append(high_idx)
                trend = -1
                low_price = p
                low_idx = i
                continue
            continue

        if trend > 0:
            if p > high_price:
                high_price = p
                high_idx = i
                continue
            drawdown = (p - high_price) / max(high_price, 1e-12)
            if drawdown <= -thr:
                tp_type[high_idx] = 1
                pivots.append(high_idx)
                trend = -1
                low_price = p
                low_idx = i
                continue
            continue

        if p < low_price:
            low_price = p
            low_idx = i
            continue
        rebound = (p - low_price) / max(low_price, 1e-12)
        if rebound >= thr:
            tp_type[low_idx] = -1
            pivots.append(low_idx)
            trend = 1
            high_price = p
            high_idx = i
            continue

    if include_ends:
        if pivots and pivots[0] != 0:
            tp_type[0] = tp_type[pivots[0]] * -1 if tp_type[pivots[0]] != 0 else 0
            pivots.insert(0, 0)
        if pivots and pivots[-1] != n - 1:
            tp_type[n - 1] = tp_type[pivots[-1]] * -1 if tp_type[pivots[-1]] != 0 else 0
            pivots.append(n - 1)

    return tp_type, pivots


def _load_series_from_csv(
    input_path: Path,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
        df.index.name = "date"
    df[price_col] = _safe_float_series(df[price_col])
    df = df.dropna(subset=[price_col])
    return df


def _write_outputs(
    df: pd.DataFrame,
    out_csv: Optional[Path],
    out_npz: Optional[Path],
    pivots: List[int],
    cfg: ZigZagConfig,
    threshold_series: np.ndarray,
) -> None:
    meta: Dict[str, object] = {
        "method": "zigzag",
        "threshold": cfg.threshold,
        "dynamic_threshold": cfg.dynamic_threshold,
        "vol_window": cfg.vol_window,
        "vol_k": cfg.vol_k,
        "min_threshold": cfg.min_threshold,
        "pivot_count": int(len(pivots)),
    }

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        df_out.to_csv(out_csv, index=True)
        (out_csv.parent / (out_csv.stem + ".meta.json")).write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_npz),
            index=df.index.to_numpy(dtype="datetime64[ns]"),
            turning_type=df["turning_type"].to_numpy(dtype=np.int8),
            turning_flag=df["turning_flag"].to_numpy(dtype=np.int8),
            price=df["price"].to_numpy(dtype=np.float32),
            threshold=threshold_series.astype(np.float32),
            pivots=np.asarray(pivots, dtype=np.int32),
            meta=json.dumps(meta, ensure_ascii=False),
        )


def _run_one(
    input_path: Path,
    date_col: str,
    price_col: str,
    cfg: ZigZagConfig,
    include_ends: bool,
    out_csv: Path,
    out_npz: Path,
) -> None:
    df = _load_series_from_csv(
        input_path=input_path,
        date_col=date_col,
        price_col=price_col,
    )
    prices = df[price_col].to_numpy(dtype=np.float32)
    thresholds = _compute_thresholds(prices, cfg)
    tp_type, pivots = zigzag_turning_points(
        prices=prices,
        thresholds=thresholds,
        include_ends=include_ends,
    )

    out_df = pd.DataFrame(index=df.index)
    out_df.index.name = "date"
    out_df["price"] = prices
    out_df["threshold"] = thresholds
    out_df["turning_type"] = tp_type
    out_df["turning_flag"] = (tp_type != 0).astype(np.int8)

    _write_outputs(
        df=out_df,
        out_csv=out_csv,
        out_npz=out_npz,
        pivots=pivots,
        cfg=cfg,
        threshold_series=thresholds,
    )

    print(f"[TurningPoints] {input_path.name} rows={len(out_df)} pivots={len(pivots)}")
    print(f"[TurningPoints] saved csv -> {out_csv}")
    print(f"[TurningPoints] saved npz -> {out_npz}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="both")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--price_col", type=str, default="Close")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_npz", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--dynamic_threshold", type=int, default=1)
    parser.add_argument("--vol_window", type=int, default=20)
    parser.add_argument("--vol_k", type=float, default=2.0)
    parser.add_argument("--min_threshold", type=float, default=0.002)
    parser.add_argument("--include_ends", type=int, default=0)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "data" / "robust_turning_points")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ZigZagConfig(
        threshold=float(args.threshold),
        dynamic_threshold=bool(args.dynamic_threshold),
        vol_window=int(args.vol_window),
        vol_k=float(args.vol_k),
        min_threshold=float(args.min_threshold),
    )
    preset = (args.preset or "").strip().lower()
    include_ends = bool(args.include_ends)

    if preset in {"both", "csi300"}:
        csi_in = project_root / "datasets" / "CSI300_2008_2025.csv"
        _run_one(
            input_path=csi_in,
            date_col="date",
            price_col="Close",
            cfg=cfg,
            include_ends=include_ends,
            out_csv=output_dir / "csi300_turning_points.csv",
            out_npz=output_dir / "csi300_turning_points.npz",
        )

    if preset in {"both", "sp500"}:
        sp_in = project_root / "datasets" / "SP500_2000_2025.csv"
        _run_one(
            input_path=sp_in,
            date_col="Date",
            price_col="Adj Close",
            cfg=cfg,
            include_ends=include_ends,
            out_csv=output_dir / "sp500_turning_points.csv",
            out_npz=output_dir / "sp500_turning_points.npz",
        )

    if preset in {"custom"}:
        if not args.input:
            raise SystemExit("--preset custom 需要提供 --input")
        input_path = Path(args.input)
        out_csv = Path(args.output_csv) if args.output_csv else (output_dir / f"{input_path.stem}_turning_points.csv")
        out_npz = Path(args.output_npz) if args.output_npz else (output_dir / f"{input_path.stem}_turning_points.npz")
        _run_one(
            input_path=input_path,
            date_col=args.date_col,
            price_col=args.price_col,
            cfg=cfg,
            include_ends=include_ends,
            out_csv=out_csv,
            out_npz=out_npz,
        )


if __name__ == "__main__":
    main()
