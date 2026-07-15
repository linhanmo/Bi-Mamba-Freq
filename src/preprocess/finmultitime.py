import bisect
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .datasets import (
    PreprocessResult,
    _apply_standardizer,
    _compute_standardizer,
    _ensure_dir,
    _split_indices,
    _try_import_numpy,
    _try_import_torch,
    _window_start_positions,
    load_csv_time_series,
)


Split = Literal["train", "val", "test"]


def _infer_market_identity(data_root: Path) -> Tuple[str, str]:
    name = data_root.name
    lowered = name.lower()
    if "hs300" in lowered:
        return "HS300", "A_share"
    if "s&p500" in lowered or "sp500" in lowered:
        return "S&P500", "US_stock"
    return name, "unknown"


def _select_finance_columns(columns: Sequence[str]) -> Dict[str, str]:
    lowered = {c.strip().lower(): c for c in columns}
    required = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    optional = {
        "dividends": "dividends",
        "stock splits": "stock_splits",
        "stock_splits": "stock_splits",
    }
    selected: Dict[str, str] = {}
    missing: List[str] = []
    for source_name, target_name in required.items():
        if source_name not in lowered:
            missing.append(source_name)
        else:
            selected[target_name] = lowered[source_name]
    for source_name, target_name in optional.items():
        if source_name in lowered and target_name not in selected:
            selected[target_name] = lowered[source_name]
    if missing:
        raise ValueError(f"Missing required finance columns: {missing}")
    return selected


def _as_numpy(values: Any):
    np = _try_import_numpy()
    if np is None:
        return values
    if hasattr(values, "__array__"):
        return np.asarray(values, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _sanitize_numeric_series(values: Any, default: float = 0.0, min_value: Optional[float] = None):
    np = _try_import_numpy()
    if np is not None and hasattr(values, "__array__"):
        arr = np.asarray(values, dtype=np.float32).copy()
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            first_valid = int(np.flatnonzero(finite_mask)[0])
            arr[:first_valid] = arr[first_valid]
            for idx in range(first_valid + 1, arr.shape[0]):
                if not np.isfinite(arr[idx]):
                    arr[idx] = arr[idx - 1]
        else:
            arr.fill(default)
        arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
        if min_value is not None:
            arr = np.clip(arr, min_value, None)
        return arr.astype(np.float32)

    cleaned: List[float] = []
    last_valid = default
    seen_valid = False
    for value in values:
        v = float(value)
        if math.isfinite(v):
            last_valid = v
            seen_valid = True
        cleaned.append(last_valid if seen_valid else default)
    if min_value is not None:
        cleaned = [max(v, min_value) for v in cleaned]
    return cleaned


def _sanitize_feature_matrix(features: Any):
    np = _try_import_numpy()
    if np is not None and hasattr(features, "__array__"):
        arr = np.asarray(features, dtype=np.float32).copy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32)
    return [[0.0 if not math.isfinite(float(v)) else float(v) for v in row] for row in features]


def _stabilize_model_features(features: Any, clip_value: float = 20.0):
    np = _try_import_numpy()
    cleaned = _sanitize_feature_matrix(features)
    if np is not None and hasattr(cleaned, "__array__"):
        arr = np.asarray(cleaned, dtype=np.float32)
        return np.clip(arr, -clip_value, clip_value).astype(np.float32)
    clipped: List[List[float]] = []
    for row in cleaned:
        clipped.append([max(-clip_value, min(clip_value, float(v))) for v in row])
    return clipped


def _sanitize_raw_ohlcv(ohlcv: Any):
    np = _try_import_numpy()
    if np is not None and hasattr(ohlcv, "__array__"):
        arr = np.asarray(ohlcv, dtype=np.float32).copy()
        for col in range(min(arr.shape[1], 4)):
            arr[:, col] = _sanitize_numeric_series(arr[:, col], default=1e-8, min_value=1e-8)
        if arr.shape[1] > 4:
            arr[:, 4] = _sanitize_numeric_series(arr[:, 4], default=0.0, min_value=0.0)
        return arr.astype(np.float32)
    columns = list(zip(*ohlcv))
    cleaned = []
    for col_idx, column in enumerate(columns):
        if col_idx < 4:
            cleaned.append(_sanitize_numeric_series(column, default=1e-8, min_value=1e-8))
        else:
            cleaned.append(_sanitize_numeric_series(column, default=0.0, min_value=0.0))
    return [list(row) for row in zip(*cleaned)]


def _column_array(values: Any, columns: Sequence[str], column_name: str):
    idx = list(columns).index(column_name)
    np = _try_import_numpy()
    if np is not None and hasattr(values, "__array__"):
        return values[:, idx].astype(np.float32)
    return [float(row[idx]) for row in values]


def _zeros_like(reference: Any):
    np = _try_import_numpy()
    if np is not None and hasattr(reference, "__array__"):
        return np.zeros_like(reference, dtype=np.float32)
    return [0.0 for _ in reference]


def _stack_features(columns: List[Any]):
    np = _try_import_numpy()
    if np is not None and all(hasattr(col, "__array__") for col in columns):
        return np.stack(columns, axis=1).astype(np.float32)
    length = len(columns[0])
    return [[float(columns[j][i]) for j in range(len(columns))] for i in range(length)]


def _build_finance_features(values: Any, columns: Sequence[str]) -> Tuple[Any, Any, List[str]]:
    selected = _select_finance_columns(columns)
    open_price = _column_array(values, columns, selected["open"])
    high_price = _column_array(values, columns, selected["high"])
    low_price = _column_array(values, columns, selected["low"])
    close_price = _column_array(values, columns, selected["close"])
    volume = _column_array(values, columns, selected["volume"])
    feature_names = ["open", "high", "low", "close", "volume"]
    feature_matrix = _stack_features([open_price, high_price, low_price, close_price, volume])
    return feature_matrix, close_price, feature_names


def _trailing_ohlcv_matrix(ohlcv: Any, window: int):
    np = _try_import_numpy()
    if np is not None and hasattr(ohlcv, "__array__"):
        out = np.empty_like(ohlcv, dtype=np.float32)
        for i in range(ohlcv.shape[0]):
            seg = ohlcv[max(0, i - window + 1) : i + 1]
            out[i, 0] = float(seg[0, 0])
            out[i, 1] = float(seg[:, 1].max())
            out[i, 2] = float(seg[:, 2].min())
            out[i, 3] = float(seg[-1, 3])
            out[i, 4] = float(seg[:, 4].sum())
        return out

    out: List[List[float]] = []
    for i in range(len(ohlcv)):
        seg = ohlcv[max(0, i - window + 1) : i + 1]
        out.append(
            [
                float(seg[0][0]),
                max(float(row[1]) for row in seg),
                min(float(row[2]) for row in seg),
                float(seg[-1][3]),
                sum(float(row[4]) for row in seg),
            ]
        )
    return out


def _concat_multiscale_ohlcv(ohlcv: Any, week_length: int = 5, month_length: int = 21):
    np = _try_import_numpy()
    weekly = _trailing_ohlcv_matrix(ohlcv, week_length)
    monthly = _trailing_ohlcv_matrix(ohlcv, month_length)
    feature_names = [
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
    if np is not None and hasattr(ohlcv, "__array__"):
        return np.concatenate([ohlcv, weekly, monthly], axis=1).astype(np.float32), feature_names
    return [list(ohlcv[i]) + list(weekly[i]) + list(monthly[i]) for i in range(len(ohlcv))], feature_names


def _rightmost_leq(sorted_values: Sequence[Any], target: Any) -> int:
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _count_windows(start: int, end: int, seq_len: int, pred_len: int, stride: int) -> int:
    return len(_window_start_positions(start, end, seq_len=seq_len, pred_len=pred_len, stride=stride))


def _save_ticker_payload(out_path: Path, payload: Dict[str, Any]) -> None:
    np = _try_import_numpy()
    if np is not None and hasattr(payload["features"], "__array__"):
        time_features = payload["time_features"]
        np.savez_compressed(
            out_path,
            features=payload["features"],
            close=payload["close"],
            time_features=time_features if time_features is not None else np.empty((0,), dtype=np.float32),
            splits_train=np.asarray(payload["splits"]["train"], dtype=np.int64),
            splits_val=np.asarray(payload["splits"]["val"], dtype=np.int64),
            splits_test=np.asarray(payload["splits"]["test"], dtype=np.int64),
            scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float32),
            scaler_std=np.asarray(payload["scaler_std"], dtype=np.float32),
        )
        return
    out_path.with_suffix(".pkl").write_bytes(pickle.dumps(payload, protocol=4))


def preprocess_finmultitime_dataset(
    data_root: Path,
    out_root: Path,
    seq_len: int = 60,
    pred_len: int = 5,
    stride: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    time_features: Literal["none", "simple"] = "simple",
    vol_scale: float = 252.0,
    max_files: Optional[int] = None,
) -> PreprocessResult:
    _ensure_dir(out_root)
    ticker_dir = out_root / "tickers"
    _ensure_dir(ticker_dir)
    market_name, market_type = _infer_market_identity(data_root)

    csv_files = sorted(data_root.glob("*.csv"))
    if max_files is not None:
        csv_files = csv_files[:max_files]
    if not csv_files:
        raise ValueError(f"No csv files found under {data_root}")

    records: List[Dict[str, Any]] = []
    calendar_dates = set()
    feature_names: List[str] = []
    for csv_path in csv_files:
        dt_list, time_feat, values, columns = load_csv_time_series(csv_path, time_features=time_features)
        values = _as_numpy(values)
        raw_ohlcv, close_price, _ = _build_finance_features(values, columns)
        raw_ohlcv = _as_numpy(raw_ohlcv)
        close_price = _as_numpy(close_price)
        raw_ohlcv = _sanitize_raw_ohlcv(raw_ohlcv)
        close_price = _sanitize_numeric_series(close_price, default=1e-8, min_value=1e-8)
        multiscale_features, multiscale_names = _concat_multiscale_ohlcv(raw_ohlcv)
        multiscale_features = _as_numpy(multiscale_features)
        multiscale_features = _sanitize_raw_ohlcv(multiscale_features)
        if time_feat is not None:
            time_feat = _as_numpy(time_feat)
            time_feat = _sanitize_feature_matrix(time_feat)
        total_length = multiscale_features.shape[0] if hasattr(multiscale_features, "shape") else len(multiscale_features)
        min_required = seq_len + pred_len + 1
        if total_length < min_required:
            continue
        if dt_list:
            calendar_dates.update(dt_list)
        if not feature_names:
            feature_names = list(multiscale_names)
        records.append(
            {
                "csv_path": csv_path,
                "ticker": csv_path.stem,
                "dt_list": dt_list,
                "time_features": time_feat,
                "features": multiscale_features,
                "close": close_price,
            }
        )

    if not records:
        raise ValueError("No valid ticker series found after preprocessing")

    calendar = sorted(calendar_dates)
    if not calendar:
        raise ValueError("FinMultitime data requires timestamps for market-wide temporal splitting")
    split_dates = _split_indices(len(calendar), train_ratio=train_ratio, val_ratio=val_ratio)
    train_date = calendar[max(split_dates["train"][1] - 1, 0)]
    val_date = calendar[max(split_dates["val"][1] - 1, 0)]

    manifest: Dict[str, Any] = {
        "market": market_name,
        "market_type": market_type,
        "source_root": str(data_root),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "stride": stride,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "split_scheme": f"{int(train_ratio*10)}:{int(val_ratio*10)}:{int((1.0-train_ratio-val_ratio)*10)}",
        "time_features": time_features,
        "vol_scale": vol_scale,
        "task_definition": {
            "classification": "Binary label, 1 if next-day log return is positive.",
            "volatility": "log1p(realized volatility) where realized volatility is sqrt(vol_scale * sum of squared log returns over the trailing vol window).",
        },
        "split_boundary_dates": {
            "train_end": train_date.isoformat(),
            "val_end": val_date.isoformat(),
        },
        "tickers": [],
        "feature_names": feature_names,
        "split_window_counts": {"train": 0, "val": 0, "test": 0},
    }

    for record in records:
        dt_list = record["dt_list"]
        time_feat = record["time_features"]
        feature_matrix = record["features"]
        close_price = record["close"]
        if dt_list:
            train_end = _rightmost_leq(dt_list, train_date)
            val_end = _rightmost_leq(dt_list, val_date)
        else:
            train_end = int(len(feature_matrix) * train_ratio)
            val_end = int(len(feature_matrix) * (train_ratio + val_ratio))
        splits = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, len(feature_matrix)),
        }
        if train_end < seq_len + 1:
            continue
        scaler_mean, scaler_std = _compute_standardizer(feature_matrix[:train_end])
        features_scaled = _apply_standardizer(feature_matrix, scaler_mean, scaler_std)
        features_scaled = _stabilize_model_features(features_scaled)

        ticker = record["ticker"]
        ticker_file = ticker_dir / f"{ticker}.npz"
        payload = {
            "features": features_scaled,
            "close": close_price,
            "time_features": time_feat,
            "splits": splits,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }
        _save_ticker_payload(ticker_file, payload)

        split_window_counts = {
            split_name: _count_windows(start, end, seq_len=seq_len, pred_len=pred_len, stride=stride)
            for split_name, (start, end) in splits.items()
        }
        for split_name, count in split_window_counts.items():
            manifest["split_window_counts"][split_name] += int(count)

        manifest["tickers"].append(
            {
                "ticker": ticker,
                "file": str((ticker_file if ticker_file.exists() else ticker_file.with_suffix(".pkl")).relative_to(out_root)),
                "length": int(total_length),
                "date_start": dt_list[0].isoformat() if dt_list else None,
                "date_end": dt_list[-1].isoformat() if dt_list else None,
                "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
                "window_counts": split_window_counts,
            }
        )

    if not manifest["tickers"]:
        raise ValueError("No valid ticker series remained after applying market-wide split boundaries")

    meta_path = out_root / "manifest.json"
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return PreprocessResult(root=out_root, series_path=out_root / "tickers", meta_path=meta_path)


class FinMultiTimeMultiTaskDataset:
    def __init__(
        self,
        processed_root: Path,
        split: Split,
        with_time_features: bool = True,
        use_multiscale_ohlcv: bool = True,
        week_length: int = 5,
        month_length: int = 21,
        vol_window: int = 30,
    ) -> None:
        self.processed_root = processed_root
        self.split = split
        self.with_time_features = with_time_features
        self.use_multiscale_ohlcv = use_multiscale_ohlcv
        self.week_length = int(week_length)
        self.month_length = int(month_length)
        self.vol_window = int(vol_window)
        self.manifest = json.loads((processed_root / "manifest.json").read_text())
        self.market = self.manifest.get("market", "unknown")
        self.market_type = self.manifest.get("market_type", "unknown")
        self.seq_len = int(self.manifest["seq_len"])
        self.pred_len = int(self.manifest["pred_len"])
        self.stride = int(self.manifest["stride"])
        self.vol_scale = float(self.manifest["vol_scale"])

        self.entries: List[Dict[str, Any]] = []
        self.cumulative_counts: List[int] = []
        total = 0
        for ticker_info in self.manifest["tickers"]:
            count = int(ticker_info["window_counts"][split])
            if count <= 0:
                continue
            self.entries.append(ticker_info)
            total += count
            self.cumulative_counts.append(total)

        self._cache_file: Optional[str] = None
        self._cache_payload: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def _load_payload(self, file_path: str) -> Dict[str, Any]:
        if self._cache_file == file_path and self._cache_payload is not None:
            return self._cache_payload

        path = Path(file_path)
        if not path.is_absolute():
            path = self.processed_root / path
        np = _try_import_numpy()
        if path.suffix == ".npz":
            if np is None:
                raise RuntimeError("Loading .npz requires numpy")
            raw = np.load(path, allow_pickle=False)
            time_features = raw["time_features"]
            payload = {
                "features": _stabilize_model_features(raw["features"]),
                "close": _sanitize_numeric_series(raw["close"], default=1e-8, min_value=1e-8),
                "time_features": None if time_features.size == 0 else _sanitize_feature_matrix(time_features),
                "splits": {
                    "train": tuple(int(v) for v in raw["splits_train"].tolist()),
                    "val": tuple(int(v) for v in raw["splits_val"].tolist()),
                    "test": tuple(int(v) for v in raw["splits_test"].tolist()),
                },
            }
        else:
            payload = pickle.loads(path.read_bytes())
            payload["features"] = _stabilize_model_features(payload["features"])
            payload["close"] = _sanitize_numeric_series(payload["close"], default=1e-8, min_value=1e-8)
            if payload.get("time_features") is not None:
                payload["time_features"] = _sanitize_feature_matrix(payload["time_features"])

        self._cache_file = file_path
        self._cache_payload = payload
        return payload

    def _compute_targets(self, close_history: Any, current_close: float, target_close: float) -> Tuple[float, float]:
        np = _try_import_numpy()
        eps = 1e-8
        current_close = float(current_close)
        target_close = float(target_close)
        if (not math.isfinite(current_close)) or current_close <= 0.0:
            current_close = eps
        if (not math.isfinite(target_close)) or target_close <= 0.0:
            target_close = eps
        y_cls = 1.0 if math.log(target_close / current_close) > 0.0 else 0.0
        if len(close_history) == 0:
            return y_cls, 0.0
        if np is not None and hasattr(close_history, "__array__"):
            close_with_target = np.concatenate(
                [np.asarray(close_history, dtype=np.float32), np.asarray([current_close, target_close], dtype=np.float32)], axis=0
            )
            close_with_target = np.nan_to_num(close_with_target, nan=eps, posinf=eps, neginf=eps)
            close_with_target = np.clip(close_with_target, eps, None)
            log_returns = np.diff(np.log(close_with_target))
            if log_returns.shape[0] > self.vol_window:
                log_returns = log_returns[-self.vol_window :]
            y_vol = float(math.log1p(math.sqrt(self.vol_scale * float((log_returns**2).sum()))))
        else:
            cleaned = []
            for value in close_history:
                v = float(value)
                if (not math.isfinite(v)) or v <= 0.0:
                    v = eps
                cleaned.append(v)
            log_returns = [math.log(cleaned[i] / cleaned[i - 1]) for i in range(1, len(cleaned))]
            log_returns.append(math.log(current_close / cleaned[-1]))
            log_returns.append(math.log(target_close / current_close))
            if len(log_returns) > self.vol_window:
                log_returns = log_returns[-self.vol_window :]
            y_vol = math.log1p(math.sqrt(self.vol_scale * sum(v * v for v in log_returns)))
        return y_cls, y_vol

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        entry_idx = bisect.bisect_right(self.cumulative_counts, idx)
        prev_total = 0 if entry_idx == 0 else self.cumulative_counts[entry_idx - 1]
        local_idx = idx - prev_total
        ticker_info = self.entries[entry_idx]
        split_start, _ = ticker_info["splits"][self.split]
        start = split_start + local_idx * self.stride

        payload = self._load_payload(ticker_info["file"])
        features = payload["features"][start : start + self.seq_len]
        target_index = start + self.seq_len
        close_history_start = max(0, target_index - self.vol_window - 1)
        close_history = payload["close"][close_history_start:target_index]
        current_close = float(payload["close"][target_index - 1])
        target_close = float(payload["close"][target_index])
        y_cls, y_vol = self._compute_targets(close_history, current_close, target_close)

        sample: Dict[str, Any] = {
            "x": features,
            "y_cls": y_cls,
            "y_vol": y_vol,
            "ticker": ticker_info["ticker"],
            "start": start,
            "market": self.market,
            "market_type": self.market_type,
        }
        if self.with_time_features and payload.get("time_features") is not None:
            sample["x_time"] = payload["time_features"][start : start + self.seq_len]

        torch_mod = _try_import_torch()
        if torch_mod is not None:
            sample["x"] = torch_mod.tensor(sample["x"], dtype=torch_mod.float32)
            sample["y_cls"] = torch_mod.tensor([sample["y_cls"]], dtype=torch_mod.float32)
            sample["y_vol"] = torch_mod.tensor([sample["y_vol"]], dtype=torch_mod.float32)
            if "x_time" in sample:
                sample["x_time"] = torch_mod.tensor(sample["x_time"], dtype=torch_mod.float32)
        return sample
