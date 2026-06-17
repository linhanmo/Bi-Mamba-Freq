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
    np = _try_import_numpy()
    selected = _select_finance_columns(columns)
    open_price = _column_array(values, columns, selected["open"])
    high_price = _column_array(values, columns, selected["high"])
    low_price = _column_array(values, columns, selected["low"])
    close_price = _column_array(values, columns, selected["close"])
    volume = _column_array(values, columns, selected["volume"])
    dividends = _column_array(values, columns, selected["dividends"]) if "dividends" in selected else _zeros_like(close_price)
    stock_splits = (
        _column_array(values, columns, selected["stock_splits"]) if "stock_splits" in selected else _zeros_like(close_price)
    )

    eps = 1e-8
    if np is not None and hasattr(close_price, "__array__"):
        prev_close = np.concatenate([close_price[:1], close_price[:-1]], axis=0)
        log_return = np.log(np.clip(close_price, eps, None) / np.clip(prev_close, eps, None)).astype(np.float32)
        log_return[0] = 0.0
        oc_return = ((close_price - open_price) / np.clip(open_price, eps, None)).astype(np.float32)
        hl_spread = ((high_price - low_price) / np.clip(close_price, eps, None)).astype(np.float32)
    else:
        prev_close = [close_price[0]] + close_price[:-1]
        log_return = [
            0.0 if i == 0 else math.log(max(float(close_price[i]), eps) / max(float(prev_close[i]), eps))
            for i in range(len(close_price))
        ]
        oc_return = [
            (float(close_price[i]) - float(open_price[i])) / max(float(open_price[i]), eps) for i in range(len(close_price))
        ]
        hl_spread = [
            (float(high_price[i]) - float(low_price[i])) / max(float(close_price[i]), eps) for i in range(len(close_price))
        ]

    feature_names = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
        "log_return",
        "open_close_return",
        "high_low_spread",
    ]
    feature_matrix = _stack_features(
        [open_price, high_price, low_price, close_price, volume, dividends, stock_splits, log_return, oc_return, hl_spread]
    )
    return feature_matrix, close_price, feature_names


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

    manifest: Dict[str, Any] = {
        "market": market_name,
        "market_type": market_type,
        "source_root": str(data_root),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "stride": stride,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "time_features": time_features,
        "vol_scale": vol_scale,
        "task_definition": {
            "classification": "Binary label, 1 if future close at t+pred_len is higher than the last close in the input window.",
            "volatility": "Realized volatility computed as std(log returns over the future horizon) * sqrt(vol_scale).",
        },
        "tickers": [],
        "feature_names": [],
        "split_window_counts": {"train": 0, "val": 0, "test": 0},
    }

    for csv_path in csv_files:
        dt_list, time_feat, values, columns = load_csv_time_series(csv_path, time_features=time_features)
        values = _as_numpy(values)
        feature_matrix, close_price, feature_names = _build_finance_features(values, columns)
        feature_matrix = _as_numpy(feature_matrix)
        close_price = _as_numpy(close_price)
        if time_feat is not None:
            time_feat = _as_numpy(time_feat)

        total_length = feature_matrix.shape[0] if hasattr(feature_matrix, "shape") else len(feature_matrix)
        min_required = seq_len + pred_len + 1
        if total_length < min_required:
            continue

        splits = _split_indices(total_length, train_ratio=train_ratio, val_ratio=val_ratio)
        train_start, train_end = splits["train"]
        scaler_mean, scaler_std = _compute_standardizer(feature_matrix[train_start:train_end])
        features_scaled = _apply_standardizer(feature_matrix, scaler_mean, scaler_std)

        ticker = csv_path.stem
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
        if not manifest["feature_names"]:
            manifest["feature_names"] = feature_names

    if not manifest["tickers"]:
        raise ValueError("No valid ticker series found after preprocessing")

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

    def _feature_index(self, name: str, *fallback_names: str) -> int:
        feature_names = self.manifest["feature_names"]
        candidates = (name,) + fallback_names
        for candidate in candidates:
            if candidate in feature_names:
                return feature_names.index(candidate)
        raise ValueError(f"None of the feature names {candidates} exist in manifest feature_names")

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
                "features": raw["features"],
                "close": raw["close"],
                "time_features": None if time_features.size == 0 else time_features,
                "splits": {
                    "train": tuple(int(v) for v in raw["splits_train"].tolist()),
                    "val": tuple(int(v) for v in raw["splits_val"].tolist()),
                    "test": tuple(int(v) for v in raw["splits_test"].tolist()),
                },
            }
        else:
            payload = pickle.loads(path.read_bytes())

        self._cache_file = file_path
        self._cache_payload = payload
        return payload

    def _compute_targets(self, close_history: Any, target_close: float) -> Tuple[float, float]:
        np = _try_import_numpy()
        eps = 1e-8
        anchor_close = float(close_history[-1])
        y_cls = 1.0 if float(target_close) > anchor_close else 0.0
        if len(close_history) == 0:
            return y_cls, 0.0
        if np is not None and hasattr(close_history, "__array__"):
            close_with_target = np.concatenate(
                [np.asarray(close_history, dtype=np.float32), np.asarray([target_close], dtype=np.float32)], axis=0
            )
            close_with_target = np.clip(close_with_target, eps, None)
            log_returns = np.diff(np.log(close_with_target))
            if log_returns.shape[0] > self.vol_window:
                log_returns = log_returns[-self.vol_window :]
            y_vol = float(math.sqrt(self.vol_scale * float((log_returns**2).sum())))
        else:
            log_returns = [
                math.log(max(float(close_history[i]), eps) / max(float(close_history[i - 1]), eps))
                for i in range(1, len(close_history))
            ]
            log_returns.append(math.log(max(float(target_close), eps) / max(float(close_history[-1]), eps)))
            if len(log_returns) > self.vol_window:
                log_returns = log_returns[-self.vol_window :]
            y_vol = math.sqrt(self.vol_scale * sum(v * v for v in log_returns))
        return y_cls, y_vol

    def _trailing_ohlcv(self, ohlcv: Any, window: int):
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
            open_v = float(seg[0][0])
            high_v = max(float(row[1]) for row in seg)
            low_v = min(float(row[2]) for row in seg)
            close_v = float(seg[-1][3])
            vol_v = sum(float(row[4]) for row in seg)
            out.append([open_v, high_v, low_v, close_v, vol_v])
        return out

    def _build_model_input(self, payload: Dict[str, Any], start: int):
        features = payload["features"]
        o = self._feature_index("open")
        h = self._feature_index("high")
        l = self._feature_index("low")
        c = self._feature_index("close")
        v = self._feature_index("volume", "log_volume")
        ohlcv = features[:, [o, h, l, c, v]] if hasattr(features, "__array__") else [
            [row[o], row[h], row[l], row[c], row[v]] for row in features
        ]
        x_day = ohlcv[start : start + self.seq_len]
        if not self.use_multiscale_ohlcv:
            return x_day

        week_all = self._trailing_ohlcv(ohlcv, self.week_length)
        month_all = self._trailing_ohlcv(ohlcv, self.month_length)
        x_week = week_all[start : start + self.seq_len]
        x_month = month_all[start : start + self.seq_len]

        np = _try_import_numpy()
        if np is not None and hasattr(x_day, "__array__"):
            return np.concatenate([x_day, x_week, x_month], axis=1).astype(np.float32)
        return [list(x_day[i]) + list(x_week[i]) + list(x_month[i]) for i in range(len(x_day))]

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
        features = self._build_model_input(payload, start)
        target_index = start + self.seq_len
        close_history_start = max(0, target_index - self.vol_window)
        close_history = payload["close"][close_history_start:target_index]
        target_close = float(payload["close"][target_index])
        y_cls, y_vol = self._compute_targets(close_history, target_close)

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
