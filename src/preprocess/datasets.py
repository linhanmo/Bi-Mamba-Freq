import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


Split = Literal["train", "val", "test"]
OutputFormat = Literal["full", "windows"]
TimeFeatures = Literal["none", "simple"]
NormalizeMode = Literal["auto", "on", "off"]


def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def _try_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _infer_time_column(headers: Sequence[str], rows_preview: Sequence[Sequence[str]]) -> Optional[int]:
    lowered = [h.strip().lower() for h in headers]
    for key in ("date", "datetime", "time", "timestamp"):
        if key in lowered:
            return lowered.index(key)
    if not rows_preview:
        return None
    first_col = [r[0] for r in rows_preview if len(r) > 0]
    if not first_col:
        return None
    numeric_like = sum(1 for v in first_col if _is_float(v))
    if numeric_like / max(1, len(first_col)) < 0.5:
        return 0
    return None


def _parse_datetimes(values: Sequence[str]) -> List[datetime]:
    parsed: List[datetime] = []
    for v in values:
        v = v.strip()
        try:
            parsed.append(datetime.fromisoformat(v))
            continue
        except Exception:
            pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
        ):
            try:
                parsed.append(datetime.strptime(v, fmt))
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Unsupported datetime format: {v}")
    return parsed


def _simple_time_features(dts: Sequence[datetime]):
    rows = [
        [dt.month / 12.0, dt.day / 31.0, dt.weekday() / 7.0, dt.hour / 24.0, dt.minute / 60.0] for dt in dts
    ]
    np = _try_import_numpy()
    if np is not None:
        return np.asarray(rows, dtype=np.float32)
    return rows


@dataclass(frozen=True)
class PreprocessResult:
    root: Path
    series_path: Path
    meta_path: Path


def resolve_dataset_defaults(
    dataset_name: Optional[str],
    pred_len: int = 96,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    time_features: Optional[TimeFeatures] = None,
    normalize: NormalizeMode = "auto",
) -> Dict[str, Any]:
    name = (dataset_name or "").lower()
    is_ett = name.startswith("etth") or name.startswith("ettm")
    return {
        "pred_len": pred_len,
        "train_ratio": 0.6 if train_ratio is None and is_ett else (0.7 if train_ratio is None else train_ratio),
        "val_ratio": 0.2 if val_ratio is None and is_ett else (0.1 if val_ratio is None else val_ratio),
        "time_features": "none" if time_features is None else time_features,
        "normalize": normalize,
        "split_scheme": "6:2:2" if is_ett else "7:1:2",
        "is_ett": is_ett,
    }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_standardizer(train_values):
    np = _try_import_numpy()
    if np is not None and hasattr(train_values, "mean"):
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0, ddof=0)
        std = np.where(std > 0, std, np.ones_like(std))
        return mean.astype(np.float32), std.astype(np.float32)

    n = len(train_values)
    if n == 0:
        raise ValueError("Empty training slice")
    d = len(train_values[0])
    sums = [0.0] * d
    sq_sums = [0.0] * d
    for row in train_values:
        for j, v in enumerate(row):
            fv = float(v)
            sums[j] += fv
            sq_sums[j] += fv * fv
    mean = [s / n for s in sums]
    var = [max(0.0, (sq_sums[j] / n) - (mean[j] * mean[j])) for j in range(d)]
    std = [(v**0.5 if v > 0 else 1.0) for v in var]
    return mean, std


def _series_statistics(values) -> Tuple[float, float, float]:
    np = _try_import_numpy()
    if np is not None and hasattr(values, "__array__"):
        mean_abs = float(np.abs(values.mean(axis=0)).mean())
        std_mean = float(values.std(axis=0, ddof=0).mean())
        max_abs = float(np.abs(values).max())
        return mean_abs, std_mean, max_abs

    flat = [[float(v) for v in row] for row in values]
    d = len(flat[0])
    n = len(flat)
    means = [sum(row[j] for row in flat) / n for j in range(d)]
    vars_ = [sum((row[j] - means[j]) ** 2 for row in flat) / n for j in range(d)]
    mean_abs = sum(abs(v) for v in means) / d
    std_mean = sum(v**0.5 for v in vars_) / d
    max_abs = max(abs(v) for row in flat for v in row)
    return float(mean_abs), float(std_mean), float(max_abs)


def _looks_already_standardized(values) -> bool:
    mean_abs, std_mean, max_abs = _series_statistics(values)
    return mean_abs < 0.75 and 0.5 <= std_mean <= 2.5 and max_abs <= 8.0


def _apply_standardizer(values, mean, std):
    np = _try_import_numpy()
    if np is not None and hasattr(values, "__array__"):
        return ((values - mean) / std).astype(np.float32)
    out: List[List[float]] = []
    for row in values:
        out.append([(float(v) - float(mean[j])) / float(std[j]) for j, v in enumerate(row)])
    return out


def load_csv_time_series(
    csv_path: Path,
    time_features: TimeFeatures,
) -> Tuple[Optional[List[datetime]], Any, Any, List[str]]:
    pd = _try_import_pandas()
    if pd is not None:
        np = _try_import_numpy()
        df = pd.read_csv(csv_path)
        headers = list(df.columns)
        preview_rows = df.head(20).astype(str).values.tolist()
        time_col = _infer_time_column(headers, preview_rows)
        dt_list: Optional[List[datetime]] = None
        feats = None
        if time_col is not None:
            time_name = headers[time_col]
            dt_series = pd.to_datetime(df[time_name])
            dt_list = [v.to_pydatetime() for v in dt_series.to_list()]
            df = df.drop(columns=[time_name])
            headers = [h for h in headers if h != time_name]
            if time_features == "simple":
                feats = _simple_time_features(dt_list)
        if np is not None:
            values = df.to_numpy(dtype=np.float32, copy=False)
        else:
            values = [[float(v) for v in row] for row in df.values.tolist()]
        return dt_list, feats, values, headers

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows_preview: List[List[str]] = []
        rows_all: List[List[str]] = []
        for i, row in enumerate(reader):
            if not row:
                continue
            if i < 20:
                rows_preview.append(row)
            rows_all.append(row)

    time_col = _infer_time_column(headers, rows_preview)
    dt_list = None
    feats = None
    if time_col is not None:
        dt_raw = [r[time_col] for r in rows_all]
        dt_list = _parse_datetimes(dt_raw)
        if time_features == "simple":
            feats = _simple_time_features(dt_list)
        headers = [h for i, h in enumerate(headers) if i != time_col]
        rows_all = [[v for i, v in enumerate(r) if i != time_col] for r in rows_all]

    np = _try_import_numpy()
    matrix = [[float(v) for v in r] for r in rows_all]
    values = np.asarray(matrix, dtype=np.float32) if np is not None else matrix
    return dt_list, feats, values, headers


def _split_indices(total_length: int, train_ratio: float, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("Invalid split ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_length),
    }


def _window_start_positions(
    start: int,
    end: int,
    seq_len: int,
    pred_len: int,
    stride: int,
) -> List[int]:
    last_start = end - (seq_len + pred_len)
    if last_start < start:
        return []
    return list(range(start, last_start + 1, stride))


def preprocess_csv_dataset(
    csv_path: Path,
    out_dir: Path,
    seq_len: int = 96,
    pred_len: int = 96,
    stride: int = 1,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    time_features: Optional[TimeFeatures] = None,
    output_format: OutputFormat = "full",
    dataset_name: Optional[str] = None,
    normalize: NormalizeMode = "auto",
) -> PreprocessResult:
    _ensure_dir(out_dir)
    resolved = resolve_dataset_defaults(
        dataset_name=dataset_name or csv_path.stem,
        pred_len=pred_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        time_features=time_features,
        normalize=normalize,
    )
    pred_len = int(resolved["pred_len"])
    train_ratio = float(resolved["train_ratio"])
    val_ratio = float(resolved["val_ratio"])
    time_features = resolved["time_features"]
    normalize = resolved["normalize"]
    dt_list, feats, values, columns = load_csv_time_series(csv_path, time_features=time_features)
    np = _try_import_numpy()
    if np is not None and hasattr(values, "ndim"):
        if values.ndim != 2:
            raise ValueError("Expected 2D time series array")
        total_length, n_features = values.shape
    else:
        if not values or not isinstance(values, list) or not isinstance(values[0], list):
            raise ValueError("Expected 2D time series array")
        total_length, n_features = len(values), len(values[0])
    if total_length < seq_len + pred_len + 2:
        raise ValueError("Series too short for requested window sizes")

    splits = _split_indices(total_length, train_ratio=train_ratio, val_ratio=val_ratio)
    train_start, train_end = splits["train"]
    train_values = values[train_start:train_end]
    skip_standardization = normalize == "off" or (normalize == "auto" and _looks_already_standardized(values))
    if skip_standardization:
        if np is not None and hasattr(values, "__array__"):
            mean = np.zeros(n_features, dtype=np.float32)
            std = np.ones(n_features, dtype=np.float32)
        else:
            mean = [0.0] * n_features
            std = [1.0] * n_features
        values_scaled = values.astype(np.float32) if np is not None and hasattr(values, "__array__") else values
        normalization_applied = "skipped"
    else:
        mean, std = _compute_standardizer(train_values)
        values_scaled = _apply_standardizer(values, mean, std)
        normalization_applied = "zscore"

    meta: Dict[str, object] = {
        "dataset_name": dataset_name or csv_path.stem,
        "source_csv": str(csv_path),
        "columns": columns,
        "total_length": int(total_length),
        "n_features": int(n_features),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_scheme": resolved["split_scheme"],
        "splits": {k: [int(a), int(b)] for k, (a, b) in splits.items()},
        "time_features": time_features,
        "output_format": output_format,
        "normalize": normalize,
        "normalization_applied": normalization_applied,
        "already_standardized_detected": bool(skip_standardization and normalize == "auto"),
        "scaler": {
            "mean": mean.tolist() if hasattr(mean, "tolist") else list(mean),
            "std": std.tolist() if hasattr(std, "tolist") else list(std),
        },
        "has_timestamps": dt_list is not None,
        "has_time_features": feats is not None,
    }

    if output_format == "full":
        payload = {
            "values": values_scaled,
            "time_features": feats,
            "splits": splits,
            "columns": columns,
            "scaler_mean": mean,
            "scaler_std": std,
        }
        if np is not None and hasattr(values_scaled, "__array__"):
            series_path = out_dir / "series.npz"
            np.savez_compressed(
                series_path,
                values=values_scaled,
                time_features=feats if feats is not None else np.empty((0,), dtype=np.float32),
                splits_train=np.asarray(splits["train"], dtype=np.int64),
                splits_val=np.asarray(splits["val"], dtype=np.int64),
                splits_test=np.asarray(splits["test"], dtype=np.int64),
                scaler_mean=np.asarray(mean, dtype=np.float32),
                scaler_std=np.asarray(std, dtype=np.float32),
            )
        else:
            series_path = out_dir / "series.pkl"
            series_path.write_bytes(pickle.dumps(payload, protocol=4))
    else:
        series_path = out_dir / "windows.pkl"
        windows: Dict[str, Dict[str, Any]] = {}
        for split_name, (s, e) in splits.items():
            starts = _window_start_positions(s, e, seq_len=seq_len, pred_len=pred_len, stride=stride)
            if not starts:
                windows[split_name] = {"x": [], "y": []}
                continue
            x_list = []
            y_list = []
            xf_list = []
            yf_list = []
            for st in starts:
                x_list.append(values_scaled[st : st + seq_len])
                y_list.append(values_scaled[st + seq_len : st + seq_len + pred_len])
                if feats is not None:
                    xf_list.append(feats[st : st + seq_len])
                    yf_list.append(feats[st + seq_len : st + seq_len + pred_len])
            split_payload: Dict[str, Any] = {"x": x_list, "y": y_list}
            if feats is not None:
                split_payload["x_time"] = xf_list
                split_payload["y_time"] = yf_list
            windows[split_name] = split_payload

        payload = {
            "windows": windows,
            "columns": columns,
            "scaler_mean": mean,
            "scaler_std": std,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "stride": stride,
        }
        series_path.write_bytes(pickle.dumps(payload, protocol=4))

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return PreprocessResult(root=out_dir, series_path=series_path, meta_path=meta_path)


class TimeSeriesWindowDataset:
    def __init__(
        self,
        series_pt_path: Path,
        split: Split,
        seq_len: int,
        pred_len: int,
        stride: int = 1,
        with_time_features: bool = True,
    ) -> None:
        np = _try_import_numpy()
        if series_pt_path.suffix == ".npz":
            if np is None:
                raise RuntimeError("Loading .npz requires numpy")
            data = np.load(series_pt_path, allow_pickle=False)
            values = data["values"]
            feats_raw = data["time_features"]
            feats = feats_raw if feats_raw.size != 0 else None
            splits = {
                "train": tuple(int(v) for v in data["splits_train"].tolist()),
                "val": tuple(int(v) for v in data["splits_val"].tolist()),
                "test": tuple(int(v) for v in data["splits_test"].tolist()),
            }
        else:
            payload = pickle.loads(series_pt_path.read_bytes())
            values = payload["values"]
            feats = payload.get("time_features")
            splits = payload["splits"]
        if split not in splits:
            raise ValueError(f"Unknown split: {split}")
        self.values = values
        self.time_features = feats if with_time_features else None
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.split_start, self.split_end = splits[split]
        self.starts = _window_start_positions(
            self.split_start,
            self.split_end,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            stride=self.stride,
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        st = self.starts[idx]
        x = self.values[st : st + self.seq_len]
        y = self.values[st + self.seq_len : st + self.seq_len + self.pred_len]
        torch_mod = _try_import_torch()
        if torch_mod is not None and not isinstance(x, torch_mod.Tensor):
            x = torch_mod.tensor(x, dtype=torch_mod.float32)
            y = torch_mod.tensor(y, dtype=torch_mod.float32)
        if self.time_features is None:
            return x, y
        x_time = self.time_features[st : st + self.seq_len]
        y_time = self.time_features[st + self.seq_len : st + self.seq_len + self.pred_len]
        if torch_mod is not None and not isinstance(x_time, torch_mod.Tensor):
            x_time = torch_mod.tensor(x_time, dtype=torch_mod.float32)
            y_time = torch_mod.tensor(y_time, dtype=torch_mod.float32)
        return x, y, x_time, y_time
