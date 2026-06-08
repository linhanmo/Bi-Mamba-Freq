"""
FRED-MD 数据预处理与因子提取（Python 复刻 fred-database_code 的 MATLAB 版本）

对应 MATLAB 文件：
- fredfactors.m (主流程)
- prepare_missing.m (按 tcode 做平稳化变换)
- remove_outliers.m (IQR outlier -> NaN)
- factors_em.m (EM + PCA + Bai & Ng(2002) 信息准则选因子数)
- mrsq.m (R2 / marginal R2)

默认读取 datasets/FRED-MD 下的某个 vintage CSV（例如 2015-04.csv），并输出到 data/fred_md。
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_fred_md_header(csv_fp: Path) -> Tuple[List[str], np.ndarray]:
    with csv_fp.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        tline = f.readline().rstrip("\n")

    header_parts = [h.strip() for h in header.split(",")]
    if len(header_parts) < 2:
        raise ValueError("FRED-MD CSV header 行格式不正确")
    series = [s for s in header_parts[1:] if s != ""]

    tcode_parts = [h.strip() for h in tline.split(",")]
    if not tcode_parts or not tcode_parts[0].lower().startswith("transform"):
        raise ValueError("FRED-MD CSV 第2行应为 Transform: 行")
    tcode_raw = [s for s in tcode_parts[1:] if s != ""]
    tcode = pd.to_numeric(pd.Series(tcode_raw), errors="coerce").to_numpy(dtype=np.int64)
    if np.isnan(tcode.astype(np.float64)).any():
        raise ValueError("Transform: 行存在无法解析的 tcode")
    if len(tcode) != len(series):
        raise ValueError("tcode 数量与 series 数量不一致")
    return series, tcode


def _load_rawdata(csv_fp: Path) -> Tuple[pd.DatetimeIndex, List[str], np.ndarray, np.ndarray]:
    series, tcode = _parse_fred_md_header(csv_fp)

    col_date = "sasdate"
    expected_cols = [col_date] + series
    expected_len = len(expected_cols)
    rows: List[List[str]] = []
    with csv_fp.open("r", encoding="utf-8") as f:
        _ = f.readline()
        _ = f.readline()
        reader = csv.reader(f, delimiter=",")
        for fields in reader:
            if not fields:
                continue
            if len(fields) > expected_len:
                fields = fields[:expected_len]
            elif len(fields) < expected_len:
                fields = fields + [""] * (expected_len - len(fields))
            rows.append(fields)
    df = pd.DataFrame(rows, columns=expected_cols)
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date]).sort_values(col_date)
    df = df.set_index(col_date)

    raw = df[series].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    final_date = df.index.max()
    if pd.isna(final_date):
        raise ValueError("无法确定 final_date")
    final_date = pd.Timestamp(final_date.year, final_date.month, 1)

    dates = pd.date_range(start="1959-01-01", end=final_date, freq="MS")
    T = len(dates)
    if raw.shape[0] < T:
        raise ValueError(f"rawdata 行数不足: raw={raw.shape[0]} T={T}")
    raw = raw[:T, :]
    return dates, series, tcode, raw


def prepare_missing(rawdata: np.ndarray, tcode: np.ndarray) -> np.ndarray:
    yt = np.empty_like(rawdata, dtype=np.float64)
    for i in range(rawdata.shape[1]):
        yt[:, i] = _transxf(rawdata[:, i], int(tcode[i]))
    return yt


def _transxf(x: np.ndarray, tcode: int) -> np.ndarray:
    n = int(x.shape[0])
    small = 1e-6
    y = np.full((n,), np.nan, dtype=np.float64)

    if tcode == 1:
        return x.astype(np.float64, copy=True)

    if tcode == 2:
        y[1:] = x[1:] - x[:-1]
        return y

    if tcode == 3:
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
        return y

    if tcode == 4:
        min_val = np.min(x)
        if not np.isnan(min_val) and min_val < small:
            return np.full((n,), np.nan, dtype=np.float64)
        return np.log(x)

    if tcode == 5:
        min_val = np.min(x)
        if (not np.isnan(min_val)) and (min_val <= small):
            return y
        lx = np.log(x)
        y[1:] = lx[1:] - lx[:-1]
        return y

    if tcode == 6:
        min_val = np.min(x)
        if (not np.isnan(min_val)) and (min_val <= small):
            return y
        lx = np.log(x)
        y[2:] = lx[2:] - 2 * lx[1:-1] + lx[:-2]
        return y

    if tcode == 7:
        y1 = np.full((n,), np.nan, dtype=np.float64)
        denom = x[:-1]
        y1[1:] = (x[1:] - x[:-1]) / denom
        y[2:] = y1[2:] - y1[1:-1]
        return y

    raise ValueError(f"Unsupported tcode: {tcode}")


def remove_outliers(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    median_X = np.nanmedian(X, axis=0)
    Q = np.nanpercentile(X, [25, 50, 75], axis=0)
    IQR = Q[2, :] - Q[0, :]

    Z = np.abs(X - median_X[None, :])
    outlier = Z > (10.0 * IQR[None, :])

    Y = X.copy()
    Y[outlier] = np.nan
    n = outlier.sum(axis=0).astype(np.int64)
    return Y, n


def _transform_data(x2: np.ndarray, demean: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, N = x2.shape
    if demean == 0:
        mut = np.zeros((T, N), dtype=np.float64)
        sdt = np.ones((T, N), dtype=np.float64)
        return x2.copy(), mut, sdt

    if demean == 1:
        mu = x2.mean(axis=0)
        mut = np.tile(mu, (T, 1))
        sdt = np.ones((T, N), dtype=np.float64)
        return x2 - mut, mut, sdt

    if demean == 2:
        mu = x2.mean(axis=0)
        sd = x2.std(axis=0, ddof=1)
        mut = np.tile(mu, (T, 1))
        sdt = np.tile(sd, (T, 1))
        return (x2 - mut) / sdt, mut, sdt

    if demean == 3:
        mut = np.empty((T, N), dtype=np.float64)
        for t in range(T):
            mut[t, :] = x2[: t + 1, :].mean(axis=0)
        sd = x2.std(axis=0, ddof=1)
        sdt = np.tile(sd, (T, 1))
        return (x2 - mut) / sdt, mut, sdt

    raise ValueError(f"Unsupported DEMEAN: {demean}")


def _pc2(X: np.ndarray, nfac: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = X.shape[1]
    U, S, _Vt = np.linalg.svd(X.T @ X, full_matrices=True)
    lam = U[:, :nfac] * np.sqrt(N)
    fhat = X @ lam / N
    chat = fhat @ lam.T
    ss = S.copy()
    return chat, fhat, lam, ss


def _minindc(x: np.ndarray) -> np.ndarray:
    pos = np.argmin(x, axis=0)
    if x.ndim == 2:
        for i in range(x.shape[1]):
            m = x[:, i].min()
            if (x[:, i] == m).sum() > 1:
                raise ValueError("Minimum value occurs more than once.")
    return pos + 1


def _baing(X: np.ndarray, kmax: int, jj: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    T, N = X.shape
    NT = N * T
    NT1 = N + T
    ii = np.arange(1, kmax + 1, dtype=np.float64)
    GCT = float(min(N, T))

    if jj == 1:
        CT = np.log(NT / NT1) * ii * NT1 / NT
    elif jj == 2:
        CT = (NT1 / NT) * np.log(min(N, T)) * ii
    elif jj == 3:
        CT = ii * np.log(GCT) / GCT
    else:
        raise ValueError("Input jj is specified incorrectly.")

    if T < N:
        ev, eigval, _ = np.linalg.svd(X @ X.T, full_matrices=True)
        Fhat0 = np.sqrt(T) * ev
        Lambda0 = (X.T @ Fhat0) / T
    else:
        ev, eigval, _ = np.linalg.svd(X.T @ X, full_matrices=True)
        Lambda0 = np.sqrt(N) * ev
        Fhat0 = X @ Lambda0 / N

    Sigma = np.zeros((kmax + 1,), dtype=np.float64)
    IC1 = np.zeros((kmax + 1,), dtype=np.float64)

    for i in range(kmax, 0, -1):
        Fhat = Fhat0[:, :i]
        lam = Lambda0[:, :i]
        chat = Fhat @ lam.T
        ehat = X - chat
        Sigma[i - 1] = np.mean(np.sum((ehat * ehat) / T, axis=0))
        IC1[i - 1] = np.log(Sigma[i - 1]) + CT[i - 1]

    Sigma[kmax] = np.mean(np.sum((X * X) / T, axis=0))
    IC1[kmax] = np.log(Sigma[kmax])

    ic1 = int(_minindc(IC1.reshape(-1, 1))[0])
    if ic1 > kmax:
        ic1 = 0

    Fhat_k = Fhat0[:, :kmax]
    Lambda_k = Lambda0[:, :kmax]
    chat_k = Fhat_k @ Lambda_k.T
    eigval_vec = np.diag(eigval) if eigval.ndim == 2 else eigval
    return ic1, chat_k, Fhat_k, eigval_vec


def factors_em(x: np.ndarray, kmax: int, jj: int, demean: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (np.isnan(x).sum(axis=1) == x.shape[1]).any():
        raise ValueError("Input x contains entire row of missing values.")
    if (np.isnan(x).sum(axis=0) == x.shape[0]).any():
        raise ValueError("Input x contains entire column of missing values.")
    if not ((1 <= kmax <= x.shape[1] and int(kmax) == kmax) or kmax == 99):
        raise ValueError("Input kmax is specified incorrectly.")
    if jj not in (1, 2, 3):
        raise ValueError("Input jj is specified incorrectly.")
    if demean not in (0, 1, 2, 3):
        raise ValueError("Input DEMEAN is specified incorrectly.")

    maxit = 50
    T, N = x.shape
    err = 999.0
    it = 0
    x1 = np.isnan(x)

    mu = np.nanmean(x, axis=0)
    mut = np.tile(mu, (T, 1))

    x2 = x.copy()
    x2[x1] = mut[x1]

    x3, mut, sdt = _transform_data(x2, demean)

    if kmax != 99:
        icstar, _, _, _ = _baing(x3, kmax, jj)
    else:
        icstar = 8

    chat, Fhat, lamhat, ve2 = _pc2(x3, icstar)
    chat0 = chat.copy()

    while err > 1e-6 and it < maxit:
        it += 1

        x2 = x.copy()
        repl = chat * sdt + mut
        x2[x1] = repl[x1]

        x3, mut, sdt = _transform_data(x2, demean)

        if kmax != 99:
            icstar, _, _, _ = _baing(x3, kmax, jj)
        else:
            icstar = 8

        chat, Fhat, lamhat, ve2 = _pc2(x3, icstar)

        diff = chat - chat0
        v1 = diff.reshape(-1)
        v2 = chat0.reshape(-1)
        denom = float(v2 @ v2)
        err = float((v1 @ v1) / denom) if denom != 0 else 0.0
        chat0 = chat.copy()

    ehat = x - chat * sdt - mut
    return ehat, Fhat, lamhat, ve2, x2


def mrsq(
    Fhat: np.ndarray,
    lamhat: np.ndarray,
    ve2: np.ndarray,
    series: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[List[str]], np.ndarray]:
    N, ic = lamhat.shape
    R2 = np.full((N, ic), np.nan, dtype=np.float64)
    mR2 = np.full((N, ic), np.nan, dtype=np.float64)

    for i in range(1, ic + 1):
        pred_all = Fhat[:, :i] @ lamhat[:, :i].T
        pred_one = Fhat[:, i - 1 : i] @ lamhat[:, i - 1 : i].T
        R2[:, i - 1] = np.var(pred_all, axis=0, ddof=1)
        mR2[:, i - 1] = np.var(pred_one, axis=0, ddof=1)

    mR2_F = ve2 / ve2.sum()
    mR2_F = mR2_F[:ic].reshape(-1)
    R2_T = float(mR2_F.sum())

    ind = np.argsort(-mR2, axis=0)
    vals = np.take_along_axis(mR2, ind, axis=0)
    t10_s: List[List[str]] = []
    t10_mR2 = np.full((10, ic), np.nan, dtype=np.float64)
    for j in range(ic):
        top_idx = ind[:10, j]
        t10_s.append([series[int(i)] for i in top_idx])
        t10_mR2[:, j] = vals[:10, j]

    return R2, mR2, mR2_F, R2_T, t10_s, t10_mR2


def _pick_default_vintage(fred_md_dir: Path) -> Path:
    candidates = [p for p in fred_md_dir.glob("*.csv") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"未找到 CSV: {fred_md_dir}")

    def _key(p: Path) -> Tuple[int, int, int, str]:
        name = p.name
        m = re.match(r"^(\d{4})-(\d{2})(?:-MD)?\.csv$", name)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            return (1, y, mo, name)
        m2 = re.match(r"^FRED-MD_(\d{4})m(\d{2})\.csv$", name)
        if m2:
            y, mo = int(m2.group(1)), int(m2.group(2))
            return (0, y, mo, name)
        return (-1, 0, 0, name)

    return sorted(candidates, key=_key)[-1]


@dataclass(frozen=True)
class FredFactorsConfig:
    demean: int = 2
    jj: int = 2
    kmax: int = 8


def run_fred_factors(
    csv_in: Path,
    cfg: FredFactorsConfig,
    output_dir: Path,
) -> Path:
    dates, series, tcode, rawdata = _load_rawdata(csv_in)

    yt = prepare_missing(rawdata, tcode)
    if len(dates) < 3:
        raise ValueError("样本长度不足")
    yt = yt[2:, :]
    dates = dates[2:]

    data, outlier_n = remove_outliers(yt)

    ehat, Fhat, lamhat, ve2, x2 = factors_em(data, cfg.kmax, cfg.jj, cfg.demean)
    R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq(Fhat, lamhat, ve2, series)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_fp = output_dir / f"{csv_in.stem}_fred_md_factors.npz"
    np.savez_compressed(
        str(out_fp),
        dates=dates.to_numpy(dtype="datetime64[ns]"),
        series=np.asarray(series, dtype=object),
        tcode=tcode.astype(np.int64),
        rawdata=rawdata.astype(np.float32),
        yt=yt.astype(np.float32),
        data=data.astype(np.float32),
        outlier_n=outlier_n.astype(np.int64),
        ehat=ehat.astype(np.float32),
        Fhat=Fhat.astype(np.float32),
        lamhat=lamhat.astype(np.float32),
        ve2=ve2.astype(np.float64),
        x2=x2.astype(np.float32),
        R2=R2.astype(np.float32),
        mR2=mR2.astype(np.float32),
        mR2_F=mR2_F.astype(np.float64),
        R2_T=np.asarray([R2_T], dtype=np.float64),
        t10_s=np.asarray(t10_s, dtype=object),
        t10_mR2=t10_mR2.astype(np.float32),
        DEMEAN=np.asarray([cfg.demean], dtype=np.int64),
        jj=np.asarray([cfg.jj], dtype=np.int64),
        kmax=np.asarray([cfg.kmax], dtype=np.int64),
        csv_in=np.asarray([str(csv_in)], dtype=object),
    )
    return out_fp


def _iter_all_csv(fred_md_dir: Path) -> List[Path]:
    csvs = [p for p in fred_md_dir.rglob("*.csv") if p.is_file()]
    csvs.sort(key=lambda p: str(p.relative_to(fred_md_dir)))
    return csvs


def run_fred_factors_batch(
    fred_md_dir: Path,
    cfg: FredFactorsConfig,
    output_dir: Path,
    skip_existing: bool,
) -> None:
    csvs = _iter_all_csv(fred_md_dir)
    if not csvs:
        raise FileNotFoundError(f"未找到任何 CSV: {fred_md_dir}")

    ok = 0
    failed = 0
    skipped = 0

    for csv_fp in csvs:
        rel = csv_fp.relative_to(fred_md_dir)
        out_subdir = output_dir / rel.parent
        out_fp = out_subdir / f"{csv_fp.stem}_fred_md_factors.npz"

        if skip_existing and out_fp.exists():
            skipped += 1
            continue

        try:
            out_subdir.mkdir(parents=True, exist_ok=True)
            saved = run_fred_factors(csv_in=csv_fp, cfg=cfg, output_dir=out_subdir)
            ok += 1
            print(f"[FRED-MD] ok {rel} -> {saved}")
        except Exception as e:
            failed += 1
            print(f"[FRED-MD] failed {rel}: {type(e).__name__}: {e}")

    print(f"[FRED-MD] done ok={ok} skipped={skipped} failed={failed} total={len(csvs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fred_md_dir", type=str, default=None)
    parser.add_argument("--csv_in", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--DEMEAN", type=int, default=2)
    parser.add_argument("--jj", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--skip_existing", type=int, default=1)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    fred_md_dir = Path(args.fred_md_dir) if args.fred_md_dir else (project_root / "datasets" / "FRED-MD")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "data" / "fred_md")

    cfg = FredFactorsConfig(demean=int(args.DEMEAN), jj=int(args.jj), kmax=int(args.kmax))

    if args.csv_in:
        csv_in = Path(args.csv_in)
        if not csv_in.is_absolute():
            csv_in = fred_md_dir / csv_in
        out_fp = run_fred_factors(csv_in=csv_in, cfg=cfg, output_dir=output_dir)
        print(f"[FRED-MD] saved -> {out_fp}")
        return

    run_fred_factors_batch(
        fred_md_dir=fred_md_dir,
        cfg=cfg,
        output_dir=output_dir,
        skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()
