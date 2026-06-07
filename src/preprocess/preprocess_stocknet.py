"""
StockNet (stocknet-dataset) NASDAQ-100 预处理脚本

目标：严格遵循原论文/官方代码（yumoxu/stocknet-code）的标签与样本构造逻辑：
- 使用 price/preprocessed/*.txt 的 movement_percent 作为标签来源（不重算收益）
- 二分类标签：movement_percent <= 1e-7 => 0，否则 => 1
- 丢弃低波动样本：主预测日 movement_percent ∈ [-0.005, 0.0055)
- 使用 max_n_days=5 的交易日窗口，并按“日历日推文 -> 下一交易日”对齐
- 对齐后每个交易日要求至少 1 条推文，否则丢弃样本

输入（默认）：
- datasets/NASDAQ-100/price/preprocessed/*.txt
- datasets/NASDAQ-100/tweet/preprocessed/<TICKER>/<YYYY-MM-DD>

输出（默认）：
- data/stocknet_nasdaq100/{train,dev,test}/{TICKER}.npz
- data/stocknet_nasdaq100/vocab.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StockNetDates:
    train_start: str = "2014-01-01"
    train_end: str = "2015-08-01"
    dev_start: str = "2015-08-01"
    dev_end: str = "2015-10-01"
    test_start: str = "2015-10-01"
    test_end: str = "2016-01-01"


def _parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _in_range(d: date, start: date, end: date) -> bool:
    return start <= d < end


def _mv_class_2(mv: float) -> int:
    return 0 if mv <= 1e-7 else 1


def _mv_one_hot(mv: float, y_size: int) -> np.ndarray:
    if y_size == 2:
        idx = _mv_class_2(mv)
        y = np.zeros((2,), dtype=np.float32)
        y[idx] = 1.0
        return y
    if y_size == 3:
        threshold_1, threshold_2 = -0.004, 0.005
        if mv < threshold_1:
            idx = 0
        elif mv < threshold_2:
            idx = 1
        else:
            idx = 2
        y = np.zeros((3,), dtype=np.float32)
        y[idx] = 1.0
        return y
    raise ValueError(f"Unsupported y_size: {y_size}")


def _mv_class(mv: float, y_size: int) -> int:
    if y_size == 2:
        return _mv_class_2(mv)
    if y_size == 3:
        threshold_1, threshold_2 = -0.004, 0.005
        if mv < threshold_1:
            return 0
        if mv < threshold_2:
            return 1
        return 2
    raise ValueError(f"Unsupported y_size: {y_size}")


def _get_ss_index(word_seq: List[str], ss: str) -> int:
    ss = ss.lower()
    ss_index = len(word_seq) - 1
    if ss in word_seq:
        return word_seq.index(ss)
    if "$" not in word_seq:
        return ss_index
    dollar_index = word_seq.index("$")
    if dollar_index != len(word_seq) - 1 and ss in word_seq[dollar_index + 1]:
        return dollar_index + 1
    for idx in range(dollar_index + 1, len(word_seq)):
        if ss in word_seq[idx]:
            return idx
    return ss_index


class Vocab:
    def __init__(self):
        self.token_to_id: Dict[str, int] = {"UNK": 0}

    def get_id(self, token: str, frozen: bool) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if frozen:
            return 0
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        return idx

    def to_json_dict(self) -> Dict[str, object]:
        id_to_token = [None] * len(self.token_to_id)
        for tok, idx in self.token_to_id.items():
            id_to_token[idx] = tok
        return {"token_to_id": self.token_to_id, "id_to_token": id_to_token}


def _load_price_table(price_fp: Path) -> Tuple[List[date], Dict[date, Tuple[float, float, float, float]]]:
    by_date: Dict[date, Tuple[float, float, float, float]] = {}
    dates_list: List[date] = []
    with price_fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            d = _parse_iso_date(parts[0])
            mv = float(parts[1])
            high = float(parts[3])
            low = float(parts[4])
            close = float(parts[5])
            by_date[d] = (mv, high, low, close)
            dates_list.append(d)
    dates_list = sorted(set(dates_list))
    return dates_list, by_date


def _get_price_sample(
    trading_dates: List[date],
    by_date: Dict[date, Tuple[float, float, float, float]],
    main_target_date: date,
    max_n_days: int,
    y_size: int,
    discard_low_mv: bool,
    low_mv_min: float,
    low_mv_max: float,
) -> Optional[Dict[str, object]]:
    if main_target_date not in by_date:
        return None
    try:
        i = trading_dates.index(main_target_date)
    except ValueError:
        return None
    if i - (max_n_days - 1) < 0:
        return None
    main_mv = by_date[main_target_date][0]
    if discard_low_mv and (low_mv_min <= float(main_mv) < low_mv_max):
        return None
    ts = trading_dates[i - max_n_days + 1 : i + 1]
    price_dates = trading_dates[i - max_n_days : i]
    if len(ts) != max_n_days or len(price_dates) != max_n_days:
        return None
    ys = np.stack([_mv_one_hot(by_date[d][0], y_size) for d in ts], axis=0).astype(np.float32)
    prices = np.stack([[by_date[d][1], by_date[d][2], by_date[d][3]] for d in price_dates], axis=0).astype(np.float32)
    mv_percents = np.array([_mv_class(by_date[d][0], y_size) for d in price_dates], dtype=np.int8)
    return {
        "T": max_n_days,
        "ts": np.array(ts, dtype="datetime64[D]"),
        "ys": ys,
        "main_mv_percent": float(main_mv),
        "mv_percents": mv_percents,
        "prices": prices,
    }


def _get_unaligned_corpora(
    tweet_stock_dir: Path,
    ss: str,
    main_target_date: date,
    max_n_days: int,
    max_n_msgs: int,
    max_n_words: int,
    vocab: Vocab,
    vocab_frozen: bool,
) -> List[Tuple[date, np.ndarray, np.ndarray, np.ndarray, int]]:
    corpora: List[Tuple[date, np.ndarray, np.ndarray, np.ndarray, int]] = []
    d_d_max = main_target_date - timedelta(days=1)
    d_d_min = main_target_date - timedelta(days=max_n_days)
    d = d_d_max
    while d >= d_d_min:
        fp = tweet_stock_dir / d.isoformat()
        if fp.exists():
            word_mat = np.zeros((max_n_msgs, max_n_words), dtype=np.int32)
            n_word_vec = np.zeros((max_n_msgs,), dtype=np.int32)
            ss_index_vec = np.zeros((max_n_msgs,), dtype=np.int32)
            msg_id = 0
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    msg_dict = json.loads(line)
                    text = msg_dict.get("text", [])
                    if not text:
                        continue
                    words = [str(w) for w in text[:max_n_words]]
                    word_ids = [vocab.get_id(w, frozen=vocab_frozen) for w in words]
                    n_words = len(word_ids)
                    n_word_vec[msg_id] = n_words
                    word_mat[msg_id, :n_words] = np.asarray(word_ids, dtype=np.int32)
                    ss_index_vec[msg_id] = _get_ss_index(words, ss)
                    msg_id += 1
                    if msg_id == max_n_msgs:
                        break
            corpora.append((d, word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id], msg_id))
        d -= timedelta(days=1)
    corpora.reverse()
    return corpora


def _trading_day_alignment(
    ts: np.ndarray,
    unaligned_corpora: List[Tuple[date, np.ndarray, np.ndarray, np.ndarray, int]],
    max_n_msgs: int,
    max_n_words: int,
) -> Optional[Dict[str, np.ndarray]]:
    T = len(ts)
    aligned_word_tensor = np.zeros((T, max_n_msgs, max_n_words), dtype=np.int32)
    aligned_ss_index_mat = np.zeros((T, max_n_msgs), dtype=np.int32)
    aligned_n_words_mat = np.zeros((T, max_n_msgs), dtype=np.int32)
    aligned_n_msgs_vec = np.zeros((T,), dtype=np.int32)

    aligned_msgs: List[List[np.ndarray]] = [[] for _ in range(T)]
    aligned_ss_indices: List[List[np.ndarray]] = [[] for _ in range(T)]
    aligned_n_words: List[List[np.ndarray]] = [[] for _ in range(T)]
    aligned_n_msgs: List[List[int]] = [[] for _ in range(T)]

    ts_py = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in ts]
    corpus_t_indices: List[int] = []
    for (d, *_rest) in unaligned_corpora:
        assigned = None
        for t in range(T):
            if d < ts_py[t]:
                assigned = t
                break
        if assigned is None:
            return None
        corpus_t_indices.append(assigned)
    if len(corpus_t_indices) != len(unaligned_corpora):
        return None

    for idx, t in enumerate(corpus_t_indices):
        d, word_mat, ss_index_vec, n_word_vec, n_msgs = unaligned_corpora[idx]
        aligned_msgs[t].append(word_mat)
        aligned_ss_indices[t].append(ss_index_vec)
        aligned_n_words[t].append(n_word_vec)
        aligned_n_msgs[t].append(int(n_msgs))

    n_fails = len([1 for n_msgs in aligned_n_msgs if sum(n_msgs) == 0])
    if n_fails > 0:
        return None

    for t in range(T):
        n_msgs = sum(aligned_n_msgs[t])
        if aligned_msgs[t] and aligned_ss_indices[t] and aligned_n_words[t]:
            msgs = np.vstack(aligned_msgs[t])
            ss_indices = np.hstack(aligned_ss_indices[t])
            n_word = np.hstack(aligned_n_words[t])
            if len(msgs) != len(ss_indices) or len(msgs) != len(n_word):
                return None
            n_msgs = min(n_msgs, max_n_msgs)
            aligned_n_msgs_vec[t] = n_msgs
            aligned_word_tensor[t, :n_msgs] = msgs[:n_msgs]
            aligned_ss_index_mat[t, :n_msgs] = ss_indices[:n_msgs]
            aligned_n_words_mat[t, :n_msgs] = n_word[:n_msgs]

    return {
        "msgs": aligned_word_tensor,
        "ss_indices": aligned_ss_index_mat,
        "n_words": aligned_n_words_mat,
        "n_msgs": aligned_n_msgs_vec,
    }


def _iter_main_target_dates(
    trading_dates: List[date],
    start: date,
    end: date,
) -> Iterable[date]:
    for d in trading_dates:
        if _in_range(d, start, end):
            yield d


def _discover_symbols(price_dir: Path, tweet_dir: Path) -> List[str]:
    price_symbols = {p.stem for p in price_dir.glob("*.txt") if p.is_file()}
    tweet_symbols = {p.name for p in tweet_dir.iterdir() if p.is_dir()}
    return sorted(price_symbols & tweet_symbols)


def preprocess_one_symbol_one_phase(
    symbol: str,
    price_dir: Path,
    tweet_dir: Path,
    phase: str,
    dates_cfg: StockNetDates,
    max_n_days: int,
    max_n_msgs: int,
    max_n_words: int,
    y_size: int,
    discard_low_mv: bool,
    low_mv_min: float,
    low_mv_max: float,
    vocab: Vocab,
    vocab_frozen: bool,
) -> Dict[str, np.ndarray]:
    if phase == "train":
        start, end = _parse_iso_date(dates_cfg.train_start), _parse_iso_date(dates_cfg.train_end)
    elif phase == "dev":
        start, end = _parse_iso_date(dates_cfg.dev_start), _parse_iso_date(dates_cfg.dev_end)
    elif phase == "test":
        start, end = _parse_iso_date(dates_cfg.test_start), _parse_iso_date(dates_cfg.test_end)
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    trading_dates, by_date = _load_price_table(price_dir / f"{symbol}.txt")
    tweet_stock_dir = tweet_dir / symbol

    X_prices: List[np.ndarray] = []
    X_msgs: List[np.ndarray] = []
    X_ss_indices: List[np.ndarray] = []
    X_n_words: List[np.ndarray] = []
    X_n_msgs: List[np.ndarray] = []
    y_seq: List[np.ndarray] = []
    main_mv: List[float] = []
    mv_seq: List[np.ndarray] = []
    target_dates: List[np.datetime64] = []

    for main_target_date in _iter_main_target_dates(trading_dates, start, end):
        prices_and_ts = _get_price_sample(
            trading_dates=trading_dates,
            by_date=by_date,
            main_target_date=main_target_date,
            max_n_days=max_n_days,
            y_size=y_size,
            discard_low_mv=discard_low_mv,
            low_mv_min=low_mv_min,
            low_mv_max=low_mv_max,
        )
        if not prices_and_ts:
            continue
        unaligned_corpora = _get_unaligned_corpora(
            tweet_stock_dir=tweet_stock_dir,
            ss=symbol,
            main_target_date=main_target_date,
            max_n_days=max_n_days,
            max_n_msgs=max_n_msgs,
            max_n_words=max_n_words,
            vocab=vocab,
            vocab_frozen=vocab_frozen,
        )
        aligned_info_dict = _trading_day_alignment(
            ts=prices_and_ts["ts"],
            unaligned_corpora=unaligned_corpora,
            max_n_msgs=max_n_msgs,
            max_n_words=max_n_words,
        )
        if not aligned_info_dict:
            continue

        X_prices.append(prices_and_ts["prices"])
        mv_seq.append(prices_and_ts["mv_percents"])
        y_seq.append(prices_and_ts["ys"])
        main_mv.append(prices_and_ts["main_mv_percent"])
        X_msgs.append(aligned_info_dict["msgs"])
        X_ss_indices.append(aligned_info_dict["ss_indices"])
        X_n_words.append(aligned_info_dict["n_words"])
        X_n_msgs.append(aligned_info_dict["n_msgs"])
        target_dates.append(np.datetime64(main_target_date.isoformat(), "D"))

    if not X_prices:
        return {
            "prices": np.zeros((0, max_n_days, 3), dtype=np.float32),
            "msgs": np.zeros((0, max_n_days, max_n_msgs, max_n_words), dtype=np.int32),
            "ss_indices": np.zeros((0, max_n_days, max_n_msgs), dtype=np.int32),
            "n_words": np.zeros((0, max_n_days, max_n_msgs), dtype=np.int32),
            "n_msgs": np.zeros((0, max_n_days), dtype=np.int32),
            "ys": np.zeros((0, max_n_days, y_size), dtype=np.float32),
            "main_mv_percent": np.zeros((0,), dtype=np.float32),
            "mv_percents": np.zeros((0, max_n_days), dtype=np.int8),
            "main_target_date": np.zeros((0,), dtype="datetime64[D]"),
        }

    return {
        "prices": np.stack(X_prices, axis=0).astype(np.float32),
        "msgs": np.stack(X_msgs, axis=0).astype(np.int32),
        "ss_indices": np.stack(X_ss_indices, axis=0).astype(np.int32),
        "n_words": np.stack(X_n_words, axis=0).astype(np.int32),
        "n_msgs": np.stack(X_n_msgs, axis=0).astype(np.int32),
        "ys": np.stack(y_seq, axis=0).astype(np.float32),
        "main_mv_percent": np.asarray(main_mv, dtype=np.float32),
        "mv_percents": np.stack(mv_seq, axis=0).astype(np.int8),
        "main_target_date": np.asarray(target_dates, dtype="datetime64[D]"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--phases", type=str, default="train,dev,test")
    parser.add_argument("--max_n_days", type=int, default=5)
    parser.add_argument("--max_n_msgs", type=int, default=30)
    parser.add_argument("--max_n_words", type=int, default=40)
    parser.add_argument("--y_size", type=int, default=2)
    parser.add_argument("--discard_low_mv", type=int, default=1)
    parser.add_argument("--low_mv_min", type=float, default=-0.005)
    parser.add_argument("--low_mv_max", type=float, default=0.0055)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (project_root / "datasets" / "NASDAQ-100")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "data" / "stocknet_nasdaq100")
    output_dir.mkdir(parents=True, exist_ok=True)

    price_dir = dataset_dir / "price" / "preprocessed"
    tweet_dir = dataset_dir / "tweet" / "preprocessed"

    all_symbols = _discover_symbols(price_dir, tweet_dir)
    if args.symbols:
        requested = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbols = [s for s in all_symbols if s in set(requested)]
    else:
        symbols = all_symbols

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    dates_cfg = StockNetDates()

    vocab = Vocab()
    for phase in phases:
        phase_out_dir = output_dir / phase
        phase_out_dir.mkdir(parents=True, exist_ok=True)
        vocab_frozen = phase != "train"
        for symbol in symbols:
            arrays = preprocess_one_symbol_one_phase(
                symbol=symbol,
                price_dir=price_dir,
                tweet_dir=tweet_dir,
                phase=phase,
                dates_cfg=dates_cfg,
                max_n_days=args.max_n_days,
                max_n_msgs=args.max_n_msgs,
                max_n_words=args.max_n_words,
                y_size=args.y_size,
                discard_low_mv=bool(args.discard_low_mv),
                low_mv_min=args.low_mv_min,
                low_mv_max=args.low_mv_max,
                vocab=vocab,
                vocab_frozen=vocab_frozen,
            )
            out_fp = phase_out_dir / f"{symbol}.npz"
            np.savez_compressed(
                str(out_fp),
                prices=arrays["prices"],
                msgs=arrays["msgs"],
                ss_indices=arrays["ss_indices"],
                n_words=arrays["n_words"],
                n_msgs=arrays["n_msgs"],
                ys=arrays["ys"],
                main_mv_percent=arrays["main_mv_percent"],
                mv_percents=arrays["mv_percents"],
                main_target_date=arrays["main_target_date"],
                symbol=np.asarray([symbol], dtype=object),
                max_n_days=np.asarray([args.max_n_days], dtype=np.int32),
                max_n_msgs=np.asarray([args.max_n_msgs], dtype=np.int32),
                max_n_words=np.asarray([args.max_n_words], dtype=np.int32),
                y_size=np.asarray([args.y_size], dtype=np.int32),
            )
            print(f"[StockNet] {phase}/{symbol}: {arrays['prices'].shape[0]} samples -> {out_fp}")

    vocab_fp = output_dir / "vocab.json"
    vocab_fp.write_text(json.dumps(vocab.to_json_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"[StockNet] vocab saved -> {vocab_fp}")


if __name__ == "__main__":
    main()
