from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd


@dataclass
class _ProgressBar:
    desc: str
    total: Optional[int] = None
    width: int = 28
    min_interval_s: float = 0.4

    _last_print_s: float = 0.0
    _last_done: int = 0

    def update_abs(self, done: int, total: Optional[int] = None, suffix: str = "") -> None:
        if total is not None:
            self.total = int(total)
        now = time.time()
        if (now - self._last_print_s) < float(self.min_interval_s) and int(done) != int(self.total or -1):
            self._last_done = int(done)
            return

        self._last_print_s = now
        self._last_done = int(done)
        total2 = self.total
        if total2 is None or total2 <= 0:
            msg = f"\r{self.desc} {done}"
            if suffix:
                msg += f" {suffix}"
            print(msg, end="", flush=True)
            return

        ratio = min(1.0, max(0.0, float(done) / float(total2)))
        fill = int(round(ratio * float(self.width)))
        bar = "#" * fill + "-" * max(0, self.width - fill)
        msg = f"\r{self.desc} [{bar}] {done}/{total2} ({ratio*100:5.1f}%)"
        if suffix:
            msg += f" {suffix}"
        print(msg, end="", flush=True)

    def close(self, suffix: str = "") -> None:
        if self.total is None:
            print()
            return
        self.update_abs(done=int(self.total), total=int(self.total), suffix=suffix)
        print()


@dataclass
class _FinBertScorer:
    torch: object
    tokenizer: object
    model: object
    device: object
    max_length: int
    pos_idx: Optional[int]
    neu_idx: Optional[int]
    neg_idx: Optional[int]


def _try_import_finbert_deps():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as e:
        return None, e
    return (torch, AutoTokenizer, AutoModelForSequenceClassification), None


def _build_finbert_scorer(
    model_name: str,
    finbert_device: int,
    finbert_max_length: int,
) -> _FinBertScorer:
    imported, import_err = _try_import_finbert_deps()
    if imported is None:
        import sys

        msg = "需要安装 transformers 和 torch 才能使用 FinBERT 情绪打分"
        if import_err is not None:
            msg += f"（import 失败: {type(import_err).__name__}: {import_err}）"
        msg += f"（当前 python: {sys.executable}）"
        raise ImportError(msg)

    torch, AutoTokenizer, AutoModelForSequenceClassification = imported
    try:
        v = str(getattr(torch, "__version__", "")).split("+", 1)[0].strip()
        major, minor = (int(x) for x in v.split(".", 2)[:2])
        torch_version_ok = (major > 2) or (major == 2 and minor >= 6)
    except Exception:
        torch_version_ok = False

    model_path = Path(model_name)
    has_safetensors = False
    if model_path.exists() and model_path.is_dir():
        if any(model_path.glob("*.safetensors")):
            has_safetensors = True
        if (model_path / "model.safetensors").exists():
            has_safetensors = True

    if (not has_safetensors) and (not torch_version_ok):
        raise ValueError(
            "当前环境 torch<2.6 且模型不是 safetensors 格式，transformers 会阻止加载 pytorch_model.bin。"
            "解决方案：升级 torch 到 >=2.6，或下载/转换为 model.safetensors（推荐）。"
        )

    device = torch.device("cpu")
    if int(finbert_device) >= 0:
        try:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{int(finbert_device)}")
                _ = torch.zeros(1, device=device)
            else:
                print("[FinMultiTime] finbert_device 指定 GPU 但当前无可用 CUDA，已自动回退到 CPU")
        except Exception as e:
            print(f"[FinMultiTime] CUDA 初始化失败，已自动回退到 CPU: {type(e).__name__}: {e}")
            device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    pos_idx: Optional[int] = None
    neu_idx: Optional[int] = None
    neg_idx: Optional[int] = None
    try:
        label2id = getattr(model.config, "label2id", None)
        if isinstance(label2id, dict):
            for k, v in label2id.items():
                kk = str(k).lower()
                if "pos" in kk:
                    pos_idx = int(v)
                elif "neu" in kk:
                    neu_idx = int(v)
                elif "neg" in kk:
                    neg_idx = int(v)
    except Exception:
        pass

    return _FinBertScorer(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(finbert_max_length),
        pos_idx=pos_idx,
        neu_idx=neu_idx,
        neg_idx=neg_idx,
    )


def _finbert_sentiment_1to5(
    scorer: _FinBertScorer,
    texts: List[str],
    batch_size: int,
) -> List[float]:
    torch = scorer.torch
    tokenizer = scorer.tokenizer
    model = scorer.model
    device = scorer.device

    cleaned = [t.strip() if t and t.strip() else "N/A" for t in texts]
    out: List[float] = []

    bs = max(1, int(batch_size))
    for i in range(0, len(cleaned), bs):
        batch = cleaned[i : i + bs]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(scorer.max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for row in probs:
            s = 3.0
            try:
                if scorer.pos_idx is not None and scorer.neu_idx is not None and scorer.neg_idx is not None:
                    p_pos = float(row[int(scorer.pos_idx)])
                    p_neu = float(row[int(scorer.neu_idx)])
                    p_neg = float(row[int(scorer.neg_idx)])
                    s = 1.0 * p_neg + 3.0 * p_neu + 5.0 * p_pos
                else:
                    if row.shape[0] == 3:
                        s = float(row[0]) * 1.0 + float(row[1]) * 3.0 + float(row[2]) * 5.0
            except Exception:
                s = 3.0
            s = float(min(5.0, max(1.0, s)))
            out.append(s)

    return out


def _try_import_sumy():
    try:
        from sumy.nlp.stemmers import Stemmer
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.utils import get_stop_words
    except Exception:
        return None
    return Stemmer, Tokenizer, PlaintextParser, LsaSummarizer, get_stop_words


def _fallback_summary_with_keywords(text: str, symbol: str, num_sentences: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    parts: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?。\n":
            s = "".join(buf).strip()
            buf = []
            if s:
                parts.append(s)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    sentences = [s.strip() for s in parts if len(s.strip()) >= 10]
    if not sentences:
        return text

    kw = (symbol or "").strip().upper()
    scored: List[Tuple[float, int, str]] = []
    for i, s in enumerate(sentences):
        up = s.upper()
        has_kw = 1.0 if kw and kw in up else 0.0
        length = min(400.0, float(len(s)))
        score = has_kw * 1.5 + (length / 400.0) * 0.2
        scored.append((score, i, s))

    top = sorted(scored, key=lambda x: (x[0], -x[1]), reverse=True)[: int(num_sentences)]
    top_sorted = sorted(top, key=lambda x: x[1])
    out = " ".join(s for _, _, s in top_sorted).strip()
    return out or text


def _lsa_summary_with_keywords(text: str, symbol: str, num_sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    imported = _try_import_sumy()
    if imported is None:
        return _fallback_summary_with_keywords(text=text, symbol=symbol, num_sentences=int(num_sentences))

    lang = "chinese" if any("\u4e00" <= ch <= "\u9fff" for ch in text[: min(len(text), 1000)]) else "english"
    try:
        Stemmer, Tokenizer, PlaintextParser, LsaSummarizer, get_stop_words = imported
        stemmer = Stemmer(lang if lang in {"english", "chinese"} else "english")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(lang if lang in {"english", "chinese"} else "english")
        tokenizer = Tokenizer(lang if lang in {"english", "chinese"} else "english")
        parser = PlaintextParser.from_string(text, tokenizer)

        sentences = list(parser.document.sentences)
        if not sentences:
            return text

        initial_summary = list(summarizer(parser.document, int(num_sentences)))
        weights: Dict[object, float] = defaultdict(float)
        kw = (symbol or "").strip().upper()

        for s in sentences:
            s_text = str(s)
            if kw and kw in s_text.upper():
                weights[s] += 1.0

        for s in initial_summary:
            weights[s] += 1.0

        ranked = sorted(weights.keys(), key=lambda x: weights[x], reverse=True)[: int(num_sentences)]
        out = " ".join(str(s) for s in ranked).strip()
        return out or " ".join(str(s) for s in initial_summary).strip() or text
    except Exception:
        return _fallback_summary_with_keywords(text=text, symbol=symbol, num_sentences=int(num_sentences))


def _extract_date_key(obj: Dict[str, object]) -> Optional[str]:
    for key in ["Date", "date", "datetime", "Datetime", "publish_time", "time"]:
        val = obj.get(key)
        if isinstance(val, str):
            s = val.strip()
            if len(s) >= 10:
                return s[:10]
    return None


def _score_text_relevance(text: str, symbol: str) -> float:
    tx = (text or "").strip()
    if not tx:
        return -1e9
    up = tx.upper()
    sym = (symbol or "").strip().upper()
    mention = float(up.count(sym)) if sym else 0.0
    length = min(4000.0, float(len(tx)))
    chinese_chars = sum(1 for ch in tx[: min(len(tx), 1000)] if "\u4e00" <= ch <= "\u9fff")
    density_bonus = 0.1 if chinese_chars > 0 else 0.0
    return mention * 2.0 + length / 1000.0 + density_bonus


def _pick_text_for_scoring(obj: Dict[str, object], symbol: str, summarize_missing_lsa: bool) -> str:
    candidates = []
    for key in [
        "lexrank_summary",
        "Lexrank_summary",
        "textrank_summary",
        "Textrank_summary",
        "lsa_summary",
        "Lsa_summary",
        "luhn_summary",
        "Luhn_summary",
        "summary",
        "Summary",
        "article_title",
        "Article_title",
        "title",
        "Title",
        "headline",
        "Headline",
    ]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())
    if candidates:
        return max(candidates, key=lambda x: _score_text_relevance(x, symbol))

    article = ""
    for key in ["article", "Article", "content", "Content", "text", "Text", "body", "Body"]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            article = v.strip()
            break

    if summarize_missing_lsa and article:
        return _lsa_summary_with_keywords(article, symbol=symbol, num_sentences=3)
    return article


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _split_time_series(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n <= 0:
        return df.iloc[:0].copy(), df.iloc[:0].copy(), df.iloc[:0].copy()
    tr_end = int(math.floor(float(train_ratio) * n))
    va_end = int(math.floor((float(train_ratio) + float(val_ratio)) * n))
    tr_end = max(0, min(n, tr_end))
    va_end = max(tr_end, min(n, va_end))
    return df.iloc[:tr_end].copy(), df.iloc[tr_end:va_end].copy(), df.iloc[va_end:].copy()


def _normalize_with_train_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    tr = train_df.copy()
    va = val_df.copy()
    te = test_df.copy()

    for c in cols:
        if c not in tr.columns:
            continue
        mean = float(pd.to_numeric(tr[c], errors="coerce").mean())
        std = float(pd.to_numeric(tr[c], errors="coerce").std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            std = 1.0
        stats[c] = (mean, std)
        tr[c] = (pd.to_numeric(tr[c], errors="coerce") - mean) / std
        if c in va.columns:
            va[c] = (pd.to_numeric(va[c], errors="coerce") - mean) / std
        if c in te.columns:
            te[c] = (pd.to_numeric(te[c], errors="coerce") - mean) / std

    return tr, va, te, stats


def _add_labels(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    vol_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce").astype(np.float32)
    out.loc[out["close"] <= 0, "close"] = np.nan
    out["return"] = out["close"].pct_change(fill_method=None)
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["rolling_vol"] = out["return"].rolling(int(vol_window)).std(ddof=0)
    out["future_return"] = out["close"].shift(-int(horizon)) / out["close"] - 1.0
    out["threshold"] = out["rolling_vol"] * float(vol_threshold)

    out["label_cls"] = 0
    out.loc[out["future_return"] > out["threshold"], "label_cls"] = 1
    out.loc[out["future_return"] < -out["threshold"], "label_cls"] = -1

    future_returns = []
    for i in range(1, int(horizon) + 1):
        future_returns.append(out["log_return"].shift(-i))
    future_returns_df = pd.concat(future_returns, axis=1)
    out["future_volatility"] = np.sqrt(np.mean(np.square(future_returns_df), axis=1))
    out = out.dropna()
    return out


def _create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    label_col: str = "label_cls",
    reg_label_col: str = "future_volatility",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_len = int(seq_len)
    data = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    labels_cls = df[label_col].to_numpy()
    labels_reg = df[reg_label_col].to_numpy()

    n = len(df) - seq_len + 1
    if n <= 0:
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
        )

    X = np.zeros((n, seq_len, len(feature_cols)), dtype=np.float32)
    y_cls = np.zeros((n,), dtype=np.int8)
    y_reg = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        X[i] = data[i : i + seq_len]
        y_cls[i] = np.int8(labels_cls[i + seq_len - 1])
        y_reg[i] = np.float32(labels_reg[i + seq_len - 1])
    return X, y_cls, y_reg


def _iter_symbol_jsonl_files(text_dir: Path) -> Iterator[Tuple[str, Path]]:
    for fp in sorted(text_dir.glob("*.jsonl")):
        if fp.is_file():
            yield fp.stem.strip().upper(), fp


def _build_news_daily_sqlite_from_jsonl_dir(
    text_dir: Path,
    db_path: Path,
    finbert_model: str,
    finbert_device: int,
    finbert_max_length: int,
    finbert_batch_size: int,
    top_k_per_day: int,
    symbols: Optional[List[str]],
    summarize_missing_lsa: bool,
) -> None:
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    files = list(_iter_symbol_jsonl_files(text_dir))
    if wanted is not None:
        files = [(sym, fp) for sym, fp in files if sym in wanted]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_daily(
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            news_count INTEGER NOT NULL,
            sentiment_sum REAL NOT NULL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("sentiment_method", "finbert"))
    conn.commit()

    scorer = _build_finbert_scorer(model_name=finbert_model, finbert_device=int(finbert_device), finbert_max_length=int(finbert_max_length))

    pbar = _ProgressBar(desc="[FinMultiTime] news->sqlite", total=len(files))
    processed = 0
    kept = 0
    for sym, fp in files:
        processed += 1
        by_day_count: Dict[str, int] = {}
        by_day_top: Dict[str, List[Tuple[int, str]]] = {}
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    date = _extract_date_key(obj)
                    if date is None:
                        continue
                    by_day_count[date] = int(by_day_count.get(date, 0)) + 1
                    text = _pick_text_for_scoring(obj=obj, symbol=sym, summarize_missing_lsa=summarize_missing_lsa)
                    if not text:
                        continue
                    lst = by_day_top.get(date)
                    if lst is None:
                        lst = []
                        by_day_top[date] = lst
                    lst.append((len(text), text))
        except Exception:
            pbar.update_abs(done=processed, total=len(files), suffix=f" ok={kept} last={sym} read_fail")
            continue

        rows: List[Tuple[str, str, int, float]] = []
        all_texts: List[str] = []
        all_meta: List[Tuple[str, str, int]] = []

        for d, cnt in by_day_count.items():
            top = by_day_top.get(d, [])
            top_sorted = sorted(top, key=lambda x: (x[0], _score_text_relevance(x[1], sym)), reverse=True)[: max(1, int(top_k_per_day))]
            texts = [t for _, t in top_sorted if t and t.strip()]
            if not texts:
                rows.append((sym, d, int(cnt), float(3.0) * float(cnt)))
                continue
            all_texts.extend(texts)
            all_meta.append((sym, d, int(cnt)))

        sentiments = _finbert_sentiment_1to5(scorer=scorer, texts=all_texts, batch_size=int(finbert_batch_size)) if all_texts else []
        off = 0
        for sym2, d2, cnt2 in all_meta:
            top = by_day_top.get(d2, [])
            top_sorted = sorted(top, key=lambda x: (x[0], _score_text_relevance(x[1], sym2)), reverse=True)[: max(1, int(top_k_per_day))]
            k = len([t for _, t in top_sorted if t and t.strip()])
            if k <= 0:
                rows.append((sym2, d2, int(cnt2), float(3.0) * float(cnt2)))
                continue
            sc = sentiments[off : off + k]
            off += k
            avg = float(np.mean(np.asarray(sc, dtype=np.float32))) if sc else 3.0
            rows.append((sym2, d2, int(cnt2), float(avg) * float(cnt2)))

        try:
            conn.executemany(
                """
                INSERT INTO news_daily(symbol, date, news_count, sentiment_sum)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    news_count = excluded.news_count,
                    sentiment_sum = excluded.sentiment_sum
                """,
                rows,
            )
            conn.commit()
        except Exception:
            pbar.update_abs(done=processed, total=len(files), suffix=f" ok={kept} last={sym} db_fail")
            continue

        kept += 1
        pbar.update_abs(done=processed, total=len(files), suffix=f" ok={kept} last={sym}")

    pbar.close(suffix=f" ok={kept}")
    conn.close()
    print(f"[FinMultiTime] saved news sqlite -> {db_path}")


def _build_news_daily_sqlite_from_zip(
    zip_path: Path,
    db_path: Path,
    finbert_model: str,
    finbert_device: int,
    finbert_max_length: int,
    finbert_batch_size: int,
    top_k_per_day: int,
    symbols: Optional[List[str]],
    summarize_missing_lsa: bool,
) -> None:
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    if not zip_path.exists():
        raise FileNotFoundError(str(zip_path))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_daily(
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            news_count INTEGER NOT NULL,
            sentiment_sum REAL NOT NULL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("sentiment_method", "finbert"))
    conn.commit()

    scorer = _build_finbert_scorer(model_name=finbert_model, finbert_device=int(finbert_device), finbert_max_length=int(finbert_max_length))

    with ZipFile(str(zip_path), "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".jsonl") and not n.endswith("/")]
        items: List[Tuple[str, str]] = []
        for n in names:
            sym = Path(n).stem.strip().upper()
            if wanted is not None and sym not in wanted:
                continue
            items.append((sym, n))

        pbar = _ProgressBar(desc="[FinMultiTime] news(zip)->sqlite", total=len(items))
        processed = 0
        kept = 0
        for sym, name in items:
            processed += 1
            by_day_count: Dict[str, int] = {}
            by_day_top: Dict[str, List[Tuple[int, str]]] = {}
            try:
                with zf.open(name, "r") as f:
                    for raw in f:
                        try:
                            line = raw.decode("utf-8", errors="ignore").strip()
                        except Exception:
                            continue
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        date = _extract_date_key(obj)
                        if date is None:
                            continue
                        by_day_count[date] = int(by_day_count.get(date, 0)) + 1
                        text = _pick_text_for_scoring(obj=obj, symbol=sym, summarize_missing_lsa=summarize_missing_lsa)
                        if not text:
                            continue
                        lst = by_day_top.get(date)
                        if lst is None:
                            lst = []
                            by_day_top[date] = lst
                        lst.append((len(text), text))
            except Exception:
                pbar.update_abs(done=processed, total=len(items), suffix=f" ok={kept} last={sym} read_fail")
                continue

            rows: List[Tuple[str, str, int, float]] = []
            all_texts: List[str] = []
            all_meta: List[Tuple[str, str, int]] = []
            for d, cnt in by_day_count.items():
                top = by_day_top.get(d, [])
                top_sorted = sorted(top, key=lambda x: (x[0], _score_text_relevance(x[1], sym)), reverse=True)[: max(1, int(top_k_per_day))]
                texts = [t for _, t in top_sorted if t and t.strip()]
                if not texts:
                    rows.append((sym, d, int(cnt), float(3.0) * float(cnt)))
                    continue
                all_texts.extend(texts)
                all_meta.append((sym, d, int(cnt)))

            sentiments = _finbert_sentiment_1to5(scorer=scorer, texts=all_texts, batch_size=int(finbert_batch_size)) if all_texts else []
            off = 0
            for sym2, d2, cnt2 in all_meta:
                top = by_day_top.get(d2, [])
                top_sorted = sorted(top, key=lambda x: (x[0], _score_text_relevance(x[1], sym2)), reverse=True)[: max(1, int(top_k_per_day))]
                k = len([t for _, t in top_sorted if t and t.strip()])
                if k <= 0:
                    rows.append((sym2, d2, int(cnt2), float(3.0) * float(cnt2)))
                    continue
                sc = sentiments[off : off + k]
                off += k
                avg = float(np.mean(np.asarray(sc, dtype=np.float32))) if sc else 3.0
                rows.append((sym2, d2, int(cnt2), float(avg) * float(cnt2)))

            try:
                conn.executemany(
                    """
                    INSERT INTO news_daily(symbol, date, news_count, sentiment_sum)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET
                        news_count = excluded.news_count,
                        sentiment_sum = excluded.sentiment_sum
                    """,
                    rows,
                )
                conn.commit()
            except Exception:
                pbar.update_abs(done=processed, total=len(items), suffix=f" ok={kept} last={sym} db_fail")
                continue

            kept += 1
            pbar.update_abs(done=processed, total=len(items), suffix=f" ok={kept} last={sym}")

        pbar.close(suffix=f" ok={kept}")

    conn.close()
    print(f"[FinMultiTime] saved news sqlite -> {db_path}")


def _load_news_daily(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    cur = conn.execute("SELECT date, news_count, sentiment_sum FROM news_daily WHERE symbol = ? ORDER BY date", (symbol,))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame({"date": [], "news_count": [], "sentiment": []})
    df = pd.DataFrame(rows, columns=["date", "news_count", "sentiment_sum"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(np.int32)
    df["sentiment_sum"] = pd.to_numeric(df["sentiment_sum"], errors="coerce").fillna(0.0).astype(np.float32)
    df["sentiment"] = np.where(df["news_count"] > 0, df["sentiment_sum"] / df["news_count"], 3.0).astype(np.float32)
    return df[["date", "news_count", "sentiment"]]


def _apply_sentiment_decay(sentiment: pd.Series, news_flag: pd.Series, decay_rate: float) -> pd.Series:
    out = sentiment.copy()
    last = 3.0
    for i in range(len(out)):
        if int(news_flag.iloc[i]) == 1:
            val = float(out.iloc[i])
            if math.isfinite(val):
                last = val
            else:
                last = 3.0
            out.iloc[i] = last
        else:
            last = last + float(decay_rate) * (3.0 - last)
            out.iloc[i] = last
    return out.astype(np.float32)


def _compute_semiannual_trend(close: pd.Series, dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates, errors="coerce")
    year = dt.dt.year.to_numpy()
    month = dt.dt.month.to_numpy()
    half = np.where(month <= 6, 1, 2)

    close_f = pd.to_numeric(close, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    out = np.full((len(close_f),), 3, dtype=np.int8)

    keys: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(close_f)):
        if not np.isfinite(close_f[i]):
            continue
        y = int(year[i]) if year[i] == year[i] else None
        if y is None:
            continue
        k = (y, int(half[i]))
        lst = keys.get(k)
        if lst is None:
            lst = []
            keys[k] = lst
        lst.append(i)

    for idxs in keys.values():
        if not idxs:
            continue
        i0 = idxs[0]
        i1 = idxs[-1]
        c0 = float(close_f[i0])
        c1 = float(close_f[i1])
        if (not math.isfinite(c0)) or (not math.isfinite(c1)) or c0 <= 0 or c1 <= 0:
            score = 3
        else:
            log_ret = float(math.log(c1 / c0))
            seg = close_f[idxs]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 2:
                lr = np.diff(np.log(seg.clip(min=1e-12)))
                vol = float(np.std(lr, ddof=0))
            else:
                vol = 0.0
            denom = max(1e-8, vol)
            z = log_ret / denom

            if z >= 1.0:
                score = 2
            elif z >= 0.2:
                score = 1
            elif z <= -1.0:
                score = 5
            elif z <= -0.2:
                score = 4
            else:
                score = 3

        for i in idxs:
            out[i] = np.int8(score)
    return out


def _add_ohlcv_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = pd.to_numeric(out["close"], errors="coerce").pct_change(fill_method=None)
    out["log_return_1d"] = np.log(pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["close"], errors="coerce").shift(1))
    out["high_low_ratio"] = pd.to_numeric(out["high"], errors="coerce") / pd.to_numeric(out["low"], errors="coerce") - 1.0
    out["close_open_ratio"] = pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["open"], errors="coerce") - 1.0
    out["volume_log1p"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0))
    out["volume_change_1d"] = pd.to_numeric(out["volume"], errors="coerce").pct_change(fill_method=None)
    close = pd.to_numeric(out["close"], errors="coerce")
    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    out["ma5_gap"] = close / ma5 - 1.0
    out["ma20_gap"] = close / ma20 - 1.0
    out["volatility_20"] = out["return_1d"].rolling(20, min_periods=1).std(ddof=0)

    cols = [
        "return_1d",
        "log_return_1d",
        "high_low_ratio",
        "close_open_ratio",
        "volume_log1p",
        "volume_change_1d",
        "ma5_gap",
        "ma20_gap",
        "volatility_20",
    ]
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def _read_price_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = _standardize_columns(df)
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = df["datetime"]
        elif "time" in df.columns:
            df["date"] = df["time"]
        else:
            raise ValueError("price csv missing date column")

    date_str = df["date"].astype(str).str.slice(0, 10)
    df["date"] = pd.to_datetime(date_str, errors="coerce")

    ren = {}
    for src, dst in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume")]:
        if src in df.columns:
            ren[src] = dst
        elif src.title() in df.columns:
            ren[src.title()] = dst
    df = df.rename(columns=ren)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date")
    return df[["date", "open", "high", "low", "close", "volume"]]


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _signed_log1p_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.sign(x) * np.log1p(np.abs(x))


def _get_market_table_value_cols(market: str) -> List[str]:
    market_l = str(market).strip().lower()
    if market_l == "hs300":
        return ["table_net_profit", "table_operating_cash_flow", "table_free_cash_flow"]
    if market_l == "sp500":
        return ["table_stockholders_equity", "table_operating_cash_flow", "table_retained_earnings"]
    raise ValueError(f"Unknown market: {market}")


def _get_ohlcv_derived_cols() -> List[str]:
    return [
        "return_1d",
        "log_return_1d",
        "high_low_ratio",
        "close_open_ratio",
        "volume_log1p",
        "volume_change_1d",
        "ma5_gap",
        "ma20_gap",
        "volatility_20",
    ]


def _load_json_array_file(fp: Path) -> List[Dict[str, object]]:
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return []
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    rows: List[Dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj2 = json.loads(line)
        except Exception:
            continue
        if isinstance(obj2, list):
            rows.extend([x for x in obj2 if isinstance(x, dict)])
        elif isinstance(obj2, dict):
            rows.append(obj2)
    return rows


def _load_hs300_table_features(dataset_dir: Path, symbol: str) -> pd.DataFrame:
    symbol_dir = dataset_dir / "table" / "hs300_tabular" / symbol
    if not symbol_dir.exists():
        return pd.DataFrame(columns=["report_date", "table_net_profit", "table_operating_cash_flow", "table_free_cash_flow"])

    income_rows = _load_json_array_file(symbol_dir / "income.jsonl")
    cash_rows = _load_json_array_file(symbol_dir / "cashflow.jsonl")

    income_map: Dict[str, float] = {}
    for row in income_rows:
        end_date = row.get("end_date")
        if not isinstance(end_date, str):
            continue
        val = _safe_float(row.get("n_income_attr_p"))
        if val is None:
            val = _safe_float(row.get("n_income"))
        if val is not None:
            income_map[end_date] = val

    cash_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for row in cash_rows:
        end_date = row.get("end_date")
        if not isinstance(end_date, str):
            continue
        ocf = _safe_float(row.get("n_cashflow_act"))
        capex = _safe_float(row.get("c_pay_acq_const_fiolta"))
        cash_map[end_date] = (ocf, capex)

    all_dates = sorted(set(income_map.keys()) | set(cash_map.keys()))
    rows: List[Dict[str, object]] = []
    for d in all_dates:
        ocf, capex = cash_map.get(d, (None, None))
        free_cf = None
        if ocf is not None:
            free_cf = float(ocf - (capex or 0.0))
        rows.append(
            {
                "report_date": pd.to_datetime(d, format="%Y%m%d", errors="coerce"),
                "table_net_profit": income_map.get(d),
                "table_operating_cash_flow": ocf,
                "table_free_cash_flow": free_cf,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["report_date", "table_net_profit", "table_operating_cash_flow", "table_free_cash_flow"])
    df = df.dropna(subset=["report_date"]).sort_values("report_date").drop_duplicates(subset=["report_date"], keep="last")
    return df


def _extract_sp500_usgaap_series(facts: Dict[str, object], concept: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    item = facts.get(concept)
    if not isinstance(item, dict):
        return out
    units = item.get("units")
    if not isinstance(units, dict):
        return out

    candidates: List[Dict[str, object]] = []
    for unit_name, entries in units.items():
        if not isinstance(unit_name, str) or "USD" not in unit_name.upper():
            continue
        if not isinstance(entries, list):
            continue
        for ent in entries:
            if isinstance(ent, dict):
                candidates.append(ent)

    best: Dict[str, Tuple[str, float]] = {}
    for ent in candidates:
        end = ent.get("end")
        if not isinstance(end, str) or len(end) < 10:
            continue
        end_key = end[:10]
        val = _safe_float(ent.get("val"))
        if val is None:
            continue
        filed = str(ent.get("filed") or "")
        prev = best.get(end_key)
        if prev is None or filed > prev[0]:
            best[end_key] = (filed, val)

    for end_key, (_, val) in best.items():
        out[end_key] = float(val)
    return out


def _load_sp500_table_features_from_zip(table_zip_path: Path, symbol: str) -> pd.DataFrame:
    base = f"financial_reports/{symbol.lower()}/"
    files = {
        "balance": base + "condensed_consolidated_balance_sheets.json",
        "equity": base + "condensed_consolidated_statement_of_equity.json",
        "cash": base + "condensed_consolidated_statement_of_cash_flows.json",
    }
    if not table_zip_path.exists():
        return pd.DataFrame(columns=["report_date", "table_stockholders_equity", "table_operating_cash_flow", "table_retained_earnings"])

    with ZipFile(str(table_zip_path), "r") as zf:
        try:
            balance_obj = json.load(zf.open(files["balance"]))
            equity_obj = json.load(zf.open(files["equity"]))
            cash_obj = json.load(zf.open(files["cash"]))
        except Exception:
            return pd.DataFrame(columns=["report_date", "table_stockholders_equity", "table_operating_cash_flow", "table_retained_earnings"])

    def _merge_filings(obj: Dict[str, object]) -> Dict[str, object]:
        filings = obj.get("filings")
        merged: Dict[str, object] = {}
        if not isinstance(filings, list):
            return merged
        for filing in filings:
            if not isinstance(filing, dict):
                continue
            facts = filing.get("facts")
            if not isinstance(facts, dict):
                continue
            us_gaap = facts.get("us-gaap")
            if not isinstance(us_gaap, dict):
                continue
            for k, v in us_gaap.items():
                if k not in merged and isinstance(v, dict):
                    merged[k] = v
        return merged

    balance_facts = _merge_filings(balance_obj)
    equity_facts = _merge_filings(equity_obj)
    cash_facts = _merge_filings(cash_obj)

    stockholders_equity = _extract_sp500_usgaap_series(balance_facts, "StockholdersEquity")
    if not stockholders_equity:
        stockholders_equity = _extract_sp500_usgaap_series(balance_facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    retained_earnings = _extract_sp500_usgaap_series(equity_facts, "RetainedEarningsAccumulatedDeficit")
    operating_cash_flow = _extract_sp500_usgaap_series(cash_facts, "NetCashProvidedByUsedInOperatingActivities")

    all_dates = sorted(set(stockholders_equity.keys()) | set(retained_earnings.keys()) | set(operating_cash_flow.keys()))
    rows: List[Dict[str, object]] = []
    for d in all_dates:
        rows.append(
            {
                "report_date": pd.to_datetime(d, errors="coerce"),
                "table_stockholders_equity": stockholders_equity.get(d),
                "table_operating_cash_flow": operating_cash_flow.get(d),
                "table_retained_earnings": retained_earnings.get(d),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["report_date", "table_stockholders_equity", "table_operating_cash_flow", "table_retained_earnings"])
    df = df.dropna(subset=["report_date"]).sort_values("report_date").drop_duplicates(subset=["report_date"], keep="last")
    return df


def _align_table_features_to_price_dates(price_df: pd.DataFrame, table_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()

    feature_cols = [c for c in table_df.columns if c != "report_date"]
    if not feature_cols:
        return out

    if table_df.empty:
        for c in feature_cols:
            out[c] = np.float32(0.0)
            out[f"{c}_missing"] = np.int8(1)
        out["table_report_age_days"] = np.float32(-1.0)
        out["table_missing_any"] = np.int8(1)
        return out

    price_dates = pd.to_datetime(out["date"], errors="coerce")
    price_vals = price_dates.to_numpy(dtype="datetime64[ns]")
    aligned_rows: List[Dict[str, object]] = []

    for _, row in table_df.iterrows():
        report_date = pd.to_datetime(row["report_date"], errors="coerce")
        if pd.isna(report_date):
            continue
        idx = int(np.searchsorted(price_vals, np.datetime64(report_date), side="right")) - 1
        if idx < 0 or idx >= len(out):
            continue
        item: Dict[str, object] = {"date": out.iloc[idx]["date"], "_table_anchor_date": report_date}
        for c in feature_cols:
            item[c] = row.get(c)
        aligned_rows.append(item)

    if not aligned_rows:
        for c in feature_cols:
            out[c] = np.float32(0.0)
            out[f"{c}_missing"] = np.int8(1)
        out["table_report_age_days"] = np.float32(-1.0)
        out["table_missing_any"] = np.int8(1)
        return out

    aligned_df = pd.DataFrame(aligned_rows).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out.merge(aligned_df, on="date", how="left")
    out = out.sort_values("date")

    out["_table_anchor_date"] = pd.to_datetime(out["_table_anchor_date"], errors="coerce").ffill()
    age_days = (pd.to_datetime(out["date"], errors="coerce") - out["_table_anchor_date"]).dt.days
    out["table_report_age_days"] = pd.to_numeric(age_days, errors="coerce").fillna(-1.0).astype(np.float32)

    missing_cols: List[str] = []
    for c in feature_cols:
        raw = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ffilled = raw.ffill()
        miss_col = f"{c}_missing"
        out[miss_col] = ffilled.isna().astype(np.int8)
        out[c] = _signed_log1p_series(ffilled).fillna(0.0).astype(np.float32)
        missing_cols.append(miss_col)
    if missing_cols:
        out["table_missing_any"] = out[missing_cols].max(axis=1).astype(np.int8)
    else:
        out["table_missing_any"] = np.int8(0)
    out = out.drop(columns=["_table_anchor_date"], errors="ignore")
    return out


def _update_table_missing_stats(stats: Dict[str, object], df_out: pd.DataFrame) -> None:
    total_rows = int(stats.get("total_rows", 0))
    stats["total_rows"] = total_rows + int(len(df_out))
    stats["symbols"] = int(stats.get("symbols", 0)) + 1

    feature_missing_counts = stats.get("feature_missing_counts")
    if not isinstance(feature_missing_counts, dict):
        feature_missing_counts = {}
        stats["feature_missing_counts"] = feature_missing_counts

    for c in df_out.columns:
        if c.endswith("_missing") or c == "table_missing_any":
            feature_missing_counts[c] = float(feature_missing_counts.get(c, 0.0)) + float(pd.to_numeric(df_out[c], errors="coerce").fillna(0).sum())

    age_sum = float(stats.get("table_report_age_days_sum", 0.0))
    age_cnt = int(stats.get("table_report_age_days_count", 0))
    if "table_report_age_days" in df_out.columns:
        age_series = pd.to_numeric(df_out["table_report_age_days"], errors="coerce")
        valid = age_series[age_series >= 0]
        stats["table_report_age_days_sum"] = age_sum + float(valid.sum())
        stats["table_report_age_days_count"] = age_cnt + int(valid.shape[0])


def _finalize_table_missing_stats(stats: Dict[str, object]) -> Dict[str, object]:
    out = dict(stats)
    total_rows = max(1, int(out.get("total_rows", 0)))
    feature_missing_counts = out.get("feature_missing_counts")
    ratios: Dict[str, float] = {}
    if isinstance(feature_missing_counts, dict):
        for k, v in feature_missing_counts.items():
            ratios[str(k)] = float(v) / float(total_rows)
    out["feature_missing_ratio"] = ratios
    age_cnt = int(out.get("table_report_age_days_count", 0))
    age_sum = float(out.get("table_report_age_days_sum", 0.0))
    out["table_report_age_days_mean"] = float(age_sum / age_cnt) if age_cnt > 0 else -1.0
    return out


def _iter_price_symbols_from_dir(price_dir: Path) -> Iterator[Tuple[str, Path]]:
    for fp in sorted(price_dir.glob("*.csv")):
        if fp.is_file():
            yield fp.stem.strip().upper(), fp


def _iter_price_symbols_from_zip(zip_path: Path) -> Iterator[Tuple[str, str]]:
    with ZipFile(str(zip_path), "r") as zf:
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            if not n.lower().endswith(".csv"):
                continue
            sym = Path(n).stem.strip().upper()
            yield sym, n


def _read_price_from_zip(zf: ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name, "r") as f:
        df = pd.read_csv(f)
    df = _standardize_columns(df)
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = df["datetime"]
        elif "time" in df.columns:
            df["date"] = df["time"]
        else:
            raise ValueError("price csv missing date column")
    date_str = df["date"].astype(str).str.slice(0, 10)
    df["date"] = pd.to_datetime(date_str, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date")
    return df[["date", "open", "high", "low", "close", "volume"]]


def _save_integrated_outputs(
    df_out: pd.DataFrame,
    out_fp: Path,
    save_csv: bool,
    save_npz: bool,
) -> None:
    if bool(save_csv):
        df_out.to_csv(out_fp, index=False)

    if bool(save_npz):
        npz_fp = out_fp.with_suffix(".npz")
        feature_cols = [c for c in df_out.columns if c != "date"]
        np.savez_compressed(
            str(npz_fp),
            date=df_out["date"].astype(str).to_numpy(dtype=object),
            data=df_out[feature_cols].to_numpy(dtype=np.float32, copy=False),
            feature_cols=np.asarray(feature_cols, dtype=object),
        )


def integrate_price_with_news(
    market: str,
    dataset_dir: Path,
    output_dir: Path,
    news_db_path: Path,
    decay_rate: float,
    symbols: Optional[List[str]],
    save_csv: bool,
    save_npz: bool,
) -> Path:
    market_l = str(market).strip().lower()
    if market_l not in {"hs300", "sp500"}:
        raise ValueError(f"Unknown market: {market}")

    out_dir = output_dir / market_l
    integrated_dir = out_dir / "integrated"
    integrated_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(news_db_path))
    table_sp500_zip = dataset_dir / "table" / "SP500_tabular.zip"
    table_stats: Dict[str, object] = {"market": market_l, "symbols": 0, "total_rows": 0, "feature_missing_counts": {}}

    price_dir = dataset_dir / "time_series" / ("HS300_time_series" if market_l == "hs300" else "S&P500_time_series")
    price_zip = dataset_dir / "time_series" / ("HS300_time_series.zip" if market_l == "hs300" else "S%26P500_time_series.zip")

    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None

    if price_dir.exists() and price_dir.is_dir():
        items = list(_iter_price_symbols_from_dir(price_dir))
        if wanted is not None:
            items = [(s, p) for s, p in items if s in wanted]
        pbar = _ProgressBar(desc=f"[FinMultiTime] integrate {market_l}", total=len(items))
        processed = 0
        kept = 0
        for sym, fp in items:
            processed += 1
            try:
                price = _read_price_csv(fp)
            except Exception:
                pbar.update_abs(done=processed, total=len(items), suffix=f" ok={kept} last={sym} price_fail")
                continue
            price = _add_ohlcv_derived_features(price)
            if market_l == "hs300":
                table_df = _load_hs300_table_features(dataset_dir=dataset_dir, symbol=sym)
            else:
                table_df = _load_sp500_table_features_from_zip(table_zip_path=table_sp500_zip, symbol=sym)
            price = _align_table_features_to_price_dates(price_df=price, table_df=table_df)
            news = _load_news_daily(conn, sym)
            df = price.merge(news, on="date", how="left")
            df["news_count"] = pd.to_numeric(df.get("news_count"), errors="coerce").fillna(0).astype(np.int32)
            df["news_flag"] = (df["news_count"] > 0).astype(np.int8)
            df["sentiment"] = pd.to_numeric(df.get("sentiment"), errors="coerce").astype(np.float32)
            df.loc[df["news_flag"] == 0, "sentiment"] = np.nan
            df["sentiment"] = _apply_sentiment_decay(df["sentiment"].fillna(3.0), df["news_flag"], decay_rate=float(decay_rate))
            df["scaled_sentiment"] = (df["sentiment"] - 0.9999) / 4.0
            df["news_count_log1p"] = np.log1p(df["news_count"].astype(np.float32))
            df["kline_trend"] = _compute_semiannual_trend(df["close"], df["date"])
            df["kline_trend_scaled"] = (df["kline_trend"].astype(np.float32) - 0.9999) / 4.0

            out_fp = integrated_dir / f"{sym}.csv"
            table_value_cols = _get_market_table_value_cols(market_l)
            table_missing_cols = [f"{c}_missing" for c in table_value_cols]
            ohlcv_cols = _get_ohlcv_derived_cols()
            df_out = df[
                ["date", "open", "high", "low", "close", "volume"]
                + ohlcv_cols
                + ["scaled_sentiment", "news_flag", "news_count_log1p", "kline_trend_scaled"]
                + table_value_cols
                + table_missing_cols
                + ["table_report_age_days", "table_missing_any"]
            ]
            _update_table_missing_stats(table_stats, df_out)
            _save_integrated_outputs(df_out=df_out, out_fp=out_fp, save_csv=save_csv, save_npz=save_npz)
            kept += 1
            pbar.update_abs(done=processed, total=len(items), suffix=f" ok={kept} last={sym}")
        pbar.close(suffix=f" ok={kept}")
        stats_fp = out_dir / "table_missing_stats.json"
        stats_fp.write_text(json.dumps(_finalize_table_missing_stats(table_stats), ensure_ascii=False, indent=2), encoding="utf-8")
        conn.close()
        return integrated_dir

    if not price_zip.exists():
        raise FileNotFoundError(f"missing price source: {price_dir} or {price_zip}")

    with ZipFile(str(price_zip), "r") as zf:
        items2 = list(_iter_price_symbols_from_zip(price_zip))
        if wanted is not None:
            items2 = [(s, n) for s, n in items2 if s in wanted]
        pbar = _ProgressBar(desc=f"[FinMultiTime] integrate(zip) {market_l}", total=len(items2))
        processed = 0
        kept = 0
        for sym, name in items2:
            processed += 1
            try:
                price = _read_price_from_zip(zf, name)
            except Exception:
                pbar.update_abs(done=processed, total=len(items2), suffix=f" ok={kept} last={sym} price_fail")
                continue
            price = _add_ohlcv_derived_features(price)
            if market_l == "hs300":
                table_df = _load_hs300_table_features(dataset_dir=dataset_dir, symbol=sym)
            else:
                table_df = _load_sp500_table_features_from_zip(table_zip_path=table_sp500_zip, symbol=sym)
            price = _align_table_features_to_price_dates(price_df=price, table_df=table_df)
            news = _load_news_daily(conn, sym)
            df = price.merge(news, on="date", how="left")
            df["news_count"] = pd.to_numeric(df.get("news_count"), errors="coerce").fillna(0).astype(np.int32)
            df["news_flag"] = (df["news_count"] > 0).astype(np.int8)
            df["sentiment"] = pd.to_numeric(df.get("sentiment"), errors="coerce").astype(np.float32)
            df.loc[df["news_flag"] == 0, "sentiment"] = np.nan
            df["sentiment"] = _apply_sentiment_decay(df["sentiment"].fillna(3.0), df["news_flag"], decay_rate=float(decay_rate))
            df["scaled_sentiment"] = (df["sentiment"] - 0.9999) / 4.0
            df["news_count_log1p"] = np.log1p(df["news_count"].astype(np.float32))
            df["kline_trend"] = _compute_semiannual_trend(df["close"], df["date"])
            df["kline_trend_scaled"] = (df["kline_trend"].astype(np.float32) - 0.9999) / 4.0

            out_fp = integrated_dir / f"{sym}.csv"
            table_value_cols = _get_market_table_value_cols(market_l)
            table_missing_cols = [f"{c}_missing" for c in table_value_cols]
            ohlcv_cols = _get_ohlcv_derived_cols()
            df_out = df[
                ["date", "open", "high", "low", "close", "volume"]
                + ohlcv_cols
                + ["scaled_sentiment", "news_flag", "news_count_log1p", "kline_trend_scaled"]
                + table_value_cols
                + table_missing_cols
                + ["table_report_age_days", "table_missing_any"]
            ]
            _update_table_missing_stats(table_stats, df_out)
            _save_integrated_outputs(df_out=df_out, out_fp=out_fp, save_csv=save_csv, save_npz=save_npz)
            kept += 1
            pbar.update_abs(done=processed, total=len(items2), suffix=f" ok={kept} last={sym}")
        pbar.close(suffix=f" ok={kept}")

    stats_fp = out_dir / "table_missing_stats.json"
    stats_fp.write_text(json.dumps(_finalize_table_missing_stats(table_stats), ensure_ascii=False, indent=2), encoding="utf-8")
    conn.close()
    return integrated_dir


def export_sequences_npz_sharded(
    integrated_dir: Path,
    output_dir: Path,
    dataset_name: str,
    symbols: Optional[List[str]],
    seq_len: int,
    horizon: int,
    vol_window: int,
    vol_threshold: float,
    train_ratio: float,
    val_ratio: float,
    normalize_numeric: bool,
    shard_max_train_samples: int,
    manifest_path: Path,
) -> None:
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    csv_files = sorted([p for p in integrated_dir.glob("*.csv") if p.is_file()])
    if wanted is not None:
        csv_files = [p for p in csv_files if p.stem.upper() in wanted]

    if not csv_files:
        raise RuntimeError(f"integrated_dir 下没有可用 CSV: {integrated_dir}")
    header_df = pd.read_csv(csv_files[0], nrows=1)
    header_cols = [str(c).strip().lower() for c in header_df.columns]
    feature_cols = [c for c in header_cols if c != "date"]
    binary_feature_cols = [c for c in feature_cols if c == "news_flag" or c.endswith("_missing") or c == "table_missing_any"]
    table_value_cols = [c for c in feature_cols if c.startswith("table_") and not c.endswith("_missing") and c not in {"table_report_age_days", "table_missing_any"}]
    normalize_exclude = set(binary_feature_cols)
    normalize_cols = [c for c in feature_cols if c not in normalize_exclude]

    output_dir.mkdir(parents=True, exist_ok=True)

    def _empty_X() -> np.ndarray:
        return np.zeros((0, int(seq_len), len(feature_cols)), dtype=np.float32)

    def _empty_y_cls() -> np.ndarray:
        return np.zeros((0,), dtype=np.int8)

    def _empty_y_reg() -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)

    X_train_buf: List[np.ndarray] = []
    y_train_cls_buf: List[np.ndarray] = []
    y_train_reg_buf: List[np.ndarray] = []
    X_val_buf: List[np.ndarray] = []
    y_val_cls_buf: List[np.ndarray] = []
    y_val_reg_buf: List[np.ndarray] = []
    X_test_buf: List[np.ndarray] = []
    y_test_cls_buf: List[np.ndarray] = []
    y_test_reg_buf: List[np.ndarray] = []

    buf_train_n = 0
    shard_idx = 0

    manifest: Dict[str, object] = {
        "format": "finmultitime_processed_shards",
        "dataset_name": str(dataset_name),
        "integrated_dir": str(integrated_dir),
        "feature_cols": feature_cols,
        "binary_feature_cols": binary_feature_cols,
        "table_value_cols": table_value_cols,
        "seq_len": int(seq_len),
        "horizon": int(horizon),
        "vol_window": int(vol_window),
        "vol_threshold": float(vol_threshold),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "normalize_numeric": bool(normalize_numeric),
        "shard_max_train_samples": int(shard_max_train_samples),
        "shards": [],
    }

    def _flush(force: bool) -> None:
        nonlocal buf_train_n, shard_idx
        if (not force) and buf_train_n < int(shard_max_train_samples):
            return
        if not X_train_buf:
            return

        X_train = np.concatenate(X_train_buf, axis=0) if X_train_buf else _empty_X()
        y_train_cls = np.concatenate(y_train_cls_buf, axis=0) if y_train_cls_buf else _empty_y_cls()
        y_train_reg = np.concatenate(y_train_reg_buf, axis=0) if y_train_reg_buf else _empty_y_reg()
        X_val = np.concatenate(X_val_buf, axis=0) if X_val_buf else _empty_X()
        y_val_cls = np.concatenate(y_val_cls_buf, axis=0) if y_val_cls_buf else _empty_y_cls()
        y_val_reg = np.concatenate(y_val_reg_buf, axis=0) if y_val_reg_buf else _empty_y_reg()
        X_test = np.concatenate(X_test_buf, axis=0) if X_test_buf else _empty_X()
        y_test_cls = np.concatenate(y_test_cls_buf, axis=0) if y_test_cls_buf else _empty_y_cls()
        y_test_reg = np.concatenate(y_test_reg_buf, axis=0) if y_test_reg_buf else _empty_y_reg()

        shard_name = f"{dataset_name}_shard_{shard_idx:05d}.npz"
        out_fp = output_dir / shard_name
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
            feature_cols=np.asarray(feature_cols, dtype=object),
            seq_len=np.int32(int(seq_len)),
            horizon=np.int32(int(horizon)),
        )

        cast_shards = manifest["shards"]
        if isinstance(cast_shards, list):
            cast_shards.append(
                {"file": shard_name, "train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])}
            )

        shard_idx += 1
        X_train_buf.clear()
        y_train_cls_buf.clear()
        y_train_reg_buf.clear()
        X_val_buf.clear()
        y_val_cls_buf.clear()
        y_val_reg_buf.clear()
        X_test_buf.clear()
        y_test_cls_buf.clear()
        y_test_reg_buf.clear()
        buf_train_n = 0

    pbar = _ProgressBar(desc=f"[FinMultiTime] shards {dataset_name}", total=len(csv_files))
    processed = 0
    kept = 0
    for fp in csv_files:
        processed += 1
        sym = fp.stem.upper()
        try:
            df = pd.read_csv(fp)
        except Exception:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym} read_fail")
            continue

        df = _standardize_columns(df)
        if "date" not in df.columns:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym} no_date")
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        need_cols = ["open", "high", "low", "close", "volume", "scaled_sentiment", "news_flag", "news_count_log1p", "kline_trend_scaled"]
        for c in need_cols:
            if c not in df.columns:
                if c == "scaled_sentiment":
                    df[c] = (3.0 - 0.9999) / 4.0
                elif c == "news_flag":
                    df[c] = 0
                elif c == "news_count_log1p":
                    df[c] = 0.0
                elif c == "kline_trend_scaled":
                    df[c] = (3.0 - 0.9999) / 4.0
                else:
                    pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym} miss_{c}")
                    continue

        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = _add_labels(df, horizon=int(horizon), vol_window=int(vol_window), vol_threshold=float(vol_threshold))
        if len(df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym} too_short")
            continue

        train_df, val_df, test_df = _split_time_series(df, train_ratio=float(train_ratio), val_ratio=float(val_ratio))
        if len(train_df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym} train_short")
            continue

        if normalize_numeric:
            train_df, val_df, test_df, _ = _normalize_with_train_stats(train_df, val_df, test_df, cols=normalize_cols)

        X_tr, y_tr_c, y_tr_r = _create_sequences(train_df, feature_cols, seq_len=int(seq_len))
        if len(X_tr) == 0:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym} no_seq")
            continue

        X_va, y_va_c, y_va_r = _create_sequences(val_df, feature_cols, seq_len=int(seq_len)) if len(val_df) >= int(seq_len) else (
            _empty_X(),
            _empty_y_cls(),
            _empty_y_reg(),
        )
        X_te, y_te_c, y_te_r = _create_sequences(test_df, feature_cols, seq_len=int(seq_len)) if len(test_df) >= int(seq_len) else (
            _empty_X(),
            _empty_y_cls(),
            _empty_y_reg(),
        )

        X_train_buf.append(X_tr)
        y_train_cls_buf.append(y_tr_c)
        y_train_reg_buf.append(y_tr_r)
        if len(X_va):
            X_val_buf.append(X_va)
            y_val_cls_buf.append(y_va_c)
            y_val_reg_buf.append(y_va_r)
        if len(X_te):
            X_test_buf.append(X_te)
            y_test_cls_buf.append(y_te_c)
            y_test_reg_buf.append(y_te_r)

        buf_train_n += int(X_tr.shape[0])
        kept += 1
        _flush(force=False)

        extra = ""
        if len(X_va) == 0:
            extra += " val0"
        if len(X_te) == 0:
            extra += " test0"
        pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} shards={shard_idx} last={sym}{extra}")

    _flush(force=True)
    pbar.close(suffix=f" ok={kept} shards={shard_idx}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"[FinMultiTime] saved manifest -> {manifest_path}")


def export_sequences_npz_single(
    integrated_dir: Path,
    output_npz: Path,
    symbols: Optional[List[str]],
    seq_len: int,
    horizon: int,
    vol_window: int,
    vol_threshold: float,
    train_ratio: float,
    val_ratio: float,
    normalize_numeric: bool,
) -> None:
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    csv_files = sorted([p for p in integrated_dir.glob("*.csv") if p.is_file()])
    if wanted is not None:
        csv_files = [p for p in csv_files if p.stem.upper() in wanted]

    if not csv_files:
        raise RuntimeError(f"integrated_dir 下没有可用 CSV: {integrated_dir}")
    header_df = pd.read_csv(csv_files[0], nrows=1)
    header_cols = [str(c).strip().lower() for c in header_df.columns]
    feature_cols = [c for c in header_cols if c != "date"]
    binary_feature_cols = [c for c in feature_cols if c == "news_flag" or c.endswith("_missing") or c == "table_missing_any"]
    normalize_exclude = set(binary_feature_cols)
    normalize_cols = [c for c in feature_cols if c not in normalize_exclude]

    X_train_list: List[np.ndarray] = []
    y_train_cls_list: List[np.ndarray] = []
    y_train_reg_list: List[np.ndarray] = []
    X_val_list: List[np.ndarray] = []
    y_val_cls_list: List[np.ndarray] = []
    y_val_reg_list: List[np.ndarray] = []
    X_test_list: List[np.ndarray] = []
    y_test_cls_list: List[np.ndarray] = []
    y_test_reg_list: List[np.ndarray] = []

    pbar = _ProgressBar(desc="[FinMultiTime] single npz", total=len(csv_files))
    processed = 0
    kept = 0

    for fp in csv_files:
        processed += 1
        sym = fp.stem.upper()
        try:
            df = pd.read_csv(fp)
        except Exception:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} last={sym} read_fail")
            continue

        df = _standardize_columns(df)
        if "date" not in df.columns:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} last={sym} no_date")
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = _add_labels(df, horizon=int(horizon), vol_window=int(vol_window), vol_threshold=float(vol_threshold))
        if len(df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} last={sym} too_short")
            continue

        train_df, val_df, test_df = _split_time_series(df, train_ratio=float(train_ratio), val_ratio=float(val_ratio))
        if len(train_df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} last={sym} train_short")
            continue

        if normalize_numeric:
            train_df, val_df, test_df, _ = _normalize_with_train_stats(train_df, val_df, test_df, cols=normalize_cols)

        X_tr, y_tr_c, y_tr_r = _create_sequences(train_df, feature_cols, seq_len=int(seq_len))
        if len(X_tr) == 0:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} last={sym} no_seq")
            continue

        X_va, y_va_c, y_va_r = _create_sequences(val_df, feature_cols, seq_len=int(seq_len)) if len(val_df) >= int(seq_len) else (
            np.zeros((0, int(seq_len), len(feature_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
        )
        X_te, y_te_c, y_te_r = _create_sequences(test_df, feature_cols, seq_len=int(seq_len)) if len(test_df) >= int(seq_len) else (
            np.zeros((0, int(seq_len), len(feature_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
        )

        X_train_list.append(X_tr)
        y_train_cls_list.append(y_tr_c)
        y_train_reg_list.append(y_tr_r)
        X_val_list.append(X_va)
        y_val_cls_list.append(y_va_c)
        y_val_reg_list.append(y_va_r)
        X_test_list.append(X_te)
        y_test_cls_list.append(y_te_c)
        y_test_reg_list.append(y_te_r)

        kept += 1
        pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept} last={sym}")

    pbar.close(suffix=f" ok={kept}")

    if not X_train_list:
        raise RuntimeError("没有生成任何样本（请检查 integrated_dir 或放宽 seq_len/split 条件）")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train_cls = np.concatenate(y_train_cls_list, axis=0)
    y_train_reg = np.concatenate(y_train_reg_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val_cls = np.concatenate(y_val_cls_list, axis=0)
    y_val_reg = np.concatenate(y_val_reg_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test_cls = np.concatenate(y_test_cls_list, axis=0)
    y_test_reg = np.concatenate(y_test_reg_list, axis=0)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
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
        feature_cols=np.asarray(feature_cols, dtype=object),
        seq_len=np.int32(int(seq_len)),
        horizon=np.int32(int(horizon)),
        vol_window=np.int32(int(vol_window)),
        vol_threshold=np.float32(float(vol_threshold)),
    )
    print(f"[FinMultiTime] saved single npz -> {output_npz}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--market", type=str, default="both", choices=["hs300", "sp500", "both"])
    parser.add_argument("--symbols", type=str, default=None)

    parser.add_argument("--build_index", type=int, default=1)
    parser.add_argument("--integrate", type=int, default=1)
    parser.add_argument("--export_shards", type=int, default=1)
    parser.add_argument("--export_single", type=int, default=0)
    parser.add_argument("--save_csv", type=int, default=1)
    parser.add_argument("--save_npz", type=int, default=1)

    parser.add_argument("--finbert_model", type=str, default=None)
    parser.add_argument("--finbert_device", type=int, default=0)
    parser.add_argument("--finbert_max_length", type=int, default=256)
    parser.add_argument("--finbert_batch_size", type=int, default=16)
    parser.add_argument("--top_k_per_day", type=int, default=3)
    parser.add_argument("--summarize_missing_lsa", type=int, default=1)

    parser.add_argument("--decay_rate", type=float, default=0.05)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--vol_window", type=int, default=20)
    parser.add_argument("--vol_threshold", type=float, default=1.5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--normalize_numeric", type=int, default=1)

    parser.add_argument("--shard_max_train_samples", type=int, default=200000)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (repo_root / "datasets" / "FinMultitime")
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "data" / "finmultitime")

    symbols = [s.strip().upper() for s in str(args.symbols).split(",")] if args.symbols else None

    finbert_model = args.finbert_model
    if not finbert_model:
        local = repo_root / "datasets" / "finbert"
        finbert_model = str(local) if local.exists() else "ProsusAI/finbert"

    markets = ["hs300", "sp500"] if str(args.market).lower() == "both" else [str(args.market).lower()]

    for mk in markets:
        out_mk = output_dir / mk
        out_mk.mkdir(parents=True, exist_ok=True)

        if int(args.build_index) == 1:
            db_path = out_mk / "news_daily_finbert.sqlite"
            if mk == "sp500":
                text_dir = dataset_dir / "text" / "sp500_news"
                text_zip = dataset_dir / "text" / "sp500_news.zip"
                if text_dir.exists() and text_dir.is_dir():
                    _build_news_daily_sqlite_from_jsonl_dir(
                        text_dir=text_dir,
                        db_path=db_path,
                        finbert_model=str(finbert_model),
                        finbert_device=int(args.finbert_device),
                        finbert_max_length=int(args.finbert_max_length),
                        finbert_batch_size=int(args.finbert_batch_size),
                        top_k_per_day=int(args.top_k_per_day),
                        symbols=symbols,
                        summarize_missing_lsa=bool(int(args.summarize_missing_lsa)),
                    )
                elif text_zip.exists():
                    _build_news_daily_sqlite_from_zip(
                        zip_path=text_zip,
                        db_path=db_path,
                        finbert_model=str(finbert_model),
                        finbert_device=int(args.finbert_device),
                        finbert_max_length=int(args.finbert_max_length),
                        finbert_batch_size=int(args.finbert_batch_size),
                        top_k_per_day=int(args.top_k_per_day),
                        symbols=symbols,
                        summarize_missing_lsa=bool(int(args.summarize_missing_lsa)),
                    )
                else:
                    raise FileNotFoundError(f"missing sp500 news: {text_dir} or {text_zip}")
            else:
                text_zip = dataset_dir / "text" / "hs300news_summary.zip"
                if not text_zip.exists():
                    raise FileNotFoundError(f"missing hs300 news zip: {text_zip}")
                _build_news_daily_sqlite_from_zip(
                    zip_path=text_zip,
                    db_path=db_path,
                    finbert_model=str(finbert_model),
                    finbert_device=int(args.finbert_device),
                    finbert_max_length=int(args.finbert_max_length),
                    finbert_batch_size=int(args.finbert_batch_size),
                    top_k_per_day=int(args.top_k_per_day),
                    symbols=symbols,
                    summarize_missing_lsa=bool(int(args.summarize_missing_lsa)),
                )

        if int(args.integrate) == 1:
            news_db = out_mk / "news_daily_finbert.sqlite"
            integrate_price_with_news(
                market=mk,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                news_db_path=news_db,
                decay_rate=float(args.decay_rate),
                symbols=symbols,
                save_csv=bool(int(args.save_csv)),
                save_npz=bool(int(args.save_npz)),
            )

        integrated_dir = output_dir / mk / "integrated"
        if int(args.export_shards) == 1:
            export_sequences_npz_sharded(
                integrated_dir=integrated_dir,
                output_dir=output_dir / mk / "finmultitime_shards",
                dataset_name=f"finmultitime_{mk}",
                symbols=symbols,
                seq_len=int(args.seq_len),
                horizon=int(args.horizon),
                vol_window=int(args.vol_window),
                vol_threshold=float(args.vol_threshold),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                normalize_numeric=bool(int(args.normalize_numeric)),
                shard_max_train_samples=int(args.shard_max_train_samples),
                manifest_path=output_dir / mk / "finmultitime_shards_manifest.json",
            )

        if int(args.export_single) == 1:
            export_sequences_npz_single(
                integrated_dir=integrated_dir,
                output_npz=output_dir / mk / "finmultitime_processed.npz",
                symbols=symbols,
                seq_len=int(args.seq_len),
                horizon=int(args.horizon),
                vol_window=int(args.vol_window),
                vol_threshold=float(args.vol_threshold),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                normalize_numeric=bool(int(args.normalize_numeric)),
            )


if __name__ == "__main__":
    main()
