"""
FNSPID (HuggingFace dump) 预处理脚本

输入（默认）:
- datasets/FNSPID-hf/Stock_news/All_external.csv
- datasets/FNSPID-hf/Stock_news/nasdaq_exteral_data.csv
- datasets/FNSPID-hf/Stock_price/full_history.zip (zip 内含 full_history/<SYMBOL>.csv)

输出（默认）:
- data/fnspid_hf/news_daily.sqlite (按 symbol+date 聚合的新闻统计)
- data/fnspid_hf/integrated/<SYMBOL>.csv (按交易日对齐后的 price + sentiment)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypedDict
from zipfile import ZipFile

import numpy as np
import pandas as pd


def _set_csv_field_size_limit() -> None:
    try:
        limit = int(sys.maxsize)
    except Exception:
        limit = 2**31 - 1
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


@dataclass
class _ProgressBar:
    desc: str
    total: Optional[int] = None
    width: int = 28
    min_interval_s: float = 0.4

    last_ts: float = 0.0
    start_ts: float = 0.0
    done: int = 0

    def __post_init__(self) -> None:
        now = time.time()
        self.start_ts = now
        self.last_ts = 0.0

    def _render(self, frac: Optional[float], suffix: str) -> str:
        if frac is None:
            bar = "." * self.width
            pct = " ??.?%"
        else:
            f = min(1.0, max(0.0, float(frac)))
            filled = int(round(f * self.width))
            bar = "=" * filled + "." * (self.width - filled)
            pct = f"{(f * 100.0):6.2f}%"
        return f"\r{self.desc} [{bar}] {pct}{suffix}"

    def update_abs(self, done: int, total: Optional[int] = None, suffix: str = "") -> None:
        if total is not None:
            self.total = int(total)
        self.done = int(done)
        now = time.time()
        if self.last_ts and (now - self.last_ts) < float(self.min_interval_s):
            return
        self.last_ts = now
        frac = None
        if self.total and self.total > 0:
            frac = float(self.done) / float(self.total)
        line = self._render(frac, suffix)
        sys.stderr.write(line)
        sys.stderr.flush()

    def close(self, suffix: str = "") -> None:
        frac = None
        if self.total and self.total > 0:
            frac = 1.0
        line = self._render(frac, suffix)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()


_POS_WORDS = {
    "beat",
    "beats",
    "benefit",
    "bull",
    "bullish",
    "grow",
    "grows",
    "growth",
    "higher",
    "improve",
    "improves",
    "improving",
    "increase",
    "increases",
    "increased",
    "jump",
    "jumps",
    "outperform",
    "outperforms",
    "profit",
    "profits",
    "rally",
    "record",
    "rise",
    "rises",
    "rose",
    "strong",
    "surge",
    "surges",
    "upgrade",
    "upgrades",
    "upgraded",
    "win",
    "wins",
}

_NEG_WORDS = {
    "bear",
    "bearish",
    "cut",
    "cuts",
    "cutting",
    "decline",
    "declines",
    "declined",
    "downgrade",
    "downgrades",
    "downgraded",
    "drop",
    "drops",
    "dropped",
    "fall",
    "falls",
    "fell",
    "lower",
    "miss",
    "misses",
    "missed",
    "plunge",
    "plunges",
    "risk",
    "risks",
    "slip",
    "slips",
    "slumped",
    "weak",
    "warning",
    "lawsuit",
    "probe",
    "investigation",
}


def _tokenize(text: str) -> List[str]:
    buf: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if "a" <= ch <= "z":
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf


def _lexicon_sentiment_1to5(text: str) -> float:
    if not text:
        return 3.0
    toks = _tokenize(text)
    if not toks:
        return 3.0
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    denom = max(1, pos + neg)
    score = (pos - neg) / float(denom)
    out = 3.0 + 2.0 * score
    if out < 1.0:
        out = 1.0
    if out > 5.0:
        out = 5.0
    return out


def _load_wordlist(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().lower()
            if not w:
                continue
            if w.startswith("#"):
                continue
            if "," in w:
                w = w.split(",", 1)[0].strip()
            if w:
                out.add(w)
    return out


_NEGATIONS = {
    "not",
    "no",
    "never",
    "none",
    "cannot",
    "cant",
    "won't",
    "wont",
    "without",
}


def _lm_lexicon_sentiment_1to5(
    text: str,
    pos_words: set[str],
    neg_words: set[str],
    negation_window: int = 3,
) -> float:
    if not text:
        return 3.0
    toks = _tokenize(text)
    if not toks:
        return 3.0

    pos = 0
    neg = 0
    for i, t in enumerate(toks):
        if t not in pos_words and t not in neg_words:
            continue
        negated = False
        start = max(0, i - int(negation_window))
        for j in range(start, i):
            if toks[j] in _NEGATIONS:
                negated = True
                break
        if t in pos_words:
            if negated:
                neg += 1
            else:
                pos += 1
        if t in neg_words:
            if negated:
                pos += 1
            else:
                neg += 1

    denom = max(1, pos + neg)
    score = (pos - neg) / float(denom)
    out = 3.0 + 2.0 * score
    if out < 1.0:
        out = 1.0
    if out > 5.0:
        out = 5.0
    return out


def _parse_utc_datetime_to_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith(" UTC"):
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S UTC")
            return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    try:
        dt2 = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt2):
            return None
        return pd.Timestamp(dt2).strftime("%Y-%m-%d")
    except Exception:
        return None


def _iter_news_rows(csv_fp: Path) -> Iterator[Dict[str, str]]:
    with csv_fp.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                yield row


def _open_db(db_fp: Path) -> sqlite3.Connection:
    db_fp.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_fp))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_daily (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            news_count INTEGER NOT NULL,
            sentiment_sum REAL NOT NULL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news_daily(symbol, date)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gpt_cache (
            cache_key TEXT PRIMARY KEY,
            sentiment REAL NOT NULL,
            model TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gpt_cache_model ON gpt_cache(model)")
    return conn


def _flush_agg(conn: sqlite3.Connection, agg: Dict[Tuple[str, str], Tuple[int, float]]) -> None:
    if not agg:
        return
    rows = [(sym, d, int(cnt), float(ssum)) for (sym, d), (cnt, ssum) in agg.items()]
    conn.executemany(
        """
        INSERT INTO news_daily(symbol, date, news_count, sentiment_sum)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            news_count = news_count + excluded.news_count,
            sentiment_sum = sentiment_sum + excluded.sentiment_sum
        """,
        rows,
    )
    conn.commit()
    agg.clear()


@dataclass(frozen=True)
class BuildIndexConfig:
    flush_n: int = 200_000
    gpt_batch_size: int = 5
    gpt_sleep_seconds: float = 1.0


class _GptItem(TypedDict):
    cache_key: str
    symbol: str
    text: str


def _ensure_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    cur = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = cur.fetchone()
    if row is None:
        conn.execute("INSERT INTO meta(key, value) VALUES(?, ?)", (key, value))
        conn.commit()
        return
    if str(row[0]) != str(value):
        raise ValueError(f"meta mismatch {key}: db={row[0]} arg={value}")


def _get_gpt_cache(conn: sqlite3.Connection, cache_key: str) -> Optional[float]:
    cur = conn.execute("SELECT sentiment FROM gpt_cache WHERE cache_key = ?", (cache_key,))
    row = cur.fetchone()
    if row is None:
        return None
    try:
        return float(row[0])
    except Exception:
        return None


def _set_gpt_cache(conn: sqlite3.Connection, cache_key: str, sentiment: float, model: str) -> None:
    conn.execute(
        """
        INSERT INTO gpt_cache(cache_key, sentiment, model, updated_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            sentiment = excluded.sentiment,
            model = excluded.model,
            updated_at = excluded.updated_at
        """,
        (cache_key, float(sentiment), str(model), int(time.time())),
    )


def _make_cache_key(symbol: str, date: str, url: str, title: str) -> str:
    base = f"{symbol}|{date}|{url.strip()}|{title.strip()}".encode("utf-8", errors="ignore")
    return hashlib.sha1(base).hexdigest()


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

        msg = "需要安装 transformers 和 torch 才能使用 --sentiment_method finbert"
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
                print("[FNSPID-HF] finbert_device 指定 GPU 但当前无可用 CUDA，已自动回退到 CPU")
        except Exception as e:
            print(f"[FNSPID-HF] CUDA 初始化失败，已自动回退到 CPU: {type(e).__name__}: {e}")
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

    sentences = [s.strip() for s in parts if len(s.strip()) >= 20]
    if not sentences:
        return text

    kw = (symbol or "").strip().upper()
    scored: List[Tuple[float, int, str]] = []
    for i, s in enumerate(sentences):
        up = s.upper()
        has_kw = 1.0 if kw and kw in up else 0.0
        length = min(400.0, float(len(s)))
        score = has_kw * 1.0 + (length / 400.0) * 0.2
        scored.append((score, i, s))

    top = sorted(scored, key=lambda x: (x[0], -x[1]), reverse=True)[: int(num_sentences)]
    top_sorted = sorted(top, key=lambda x: x[1])
    out = " ".join(s for _sc, _i, s in top_sorted).strip()
    return out or text


def _lsa_summary_with_keywords(text: str, symbol: str, num_sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    imported = _try_import_sumy()
    if imported is None:
        return _fallback_summary_with_keywords(text=text, symbol=symbol, num_sentences=int(num_sentences))

    Stemmer, Tokenizer, PlaintextParser, LsaSummarizer, get_stop_words = imported
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    tokenizer = Tokenizer("english")
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
        weights[s] += 0.0

    for s in initial_summary:
        weights[s] += 1.0

    ranked = sorted(weights.keys(), key=lambda x: weights[x], reverse=True)[: int(num_sentences)]
    out = " ".join(str(s) for s in ranked).strip()
    return out or " ".join(str(s) for s in initial_summary).strip() or text


def _pick_text_for_scoring(row: Dict[str, str], symbol: str, summarize_missing_lsa: bool) -> str:
    title = (row.get("Article_title") or row.get("article_title") or "").strip()
    lsa = (row.get("Lsa_summary") or row.get("lsa_summary") or "").strip()
    luhn = (row.get("Luhn_summary") or row.get("luhn_summary") or "").strip()
    tx = (row.get("Textrank_summary") or row.get("textrank_summary") or "").strip()
    lx = (row.get("Lexrank_summary") or row.get("lexrank_summary") or "").strip()
    article = (row.get("Article") or row.get("article") or "").strip()

    if lx:
        return lx
    if tx:
        return tx
    if lsa:
        return lsa
    if luhn:
        return luhn

    if summarize_missing_lsa and article:
        return _lsa_summary_with_keywords(article, symbol=symbol, num_sentences=3)

    return title or article


def _get_openai_client(api_key: Optional[str], base_url: Optional[str]):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("需要安装 openai>=1.x 才能使用 --sentiment_method gpt") from e
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("未提供 OpenAI API Key：请设置环境变量 OPENAI_API_KEY 或传入 --openai_api_key")
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)


def _gpt_score_batch(
    client,
    model: str,
    items: List[_GptItem],
    max_retries: int,
    backoff_seconds: float,
    backoff_max_seconds: float,
) -> List[Optional[float]]:
    texts = []
    for it in items:
        t = (it["text"] or "").strip()
        if not t:
            t = "N/A"
        texts.append(f"### News to Stock Symbol -- {it['symbol']}: {t}")
    text_content = " ".join(texts)

    conversation = [
        {
            "role": "system",
            "content": (
                "Forget all your previous instructions. You are a financial expert with stock recommendation experience. "
                "Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, "
                "3 is neutral, 4 is somewhat positive, 5 is positive. "
                f"{len(items)} summerized news will be passed in each time, you will give score in format as shown below in the response from assistant."
            ),
        },
        {
            "role": "user",
            "content": (
                "News to Stock Symbol -- AAPL: Apple (AAPL) increase 22% "
                "### News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30% "
                "### News to Stock Symbol -- MSFT: Microsoft (MSTF) price has no change"
            ),
        },
        {"role": "assistant", "content": "5, 1, 3"},
        {
            "role": "user",
            "content": (
                "News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15 "
                "### News to Stock Symbol -- AAPL: Apple (AAPL) will release VisonPro on Feb 2, 2024"
            ),
        },
        {"role": "assistant", "content": "4, 4"},
        {"role": "user", "content": text_content},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(max_retries))):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=conversation,
                temperature=0,
                max_tokens=50,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            sleep_s = min(float(backoff_max_seconds), float(backoff_seconds) * (2.0 ** attempt))
            time.sleep(max(0.0, float(sleep_s)))

    if last_err is not None:
        raise last_err

    content = resp.choices[0].message.content or ""
    parts = [p.strip() for p in content.split(",") if p.strip()]
    out: List[Optional[float]] = []
    for p in parts[: len(items)]:
        try:
            v = float(int(p))
        except Exception:
            v = float("nan")
        out.append(v)
    while len(out) < len(items):
        out.append(float("nan"))
    return out


def build_news_daily_index(
    news_csvs: List[Path],
    db_fp: Path,
    cfg: BuildIndexConfig,
    symbols: Optional[List[str]],
    sentiment_method: str,
    summarize_missing_lsa: bool,
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    gpt_model: str,
    gpt_max_retries: int,
    gpt_backoff_seconds: float,
    gpt_backoff_max_seconds: float,
    gpt_on_failure: str,
    lm_positive_path: Optional[str],
    lm_negative_path: Optional[str],
    finbert_model: Optional[str],
    finbert_batch_size: int,
    finbert_max_length: int,
    finbert_device: int,
) -> None:
    _set_csv_field_size_limit()
    conn = _open_db(db_fp)
    _ensure_meta(conn, "sentiment_method", sentiment_method)
    _ensure_meta(conn, "summarize_missing_lsa", "1" if summarize_missing_lsa else "0")
    if sentiment_method == "gpt":
        _ensure_meta(conn, "gpt_model", gpt_model)
    if sentiment_method == "lm_lexicon":
        _ensure_meta(conn, "lm_positive_path", str(lm_positive_path or ""))
        _ensure_meta(conn, "lm_negative_path", str(lm_negative_path or ""))
    if sentiment_method == "finbert":
        _ensure_meta(conn, "finbert_model", str(finbert_model or ""))
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None

    client = None
    if sentiment_method == "gpt":
        client = _get_openai_client(api_key=openai_api_key, base_url=openai_base_url)

    lm_pos: Optional[set[str]] = None
    lm_neg: Optional[set[str]] = None
    if sentiment_method == "lm_lexicon":
        if not lm_positive_path or not lm_negative_path:
            raise ValueError("--sentiment_method lm_lexicon 需要同时提供 --lm_positive_path 和 --lm_negative_path")
        lm_pos = _load_wordlist(Path(lm_positive_path))
        lm_neg = _load_wordlist(Path(lm_negative_path))

    fin_pipe = None
    if sentiment_method == "finbert":
        model_name = (finbert_model or "").strip()
        if not model_name:
            raise ValueError("--sentiment_method finbert 需要提供 --finbert_model（可以是 HuggingFace 模型名或本地路径）")
        fin_pipe = _build_finbert_scorer(
            model_name=model_name,
            finbert_device=int(finbert_device),
            finbert_max_length=int(finbert_max_length),
        )

    agg: Dict[Tuple[str, str], Tuple[int, float]] = {}
    n_rows = 0
    gpt_pending: List[_GptItem] = []
    pending_meta: List[Tuple[str, str]] = []
    fin_pending_texts: List[str] = []
    fin_pending_meta: List[Tuple[str, str]] = []

    for fp in news_csvs:
        file_size = None
        try:
            file_size = fp.stat().st_size
        except Exception:
            file_size = None
        pbar = _ProgressBar(desc=f"[FNSPID-HF] index {fp.name}", total=file_size)
        rows_in_file = 0
        with fp.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                rows_in_file += 1
                if rows_in_file % 20000 == 0:
                    try:
                        pbar.update_abs(done=int(f.tell()), total=file_size, suffix=f" rows={rows_in_file}")
                    except Exception:
                        pbar.update_abs(done=rows_in_file, total=None, suffix=f" rows={rows_in_file}")
            sym = (row.get("Stock_symbol") or row.get("stock_symbol") or "").strip().upper()
            if not sym:
                continue
            if wanted is not None and sym not in wanted:
                continue
            date_raw = row.get("Date") or row.get("date") or ""
            d = _parse_utc_datetime_to_date(date_raw)
            if not d:
                continue

            title = (row.get("Article_title") or row.get("article_title") or "").strip()
            url = (row.get("Url") or row.get("url") or "").strip()
            cache_key = _make_cache_key(sym, d, url, title)

            text = _pick_text_for_scoring(row=row, symbol=sym, summarize_missing_lsa=summarize_missing_lsa)
            if sentiment_method == "lexicon":
                sentiment = float(_lexicon_sentiment_1to5(text))
                key = (sym, d)
                prev = agg.get(key)
                if prev is None:
                    agg[key] = (1, sentiment)
                else:
                    agg[key] = (prev[0] + 1, prev[1] + sentiment)
            elif sentiment_method == "lm_lexicon":
                if lm_pos is None or lm_neg is None:
                    raise ValueError("lm lexicon is not initialized")
                sentiment = float(_lm_lexicon_sentiment_1to5(text, pos_words=lm_pos, neg_words=lm_neg))
                key = (sym, d)
                prev = agg.get(key)
                if prev is None:
                    agg[key] = (1, sentiment)
                else:
                    agg[key] = (prev[0] + 1, prev[1] + sentiment)
            elif sentiment_method == "finbert":
                if fin_pipe is None:
                    raise ValueError("finbert pipeline is not initialized")
                fin_pending_texts.append(text)
                fin_pending_meta.append((sym, d))
                if len(fin_pending_texts) >= int(finbert_batch_size):
                    scores = _finbert_sentiment_1to5(
                        scorer=fin_pipe,
                        texts=fin_pending_texts,
                        batch_size=int(finbert_batch_size),
                    )
                    for (sym2, d2), s in zip(fin_pending_meta, scores):
                        key = (sym2, d2)
                        prev = agg.get(key)
                        if prev is None:
                            agg[key] = (1, float(s))
                        else:
                            agg[key] = (prev[0] + 1, prev[1] + float(s))
                    fin_pending_texts.clear()
                    fin_pending_meta.clear()
            elif sentiment_method == "gpt":
                cached = _get_gpt_cache(conn, cache_key)
                if cached is not None and not math.isnan(cached):
                    sentiment = float(cached)
                    key = (sym, d)
                    prev = agg.get(key)
                    if prev is None:
                        agg[key] = (1, sentiment)
                    else:
                        agg[key] = (prev[0] + 1, prev[1] + sentiment)
                else:
                    gpt_pending.append({"cache_key": cache_key, "symbol": sym, "text": text})
                    pending_meta.append((sym, d))
                    if len(gpt_pending) >= int(cfg.gpt_batch_size):
                        if client is None:
                            raise ValueError("gpt client is not initialized")
                        try:
                            scores = _gpt_score_batch(
                                client=client,
                                model=gpt_model,
                                items=gpt_pending,
                                max_retries=int(gpt_max_retries),
                                backoff_seconds=float(gpt_backoff_seconds),
                                backoff_max_seconds=float(gpt_backoff_max_seconds),
                            )
                            for (sym2, d2), it, s in zip(pending_meta, gpt_pending, scores):
                                val = float(s) if s is not None and not math.isnan(float(s)) else 3.0
                                val = float(min(5.0, max(1.0, val)))
                                _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                                key = (sym2, d2)
                                prev = agg.get(key)
                                if prev is None:
                                    agg[key] = (1, val)
                                else:
                                    agg[key] = (prev[0] + 1, prev[1] + val)
                            conn.commit()
                        except Exception as e:
                            if gpt_on_failure == "raise":
                                raise
                            if gpt_on_failure == "neutral":
                                for (sym2, d2), it in zip(pending_meta, gpt_pending):
                                    val = 3.0
                                    _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                                    key = (sym2, d2)
                                    prev = agg.get(key)
                                    if prev is None:
                                        agg[key] = (1, val)
                                    else:
                                        agg[key] = (prev[0] + 1, prev[1] + val)
                                conn.commit()
                            elif gpt_on_failure == "lexicon":
                                for (sym2, d2), it in zip(pending_meta, gpt_pending):
                                    val = float(_lexicon_sentiment_1to5(it["text"]))
                                    _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                                    key = (sym2, d2)
                                    prev = agg.get(key)
                                    if prev is None:
                                        agg[key] = (1, val)
                                    else:
                                        agg[key] = (prev[0] + 1, prev[1] + val)
                                conn.commit()
                            else:
                                raise ValueError(f"Unknown gpt_on_failure: {gpt_on_failure}") from e
                        gpt_pending.clear()
                        pending_meta.clear()
                        time.sleep(float(cfg.gpt_sleep_seconds))
            else:
                raise ValueError(f"Unknown sentiment_method: {sentiment_method}")

            n_rows += 1
            if n_rows % cfg.flush_n == 0:
                if sentiment_method == "finbert" and fin_pending_texts:
                    if fin_pipe is None:
                        raise ValueError("finbert pipeline is not initialized")
                    scores = _finbert_sentiment_1to5(
                        scorer=fin_pipe,
                        texts=fin_pending_texts,
                        batch_size=int(finbert_batch_size),
                    )
                    for (sym2, d2), s in zip(fin_pending_meta, scores):
                        key = (sym2, d2)
                        prev = agg.get(key)
                        if prev is None:
                            agg[key] = (1, float(s))
                        else:
                            agg[key] = (prev[0] + 1, prev[1] + float(s))
                    fin_pending_texts.clear()
                    fin_pending_meta.clear()
                if sentiment_method == "gpt" and gpt_pending:
                    if client is None:
                        raise ValueError("gpt client is not initialized")
                    try:
                        scores = _gpt_score_batch(
                            client=client,
                            model=gpt_model,
                            items=gpt_pending,
                            max_retries=int(gpt_max_retries),
                            backoff_seconds=float(gpt_backoff_seconds),
                            backoff_max_seconds=float(gpt_backoff_max_seconds),
                        )
                        for (sym2, d2), it, s in zip(pending_meta, gpt_pending, scores):
                            val = float(s) if s is not None and not math.isnan(float(s)) else 3.0
                            val = float(min(5.0, max(1.0, val)))
                            _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                            key = (sym2, d2)
                            prev = agg.get(key)
                            if prev is None:
                                agg[key] = (1, val)
                            else:
                                agg[key] = (prev[0] + 1, prev[1] + val)
                        conn.commit()
                    except Exception as e:
                        if gpt_on_failure == "raise":
                            raise
                        if gpt_on_failure == "neutral":
                            for (sym2, d2), it in zip(pending_meta, gpt_pending):
                                val = 3.0
                                _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                                key = (sym2, d2)
                                prev = agg.get(key)
                                if prev is None:
                                    agg[key] = (1, val)
                                else:
                                    agg[key] = (prev[0] + 1, prev[1] + val)
                            conn.commit()
                        elif gpt_on_failure == "lexicon":
                            for (sym2, d2), it in zip(pending_meta, gpt_pending):
                                val = float(_lexicon_sentiment_1to5(it["text"]))
                                _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                                key = (sym2, d2)
                                prev = agg.get(key)
                                if prev is None:
                                    agg[key] = (1, val)
                                else:
                                    agg[key] = (prev[0] + 1, prev[1] + val)
                            conn.commit()
                        else:
                            raise ValueError(f"Unknown gpt_on_failure: {gpt_on_failure}") from e
                    gpt_pending.clear()
                    pending_meta.clear()
                    time.sleep(float(cfg.gpt_sleep_seconds))
                _flush_agg(conn, agg)
                print(f"[FNSPID-HF] indexed rows={n_rows} -> {db_fp.name}")
        pbar.close(suffix=f" rows={rows_in_file}")

    if sentiment_method == "finbert" and fin_pending_texts:
        if fin_pipe is None:
            raise ValueError("finbert pipeline is not initialized")
        scores = _finbert_sentiment_1to5(
            scorer=fin_pipe,
            texts=fin_pending_texts,
            batch_size=int(finbert_batch_size),
        )
        for (sym2, d2), s in zip(fin_pending_meta, scores):
            key = (sym2, d2)
            prev = agg.get(key)
            if prev is None:
                agg[key] = (1, float(s))
            else:
                agg[key] = (prev[0] + 1, prev[1] + float(s))
        fin_pending_texts.clear()
        fin_pending_meta.clear()

    if sentiment_method == "gpt" and gpt_pending:
        if client is None:
            raise ValueError("gpt client is not initialized")
        try:
            scores = _gpt_score_batch(
                client=client,
                model=gpt_model,
                items=gpt_pending,
                max_retries=int(gpt_max_retries),
                backoff_seconds=float(gpt_backoff_seconds),
                backoff_max_seconds=float(gpt_backoff_max_seconds),
            )
            for (sym2, d2), it, s in zip(pending_meta, gpt_pending, scores):
                val = float(s) if s is not None and not math.isnan(float(s)) else 3.0
                val = float(min(5.0, max(1.0, val)))
                _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                key = (sym2, d2)
                prev = agg.get(key)
                if prev is None:
                    agg[key] = (1, val)
                else:
                    agg[key] = (prev[0] + 1, prev[1] + val)
            conn.commit()
        except Exception as e:
            if gpt_on_failure == "raise":
                raise
            if gpt_on_failure == "neutral":
                for (sym2, d2), it in zip(pending_meta, gpt_pending):
                    val = 3.0
                    _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                    key = (sym2, d2)
                    prev = agg.get(key)
                    if prev is None:
                        agg[key] = (1, val)
                    else:
                        agg[key] = (prev[0] + 1, prev[1] + val)
                conn.commit()
            elif gpt_on_failure == "lexicon":
                for (sym2, d2), it in zip(pending_meta, gpt_pending):
                    val = float(_lexicon_sentiment_1to5(it["text"]))
                    _set_gpt_cache(conn, it["cache_key"], val, gpt_model)
                    key = (sym2, d2)
                    prev = agg.get(key)
                    if prev is None:
                        agg[key] = (1, val)
                    else:
                        agg[key] = (prev[0] + 1, prev[1] + val)
                conn.commit()
            else:
                raise ValueError(f"Unknown gpt_on_failure: {gpt_on_failure}") from e
        gpt_pending.clear()
        pending_meta.clear()

    _flush_agg(conn, agg)
    conn.close()
    print(f"[FNSPID-HF] index done rows={n_rows} -> {db_fp}")


def _list_price_members(zip_fp: Path) -> List[str]:
    with ZipFile(zip_fp, "r") as zf:
        names = [n for n in zf.namelist() if n.startswith("full_history/") and n.endswith(".csv")]
    names = [n for n in names if "/__MACOSX/" not in n and not n.startswith("__MACOSX/")]
    names.sort()
    return names


def _read_price_csv_from_zip(zip_fp: Path, member: str) -> pd.DataFrame:
    with ZipFile(zip_fp, "r") as zf:
        with zf.open(member, "r") as f:
            df = pd.read_csv(f)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("price csv missing date")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _list_price_files(price_dir: Path) -> List[Path]:
    if not price_dir.exists():
        return []
    files = [p for p in price_dir.glob("*.csv") if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def _read_price_csv_from_file(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("price csv missing date")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _load_news_daily_for_symbol(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    cur = conn.execute(
        "SELECT date, news_count, sentiment_sum FROM news_daily WHERE symbol = ? ORDER BY date ASC",
        (symbol,),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "news_count", "sentiment_sum"])
    df = pd.DataFrame(rows, columns=["date", "news_count", "sentiment_sum"])
    df["date"] = df["date"].astype(str)
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(np.int64)
    df["sentiment_sum"] = pd.to_numeric(df["sentiment_sum"], errors="coerce").fillna(0.0).astype(np.float64)
    return df


def _apply_exponential_decay_to_neutral(
    dates: List[datetime],
    sentiments: List[Optional[float]],
    flags: List[int],
    decay_rate: float,
    neutral: float = 3.0,
) -> List[float]:
    out: List[float] = []
    last_sent: Optional[float] = None
    last_date: Optional[datetime] = None
    for d, s, f in zip(dates, sentiments, flags):
        if f == 1 and s is not None and not (isinstance(s, float) and math.isnan(s)):
            val = float(s)
            last_sent = val
            last_date = d
            out.append(val)
            continue
        if last_sent is None or last_date is None:
            out.append(float(neutral))
            continue
        delta = (d - last_date).days
        val2 = float(neutral) + (float(last_sent) - float(neutral)) * math.exp(-float(decay_rate) * float(delta))
        out.append(val2)
    return out


def integrate_price_with_news(
    price_zip_fp: Optional[Path],
    price_dir: Optional[Path],
    db_fp: Path,
    output_dir: Path,
    save_csv: bool,
    save_npz: bool,
    decay_rate: float,
    min_price_rows: int,
    symbols: Optional[List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None

    members: List[str] = []
    files: List[Path] = []
    if price_dir is not None:
        files = _list_price_files(price_dir)
        if wanted is not None:
            files = [p for p in files if p.stem.upper() in wanted]
    if not files:
        if price_zip_fp is None or (not price_zip_fp.exists()):
            raise FileNotFoundError("未找到可用的 price 数据：full_history/ 或 full_history.zip")
        members = _list_price_members(price_zip_fp)
        if wanted is not None:
            members = [m for m in members if Path(m).stem.upper() in wanted]

    conn = sqlite3.connect(str(db_fp))
    ok = 0
    skipped = 0

    it: Iterable[Tuple[str, Optional[str], Optional[Path]]] = []
    if files:
        it = [(p.stem.upper(), None, p) for p in files]
    else:
        it = [(Path(m).stem.upper(), m, None) for m in members]

    total_items = len(it) if isinstance(it, list) else (len(files) if files else len(members))
    pbar = _ProgressBar(desc="[FNSPID-HF] integrate", total=total_items)
    processed = 0
    for symbol, member, fp in it:
        try:
            if fp is not None:
                px = _read_price_csv_from_file(fp)
            else:
                if price_zip_fp is None:
                    raise FileNotFoundError("price_zip_fp is None")
                px = _read_price_csv_from_zip(price_zip_fp, str(member))
        except Exception as e:
            print(f"[FNSPID-HF] price failed {symbol}: {type(e).__name__}: {e}")
            continue

        if int(min_price_rows) > 0 and len(px) < int(min_price_rows):
            skipped += 1
            continue

        news = _load_news_daily_for_symbol(conn, symbol)
        merged = px.merge(news, on="date", how="left")
        merged["news_count"] = pd.to_numeric(merged["news_count"], errors="coerce").fillna(0).astype(np.int64)
        merged["sentiment_sum"] = pd.to_numeric(merged["sentiment_sum"], errors="coerce").fillna(0.0).astype(np.float64)

        news_flag = (merged["news_count"] > 0).astype(np.int8)
        merged["news_flag"] = news_flag
        sentiment_mean = np.where(
            merged["news_count"].to_numpy() > 0,
            merged["sentiment_sum"].to_numpy() / np.maximum(1, merged["news_count"].to_numpy()),
            np.nan,
        ).astype(np.float64)
        merged["sentiment"] = sentiment_mean

        dt_list = [datetime.strptime(d, "%Y-%m-%d") for d in merged["date"].tolist()]
        sent_list: List[Optional[float]] = [None if np.isnan(x) else float(x) for x in merged["sentiment"].to_numpy()]
        filled = _apply_exponential_decay_to_neutral(
            dates=dt_list,
            sentiments=sent_list,
            flags=merged["news_flag"].to_numpy(dtype=np.int8).tolist(),
            decay_rate=float(decay_rate),
            neutral=3.0,
        )
        filled_arr = np.asarray(filled, dtype=np.float64)
        filled_arr = np.clip(filled_arr, 1.0, 5.0)
        merged["sentiment"] = filled_arr
        merged["scaled_sentiment"] = (merged["sentiment"].astype(np.float64) - 0.9999) / 4.0

        if save_csv:
            out_fp = output_dir / f"{symbol}.csv"
            merged.to_csv(out_fp, index=False)
        if save_npz:
            feature_cols = [c for c in merged.columns if c != "date"]
            data = merged[feature_cols].to_numpy(dtype=np.float32, copy=False)
            dates = merged["date"].astype(str).to_numpy()
            np.savez_compressed(
                str(output_dir / f"{symbol}.npz"),
                data=data,
                dates=dates,
                feature_cols=np.asarray(feature_cols, dtype=object),
            )
        ok += 1
        processed += 1
        pbar.update_abs(done=processed, total=total_items, suffix=f" ok={ok} skipped={skipped} last={symbol}")
        if ok % 50 == 0:
            print(f"[FNSPID-HF] integrated ok={ok} skipped={skipped} last={symbol}")

    conn.close()
    pbar.close(suffix=f" ok={ok} skipped={skipped}")
    print(f"[FNSPID-HF] integrate done ok={ok} skipped={skipped} -> {output_dir}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _add_fnspid_labels(
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


def _split_time_series(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * float(train_ratio))
    val_end = int(n * float(train_ratio + val_ratio))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def _normalize_with_train_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    cols = list(cols)
    train_values = train_df[cols].astype(np.float32)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0, ddof=0).replace(0.0, 1.0)

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    train_out[cols] = (train_out[cols].astype(np.float32) - mean) / std
    val_out[cols] = (val_out[cols].astype(np.float32) - mean) / std
    test_out[cols] = (test_out[cols].astype(np.float32) - mean) / std
    scaler = {"mean": mean.to_numpy(dtype=np.float32), "std": std.to_numpy(dtype=np.float32)}
    return train_out, val_out, test_out, scaler


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


def export_fnspid_sequences_npz(
    integrated_dir: Path,
    output_npz: Path,
    symbols: Optional[List[str]],
    seq_len: int,
    horizon: int,
    vol_window: int,
    vol_threshold: float,
    train_ratio: float,
    val_ratio: float,
    normalize_price: bool,
) -> None:
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    csv_files = sorted([p for p in integrated_dir.glob("*.csv") if p.is_file()])
    if wanted is not None:
        csv_files = [p for p in csv_files if p.stem.upper() in wanted]

    feature_cols = ["open", "high", "low", "close", "volume", "scaled_sentiment", "news_flag"]
    price_cols = ["open", "high", "low", "close", "volume"]

    X_train_list: List[np.ndarray] = []
    y_train_cls_list: List[np.ndarray] = []
    y_train_reg_list: List[np.ndarray] = []
    X_val_list: List[np.ndarray] = []
    y_val_cls_list: List[np.ndarray] = []
    y_val_reg_list: List[np.ndarray] = []
    X_test_list: List[np.ndarray] = []
    y_test_cls_list: List[np.ndarray] = []
    y_test_reg_list: List[np.ndarray] = []

    pbar = _ProgressBar(desc="[FNSPID-HF] sequences", total=len(csv_files))
    processed = 0
    kept_symbols = 0

    for fp in csv_files:
        processed += 1
        symbol = fp.stem.upper()
        try:
            df = pd.read_csv(fp)
        except Exception:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} failed_read")
            continue
        df = _standardize_columns(df)
        if "date" not in df.columns:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} no_date")
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        if "scaled_sentiment" not in df.columns:
            if "sentiment" in df.columns:
                df["scaled_sentiment"] = (pd.to_numeric(df["sentiment"], errors="coerce").fillna(3.0) - 0.9999) / 4.0
            else:
                df["scaled_sentiment"] = (3.0 - 0.9999) / 4.0
        df["scaled_sentiment"] = pd.to_numeric(df["scaled_sentiment"], errors="coerce").fillna((3.0 - 0.9999) / 4.0).astype(np.float32)

        if "news_flag" not in df.columns:
            if "news_count" in df.columns:
                df["news_flag"] = (pd.to_numeric(df["news_count"], errors="coerce").fillna(0) > 0).astype(np.int8)
            else:
                df["news_flag"] = 0
        df["news_flag"] = pd.to_numeric(df["news_flag"], errors="coerce").fillna(0).astype(np.int8)

        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} missing_{c}")
                df = pd.DataFrame()
                break
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        if df.empty:
            continue
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        df = _add_fnspid_labels(
            df=df,
            horizon=int(horizon),
            vol_window=int(vol_window),
            vol_threshold=float(vol_threshold),
        )
        if len(df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} too_short")
            continue

        train_df, val_df, test_df = _split_time_series(df, train_ratio=float(train_ratio), val_ratio=float(val_ratio))
        if len(train_df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} train_short")
            continue

        if normalize_price:
            train_df, val_df, test_df, _ = _normalize_with_train_stats(train_df, val_df, test_df, cols=price_cols)

        X_tr, y_tr_c, y_tr_r = _create_sequences(train_df, feature_cols, seq_len=int(seq_len))
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

        if len(X_tr) == 0:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} no_seq")
            continue

        X_train_list.append(X_tr)
        y_train_cls_list.append(y_tr_c)
        y_train_reg_list.append(y_tr_r)
        X_val_list.append(X_va)
        y_val_cls_list.append(y_va_c)
        y_val_reg_list.append(y_va_r)
        X_test_list.append(X_te)
        y_test_cls_list.append(y_te_c)
        y_test_reg_list.append(y_te_r)
        kept_symbols += 1
        extra = ""
        if len(X_va) == 0:
            extra += " val0"
        if len(X_te) == 0:
            extra += " test0"
        pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol}{extra}")

    pbar.close(suffix=f" ok={kept_symbols}")

    if not X_train_list:
        raise RuntimeError("没有生成任何序列样本（请检查 integrated_dir 是否存在 CSV，或放宽 split/seq_len 条件）")

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
    print(f"[FNSPID-HF] saved merged npz -> {output_npz}")


def export_fnspid_sequences_npz_sharded(
    integrated_dir: Path,
    output_dir: Path,
    symbols: Optional[List[str]],
    seq_len: int,
    horizon: int,
    vol_window: int,
    vol_threshold: float,
    train_ratio: float,
    val_ratio: float,
    normalize_numeric: bool,
    include_news_count: bool,
    shard_max_train_samples: int,
    manifest_path: Path,
) -> None:
    wanted = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    csv_files = sorted([p for p in integrated_dir.glob("*.csv") if p.is_file()])
    if wanted is not None:
        csv_files = [p for p in csv_files if p.stem.upper() in wanted]

    base_feature_cols = ["open", "high", "low", "close", "volume", "scaled_sentiment", "news_flag"]
    extra_cols = ["news_count_log1p"] if include_news_count else []
    feature_cols = base_feature_cols + extra_cols
    normalize_cols = ["open", "high", "low", "close", "volume"] + (extra_cols if normalize_numeric else [])

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
        "format": "fnspid_processed_shards",
        "integrated_dir": str(integrated_dir),
        "feature_cols": feature_cols,
        "seq_len": int(seq_len),
        "horizon": int(horizon),
        "vol_window": int(vol_window),
        "vol_threshold": float(vol_threshold),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "normalize_numeric": bool(normalize_numeric),
        "include_news_count": bool(include_news_count),
        "shard_max_train_samples": int(shard_max_train_samples),
        "shards": [],
    }

    def _flush_shard(force: bool = False) -> None:
        nonlocal shard_idx, buf_train_n
        if (not force) and buf_train_n < int(shard_max_train_samples):
            return
        if buf_train_n == 0 and not force:
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

        shard_name = f"fnspid_shard_{shard_idx:05d}.npz"
        shard_path = output_dir / shard_name
        np.savez_compressed(
            str(shard_path),
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
                {
                    "file": shard_name,
                    "train": int(X_train.shape[0]),
                    "val": int(X_val.shape[0]),
                    "test": int(X_test.shape[0]),
                }
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

    pbar = _ProgressBar(desc="[FNSPID-HF] shards", total=len(csv_files))
    processed = 0
    kept_symbols = 0
    for fp in csv_files:
        processed += 1
        symbol = fp.stem.upper()
        try:
            df = pd.read_csv(fp)
        except Exception:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} failed_read")
            continue

        df = _standardize_columns(df)
        if "date" not in df.columns:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} no_date")
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        if "scaled_sentiment" not in df.columns:
            if "sentiment" in df.columns:
                df["scaled_sentiment"] = (pd.to_numeric(df["sentiment"], errors="coerce").fillna(3.0) - 0.9999) / 4.0
            else:
                df["scaled_sentiment"] = (3.0 - 0.9999) / 4.0
        df["scaled_sentiment"] = pd.to_numeric(df["scaled_sentiment"], errors="coerce").fillna((3.0 - 0.9999) / 4.0).astype(
            np.float32
        )

        if "news_flag" not in df.columns:
            if "news_count" in df.columns:
                df["news_flag"] = (pd.to_numeric(df["news_count"], errors="coerce").fillna(0) > 0).astype(np.int8)
            else:
                df["news_flag"] = 0
        df["news_flag"] = pd.to_numeric(df["news_flag"], errors="coerce").fillna(0).astype(np.int8)

        if include_news_count:
            nc = pd.to_numeric(df["news_count"], errors="coerce").fillna(0.0).astype(np.float32) if "news_count" in df.columns else 0.0
            df["news_count_log1p"] = np.log1p(nc).astype(np.float32)

        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} missing_{c}")
                df = pd.DataFrame()
                break
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        if df.empty:
            continue
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        df = _add_fnspid_labels(
            df=df,
            horizon=int(horizon),
            vol_window=int(vol_window),
            vol_threshold=float(vol_threshold),
        )
        if len(df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} too_short")
            continue

        train_df, val_df, test_df = _split_time_series(df, train_ratio=float(train_ratio), val_ratio=float(val_ratio))
        if len(train_df) < int(seq_len):
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} train_short")
            continue

        if normalize_numeric:
            train_df, val_df, test_df, _ = _normalize_with_train_stats(train_df, val_df, test_df, cols=normalize_cols)

        X_tr, y_tr_c, y_tr_r = _create_sequences(train_df, feature_cols, seq_len=int(seq_len))
        if len(X_tr) == 0:
            pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} last={symbol} no_seq")
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
        kept_symbols += 1
        _flush_shard(force=False)

        extra = ""
        if len(X_va) == 0:
            extra += " val0"
        if len(X_te) == 0:
            extra += " test0"
        pbar.update_abs(done=processed, total=len(csv_files), suffix=f" ok={kept_symbols} shards={shard_idx} last={symbol}{extra}")

    _flush_shard(force=True)
    pbar.close(suffix=f" ok={kept_symbols} shards={shard_idx}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"[FNSPID-HF] saved manifest -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--build_index", type=int, default=1)
    parser.add_argument("--integrate", type=int, default=1)
    parser.add_argument("--export_dataset", type=int, default=1)
    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--decay_rate", type=float, default=0.05)
    parser.add_argument("--min_price_rows", type=int, default=0)
    parser.add_argument("--save_csv", type=int, default=1)
    parser.add_argument("--save_npz", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--vol_window", type=int, default=20)
    parser.add_argument("--vol_threshold", type=float, default=1.5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--export_mode", type=str, default="shards")
    parser.add_argument("--shard_max_train_samples", type=int, default=200000)
    parser.add_argument("--include_news_count", type=int, default=1)
    parser.add_argument("--normalize_numeric", type=int, default=1)
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--summarize_missing_lsa", type=int, default=1)
    parser.add_argument("--sentiment_method", type=str, default="finbert")
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--gpt_batch_size", type=int, default=5)
    parser.add_argument("--gpt_sleep_seconds", type=float, default=1.0)
    parser.add_argument("--gpt_max_retries", type=int, default=8)
    parser.add_argument("--gpt_backoff_seconds", type=float, default=2.0)
    parser.add_argument("--gpt_backoff_max_seconds", type=float, default=120.0)
    parser.add_argument("--gpt_on_failure", type=str, default="raise")
    parser.add_argument("--lm_positive_path", type=str, default=None)
    parser.add_argument("--lm_negative_path", type=str, default=None)
    parser.add_argument("--finbert_model", type=str, default="ProsusAI/finbert")
    parser.add_argument("--finbert_batch_size", type=int, default=16)
    parser.add_argument("--finbert_max_length", type=int, default=256)
    parser.add_argument("--finbert_device", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (project_root / "datasets" / "FNSPID-hf")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "data" / "fnspid_hf")
    output_dir.mkdir(parents=True, exist_ok=True)

    news_dir = dataset_dir / "Stock_news"
    price_zip = dataset_dir / "Stock_price" / "full_history.zip"
    price_dir_candidates = [
        dataset_dir / "Stock_price" / "full_history" / "full_history",
        dataset_dir / "Stock_price" / "full_history",
    ]
    price_dir = None
    for cand in price_dir_candidates:
        if cand.exists() and any(p.suffix.lower() == ".csv" for p in cand.glob("*.csv")):
            price_dir = cand
            break
    news_csvs = []
    for name in ["All_external.csv", "nasdaq_exteral_data.csv"]:
        fp = news_dir / name
        if fp.exists():
            news_csvs.append(fp)

    sentiment_method = str(args.sentiment_method).strip().lower()
    if args.db_path:
        db_fp = Path(args.db_path)
    else:
        db_fp = output_dir / f"news_daily_{sentiment_method}.sqlite"
    if sentiment_method == "finbert" and str(args.finbert_model).strip() == "ProsusAI/finbert":
        local_finbert = project_root / "datasets" / "finbert"
        if (local_finbert / "config.json").exists() and (
            (local_finbert / "pytorch_model.bin").exists() or (local_finbert / "model.safetensors").exists()
        ):
            args.finbert_model = str(local_finbert)

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    if int(args.build_index) == 1:
        build_news_daily_index(
            news_csvs=news_csvs,
            db_fp=db_fp,
            cfg=BuildIndexConfig(
                gpt_batch_size=int(args.gpt_batch_size),
                gpt_sleep_seconds=float(args.gpt_sleep_seconds),
            ),
            symbols=symbols,
            sentiment_method=sentiment_method,
            summarize_missing_lsa=bool(int(args.summarize_missing_lsa)),
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            gpt_model=str(args.gpt_model),
            gpt_max_retries=int(args.gpt_max_retries),
            gpt_backoff_seconds=float(args.gpt_backoff_seconds),
            gpt_backoff_max_seconds=float(args.gpt_backoff_max_seconds),
            gpt_on_failure=str(args.gpt_on_failure).strip().lower(),
            lm_positive_path=args.lm_positive_path,
            lm_negative_path=args.lm_negative_path,
            finbert_model=args.finbert_model,
            finbert_batch_size=int(args.finbert_batch_size),
            finbert_max_length=int(args.finbert_max_length),
            finbert_device=int(args.finbert_device),
        )

    if int(args.integrate) == 1:
        print(f"[FNSPID-HF] news csvs: {[str(p) for p in news_csvs]}")
        print(f"[FNSPID-HF] price dir: {price_dir}")
        print(f"[FNSPID-HF] price zip: {price_zip}")
        integrate_price_with_news(
            price_zip_fp=price_zip,
            price_dir=price_dir,
            db_fp=db_fp,
            output_dir=output_dir / "integrated",
            save_csv=bool(int(args.save_csv)),
            save_npz=bool(int(args.save_npz)),
            decay_rate=float(args.decay_rate),
            min_price_rows=int(args.min_price_rows),
            symbols=symbols,
        )

    if int(args.export_dataset) == 1:
        export_mode = str(args.export_mode).strip().lower()
        if export_mode == "single":
            export_fnspid_sequences_npz(
                integrated_dir=output_dir / "integrated",
                output_npz=output_dir / "fnspid_processed.npz",
                symbols=symbols,
                seq_len=int(args.seq_len),
                horizon=int(args.horizon),
                vol_window=int(args.vol_window),
                vol_threshold=float(args.vol_threshold),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                normalize_price=bool(int(args.normalize_numeric)),
            )
        elif export_mode == "shards":
            export_fnspid_sequences_npz_sharded(
                integrated_dir=output_dir / "integrated",
                output_dir=output_dir / "fnspid_shards",
                symbols=symbols,
                seq_len=int(args.seq_len),
                horizon=int(args.horizon),
                vol_window=int(args.vol_window),
                vol_threshold=float(args.vol_threshold),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                normalize_numeric=bool(int(args.normalize_numeric)),
                include_news_count=bool(int(args.include_news_count)),
                shard_max_train_samples=int(args.shard_max_train_samples),
                manifest_path=output_dir / "fnspid_shards_manifest.json",
            )
        else:
            raise ValueError(f"Unknown export_mode: {export_mode}")


if __name__ == "__main__":
    main()
