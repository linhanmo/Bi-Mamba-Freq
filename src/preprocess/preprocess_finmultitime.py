import argparse
from pathlib import Path
from typing import List, Optional

from .finmultitime import preprocess_finmultitime_dataset


def _slugify_market_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum():
            safe.append(ch)
        elif ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    slug = "".join(safe).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "market"


def _discover_market_dirs(root: Path) -> List[Path]:
    if root.is_dir():
        direct_csv = list(root.glob("*.csv"))
        if direct_csv:
            return [root]
        market_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith("_time_series")]
        return sorted(market_dirs)
    return []


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="preprocess_finmultitime")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/FinMultitime"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/FinMultitime"),
    )
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--pred-len", type=int, default=5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--time-features", choices=["none", "simple"], default="simple")
    p.add_argument(
        "--vol-scale",
        type=float,
        default=252.0,
        help="Realized volatility label uses std(log-returns) * sqrt(vol-scale). Use 1.0 to disable annualization.",
    )
    p.add_argument("--max-files", type=int, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    market_dirs = _discover_market_dirs(args.data_root)
    if not market_dirs:
        raise SystemExit(f"No market directory or csv files found under: {args.data_root}")

    for market_dir in market_dirs:
        market_name = _slugify_market_name(market_dir.name)
        out_root = args.out_root / market_name if len(market_dirs) > 1 or market_dir != args.data_root else args.out_root
        result = preprocess_finmultitime_dataset(
            data_root=market_dir,
            out_root=out_root,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=args.stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            time_features=args.time_features,
            vol_scale=args.vol_scale,
            max_files=args.max_files,
        )
        print(f"{market_dir.name}: saved ticker directory: {result.series_path}")
        print(f"{market_dir.name}: saved manifest: {result.meta_path}")
        print(f"{market_dir.name}: market is processed separately and should be trained independently.")


if __name__ == "__main__":
    main()
