import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .datasets import preprocess_csv_dataset


DATASET_FILES: Dict[str, str] = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "electricity": "electricity.csv",
    "exchange_rate": "exchange_rate.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="preprocess_datasets")
    p.add_argument("--data-root", type=Path, default=Path("datasets"))
    p.add_argument("--out-root", type=Path, default=Path("data"))
    p.add_argument("--datasets", nargs="*", default=["all"])
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--time-features", choices=["none", "simple"], default="simple")
    p.add_argument("--format", choices=["full", "windows"], default="full")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    chosen = args.datasets
    if len(chosen) == 1 and chosen[0].lower() == "all":
        chosen = list(DATASET_FILES.keys())

    failures: List[str] = []
    for name in chosen:
        if name not in DATASET_FILES:
            raise SystemExit(f"Unknown dataset '{name}'. Choices: {sorted(DATASET_FILES.keys())} or 'all'")
        csv_path = args.data_root / DATASET_FILES[name]
        if not csv_path.exists():
            raise SystemExit(f"Missing dataset file: {csv_path}")
        out_dir = args.out_root / name
        try:
            result = preprocess_csv_dataset(
                csv_path=csv_path,
                out_dir=out_dir,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                stride=args.stride,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                time_features=args.time_features,
                output_format=args.format,
            )
            print(f"{name}: saved {result.series_path} and {result.meta_path}")
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"{name}: FAILED - {type(e).__name__}: {e}")

    if failures:
        raise SystemExit("Some datasets failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
