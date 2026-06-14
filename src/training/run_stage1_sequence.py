import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_message(log_path: Path, message: str) -> None:
    line = f"[{_now_text()}] {message}"
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(cmd: list[str], log_path: Path) -> None:
    _log_message(log_path, "[Stage1] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _copy_best(src_dir: Path, tag: str, snapshot_dir: Path, log_path: Path) -> None:
    src = src_dir / "best_model.pt"
    if not src.exists():
        raise FileNotFoundError(f"未找到 best_model.pt: {src}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dst = snapshot_dir / f"{tag}_best_stage1.pt"
    shutil.copy2(src, dst)
    _log_message(log_path, f"[Stage1] snapshot saved -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-1 standalone training sequentially on NASDAQ, CSI300, S&P500, and FRED-MD.")
    parser.add_argument("--datasets", type=str, default="nasdaq,csi300,sp500,fred_md")
    parser.add_argument("--prepared_dir", type=str, default=str(ROOT / "data" / "stage1_ready"))
    parser.add_argument("--output_root", type=str, default=str(ROOT / "outputs" / "stage1_sequence"))
    parser.add_argument("--stocknet_input_dir", type=str, default=str(ROOT / "data" / "stocknet_nasdaq100"))
    parser.add_argument("--csi_input_npz", type=str, default=str(ROOT / "data" / "csi300_processed.npz"))
    parser.add_argument("--sp_input_npz", type=str, default=str(ROOT / "data" / "sp500_processed.npz"))
    parser.add_argument("--fred_input_npz", type=str, default=str(ROOT / "data" / "fred_md" / "FRED-MD_2024m12_fred_md_factors.npz"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--reg_loss_weight", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand_factor", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--dt_rank", type=str, default="auto")
    parser.add_argument("--freq_groups", type=int, default=3)
    parser.add_argument("--freq_group_dt", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--low_rank", type=int, default=32)
    parser.add_argument("--bidirectional_merge", type=str, default="concat")
    parser.add_argument("--pool", type=str, default="mean")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pscan", type=int, default=1)
    parser.add_argument("--save_interpret", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--force_prepare", type=int, default=0)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    output_root = Path(args.output_root)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_root / "snapshots"
    stage1_log_path = output_root / "stage1_sequence.log"
    stage1_log_path.write_text("", encoding="utf-8")

    py = sys.executable
    train_py = str(SRC_DIR / "training" / "train_bimamba_freq.py")
    stocknet_prepare_py = str(SRC_DIR / "training" / "prepare_stocknet_stage1.py")
    fred_prepare_py = str(SRC_DIR / "training" / "prepare_fred_md_stage1.py")

    dataset_order = [x.strip() for x in args.datasets.split(",") if x.strip()]
    shared_train_args = [
        "--device", args.device,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--min_lr", str(args.min_lr),
        "--max_grad_norm", str(args.max_grad_norm),
        "--reg_loss_weight", str(args.reg_loss_weight),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--d_state", str(args.d_state),
        "--expand_factor", str(args.expand_factor),
        "--d_conv", str(args.d_conv),
        "--dt_rank", str(args.dt_rank),
        "--freq_groups", str(args.freq_groups),
        "--freq_group_dt", str(args.freq_group_dt),
        "--low_rank", str(args.low_rank),
        "--bidirectional_merge", str(args.bidirectional_merge),
        "--pool", str(args.pool),
        "--dropout", str(args.dropout),
        "--pscan", str(args.pscan),
        "--save_interpret", str(args.save_interpret),
        "--save_every", str(args.save_every),
    ]

    prev_best_checkpoint: str | None = None
    completed: list[str] = []
    for dataset in dataset_order:
        extra_init_args: list[str] = []
        if prev_best_checkpoint:
            extra_init_args = ["--init_checkpoint", prev_best_checkpoint]

        if dataset == "nasdaq":
            stage1_npz = prepared_dir / "stocknet_stage1.npz"
            if bool(int(args.force_prepare)) or not stage1_npz.exists():
                _run([py, stocknet_prepare_py, "--input_dir", args.stocknet_input_dir, "--output_npz", str(stage1_npz)], stage1_log_path)
            out_dir = output_root / "nasdaq_stage1"
            _run([py, train_py, "--data", str(stage1_npz), "--output_dir", str(out_dir), *extra_init_args, *shared_train_args], stage1_log_path)
            prev_best_checkpoint = str(out_dir / "best_model.pt")
            completed.append("nasdaq")
            _copy_best(out_dir, "nasdaq_only", snapshot_dir, stage1_log_path)
        elif dataset == "csi300":
            out_dir = output_root / "csi300_stage1"
            _run([py, train_py, "--data", args.csi_input_npz, "--output_dir", str(out_dir), *extra_init_args, *shared_train_args], stage1_log_path)
            prev_best_checkpoint = str(out_dir / "best_model.pt")
            completed.append("csi300")
        elif dataset == "sp500":
            out_dir = output_root / "sp500_stage1"
            _run([py, train_py, "--data", args.sp_input_npz, "--output_dir", str(out_dir), *extra_init_args, *shared_train_args], stage1_log_path)
            prev_best_checkpoint = str(out_dir / "best_model.pt")
            completed.append("sp500")
            if completed[:3] == ["nasdaq", "csi300", "sp500"]:
                _copy_best(out_dir, "nasdaq_csi300_sp500", snapshot_dir, stage1_log_path)
        elif dataset == "fred_md":
            stage1_npz = prepared_dir / "fred_md_stage1.npz"
            if bool(int(args.force_prepare)) or not stage1_npz.exists():
                _run([py, fred_prepare_py, "--input_npz", args.fred_input_npz, "--output_npz", str(stage1_npz)], stage1_log_path)
            out_dir = output_root / "fred_md_stage1"
            _run([py, train_py, "--data", str(stage1_npz), "--output_dir", str(out_dir), *extra_init_args, *shared_train_args], stage1_log_path)
            prev_best_checkpoint = str(out_dir / "best_model.pt")
            completed.append("fred_md")
            if completed[:4] == ["nasdaq", "csi300", "sp500", "fred_md"]:
                _copy_best(out_dir, "all_four", snapshot_dir, stage1_log_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    _log_message(stage1_log_path, f"[Stage1] all done -> {output_root}")


if __name__ == "__main__":
    main()
