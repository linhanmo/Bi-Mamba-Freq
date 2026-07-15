from pathlib import Path

from .train_finmultitime import main as fin_main


if __name__ == "__main__":
    import sys

    default_args = [
        "--processed-root",
        str(Path("data/FinMultitime/S_P500_time_series")),
        "--expected-market",
        "S&P500",
        "--batch-size",
        "64",
        "--d-model",
        "192",
        "--d-state",
        "16",
        "--n-layers",
        "4",
        "--low-rank",
        "48",
        "--dropout",
        "0.1",
        "--lr",
        "1e-4",
        "--weight-decay",
        "1e-4",
        "--num-workers",
        "4",
    ]
    fin_main(default_args + sys.argv[1:])
