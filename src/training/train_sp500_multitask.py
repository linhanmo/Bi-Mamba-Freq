from pathlib import Path

from .train_finmultitime import main as fin_main


if __name__ == "__main__":
    import sys

    default_args = [
        "--processed-root",
        str(Path("data/FinMultitime/S_P500_time_series")),
        "--expected-market",
        "S&P500",
    ]
    fin_main(default_args + sys.argv[1:])

