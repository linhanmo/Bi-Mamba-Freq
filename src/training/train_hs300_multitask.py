from pathlib import Path

from .train_finmultitime import main as fin_main


if __name__ == "__main__":
    import sys

    default_args = [
        "--processed-root",
        str(Path("data/FinMultitime/HS300_time_series")),
        "--expected-market",
        "HS300",
    ]
    fin_main(default_args + sys.argv[1:])
