"""E2 sub-experiment (a): rolling PE daily on ^GSPC 1950-2025."""
from __future__ import annotations

from helpers import (
    DAILY_END, DAILY_MAIN, DAILY_SENS, DAILY_START, DAILY_TICKER,
    load_prices, run_subexperiment,
)


def main():
    prices = load_prices(DAILY_TICKER, DAILY_START, DAILY_END, interval="1d")
    price = prices["Adj Close"].astype(float)
    run_subexperiment(
        name="daily", price=price,
        main_cfg=DAILY_MAIN, sens_cfg=DAILY_SENS,
        title_main="PE rolling daily sobre ^GSPC (1928-2025)",
        mk_freq="1YE", br_freq="1ME",
        n_breaks=10,
    )


if __name__ == "__main__":
    main()
