"""E2 sub-experiment (b1): rolling PE hourly on SPY 2008-2021."""
from __future__ import annotations

from helpers import HOURLY_MAIN, HOURLY_SENS, load_intraday, run_subexperiment


def main():
    price = load_intraday("1h")
    run_subexperiment(
        name="hourly", price=price,
        main_cfg=HOURLY_MAIN, sens_cfg=HOURLY_SENS,
        title_main="PE rolling hourly sobre SPY (2008-2021)",
        mk_freq="1YE", br_freq="1W",
    )


if __name__ == "__main__":
    main()
