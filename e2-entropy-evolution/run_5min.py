"""E2 sub-experiment (b2): rolling PE 5-min on SPY 2008-2021."""
from __future__ import annotations

from helpers import FIVEMIN_MAIN, FIVEMIN_SENS, load_intraday, run_subexperiment


def main():
    price = load_intraday("5min")
    run_subexperiment(
        name="5min", price=price,
        main_cfg=FIVEMIN_MAIN, sens_cfg=FIVEMIN_SENS,
        title_main="PE rolling 5-min sobre SPY (2008-2021)",
        mk_freq="3ME", br_freq="1D",
    )


if __name__ == "__main__":
    main()
