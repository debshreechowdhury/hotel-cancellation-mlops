"""Generate a small SYNTHETIC file with the same schema as Kaggle's hotel_bookings.csv.

Only for tests / CI / smoke-testing Docker without cloud credentials.
Real training must use the real dataset (see README).

    python scripts/make_sample_data.py --rows 3000 --out data/raw/sample_hotel_bookings.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def make(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hotel = rng.choice(["City Hotel", "Resort Hotel"], rows, p=[0.66, 0.34])
    lead = rng.gamma(1.2, 85, rows).astype(int)
    deposit = rng.choice(["No Deposit", "Non Refund", "Refundable"], rows, p=[0.86, 0.13, 0.01])
    segment = rng.choice(["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"],
                         rows, p=[0.47, 0.2, 0.17, 0.1, 0.06])
    repeated = rng.binomial(1, 0.03, rows)
    special = rng.poisson(0.6, rows)
    parking = rng.binomial(1, 0.06, rows)
    prev_cancel = rng.poisson(0.08, rows)

    logit = (-1.3 + 0.006 * lead + 2.5 * (deposit == "Non Refund") - 0.5 * special
             - 2.0 * parking - 1.0 * repeated + 0.6 * (segment == "Online TA")
             + 0.8 * np.minimum(prev_cancel, 3) + 0.3 * (hotel == "City Hotel"))
    canceled = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "hotel": hotel,
        "is_canceled": canceled,
        "lead_time": lead,
        "arrival_date_year": rng.choice([2015, 2016, 2017], rows),
        "arrival_date_month": rng.choice(MONTHS, rows),
        "arrival_date_week_number": rng.integers(1, 54, rows),
        "arrival_date_day_of_month": rng.integers(1, 32, rows),
        "stays_in_weekend_nights": rng.poisson(0.9, rows),
        "stays_in_week_nights": rng.poisson(2.5, rows),
        "adults": rng.choice([1, 2, 3], rows, p=[0.2, 0.72, 0.08]),
        "children": rng.choice([0.0, 1.0, 2.0, np.nan], rows, p=[0.92, 0.04, 0.035, 0.005]),
        "babies": rng.binomial(1, 0.01, rows),
        "meal": rng.choice(["BB", "HB", "SC", "FB", "Undefined"], rows, p=[0.77, 0.12, 0.09, 0.01, 0.01]),
        "country": rng.choice(["PRT", "GBR", "FRA", "ESP", "DEU", np.nan], rows,
                              p=[0.4, 0.2, 0.15, 0.13, 0.115, 0.005]),
        "market_segment": segment,
        "distribution_channel": rng.choice(["TA/TO", "Direct", "Corporate"], rows, p=[0.82, 0.12, 0.06]),
        "is_repeated_guest": repeated,
        "previous_cancellations": prev_cancel,
        "previous_bookings_not_canceled": rng.poisson(0.1, rows),
        "reserved_room_type": rng.choice(list("ADEFG"), rows, p=[0.72, 0.16, 0.06, 0.03, 0.03]),
        "assigned_room_type": rng.choice(list("ADEFG"), rows),
        "booking_changes": rng.poisson(0.2, rows),
        "deposit_type": deposit,
        "agent": rng.choice([9.0, 240.0, np.nan], rows),
        "company": np.nan,
        "days_in_waiting_list": rng.choice([0, 0, 0, 0, 10, 40], rows),
        "customer_type": rng.choice(["Transient", "Transient-Party", "Contract", "Group"],
                                    rows, p=[0.75, 0.21, 0.03, 0.01]),
        "adr": np.round(rng.gamma(6, 17, rows), 2),
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special,
        "reservation_status": np.where(canceled == 1, "Canceled", "Check-Out"),
        "reservation_status_date": "2016-01-01",
    })
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/raw/sample_hotel_bookings.csv")
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make(a.rows, a.seed).to_csv(out, index=False)
    print(f"wrote {a.rows} synthetic rows -> {out}")


if __name__ == "__main__":
    main()
