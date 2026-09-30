"""Synthetic Blue Book data in the exact 53-column Kaggle schema, with the real
dataset's noise patterns: YearMade recorded as 1000, zeroed hour meters, mostly-empty
option columns, thousands of model codes. Used only by the smoke test.

Usage: python scripts/05_make_synthetic.py --n-train 6000 --n-valid 600 --n-test 600
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schema import ALL_COLUMNS  # noqa: E402

GROUPS = {
    "TEX": ("Track Excavators", 10.6), "WL": ("Wheel Loader", 10.4),
    "TTT": ("Track Type Tractors", 10.5), "BL": ("Backhoe Loaders", 9.9),
    "SSL": ("Skid Steer Loaders", 9.2), "MG": ("Motor Graders", 10.8),
}
SIZES = ["Mini", "Small", "Compact", "Medium", "Large / Medium", "Large", None]
SIZE_EFF = {"Mini": -0.7, "Small": -0.35, "Compact": -0.5, "Medium": 0.0,
            "Large / Medium": 0.2, "Large": 0.45, None: 0.0}
STATES = ["Texas", "Florida", "California", "Georgia", "Ohio", "Illinois", "Colorado",
          "Arizona", "New York", "Tennessee", "Missouri", "North Carolina"]
OPTION_LEVELS = ["None or Unspecified", "Standard", "High Flow", "Hydraulic", "Manual",
                 "Extended", "Double", "Triple", "Yes", "No"]


def make_machines(n_models, n_machines, rng):
    model_ids = rng.integers(1000, 40000, size=n_models)
    model_eff = rng.normal(0, 0.35, size=n_models)
    model_group = rng.choice(list(GROUPS), size=n_models)
    rows = []
    for mid in range(n_machines):
        k = rng.integers(0, n_models)
        year_true = int(rng.integers(1974, 2011))
        rows.append({
            "MachineID": 1_000_000 + mid,
            "model_k": k, "ModelID": int(model_ids[k]),
            "group": model_group[k], "model_eff": float(model_eff[k]),
            "year_true": year_true,
            "manufacturer": int(rng.integers(1, 60)),
            "size": SIZES[rng.integers(0, len(SIZES))],
        })
    return pd.DataFrame(rows)


def sales_for(machines, n, date_lo, date_hi, rng, start_id):
    m = machines.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31)))
    m = m.reset_index(drop=True)
    days = rng.integers(0, (pd.Timestamp(date_hi) - pd.Timestamp(date_lo)).days + 1, n)
    saledate = pd.Timestamp(date_lo) + pd.to_timedelta(days, unit="D")
    sale_year = pd.DatetimeIndex(saledate).year.to_numpy()

    age = np.clip(sale_year - m["year_true"].to_numpy(), 0, None)
    hours_true = np.expm1(rng.normal(7.6, 1.1, n)) * (1 + 0.15 * age)
    usage = np.where(hours_true > 6000, "High", np.where(hours_true > 2500, "Medium", "Low"))

    group_lvl = m["group"].map(lambda g: GROUPS[g][1]).to_numpy()
    size_eff = m["size"].map(lambda s: SIZE_EFF[s]).to_numpy()
    log_price = (group_lvl + m["model_eff"].to_numpy() + size_eff
                 - 0.045 * age - 0.035 * np.log1p(hours_true)
                 + 0.012 * (sale_year - 2000)
                 + 0.03 * np.sin(2 * np.pi * pd.DatetimeIndex(saledate).month / 12)
                 + rng.normal(0, 0.32, n))
    price = np.round(np.exp(log_price) / 100) * 100

    df = pd.DataFrame({
        "SalesID": np.arange(start_id, start_id + n),
        "SalePrice": price,
        "MachineID": m["MachineID"], "ModelID": m["ModelID"],
        "datasource": rng.choice([121, 132, 136, 149, 172], n),
        "auctioneerID": np.where(rng.random(n) < 0.05, np.nan, rng.integers(1, 28, n)),
        # the recorded year: 12% corrupted to 1000, the classic Blue Book defect
        "YearMade": np.where(rng.random(n) < 0.12, 1000, m["year_true"]),
        # the recorded meter: 40% zeroed, 20% missing
        "MachineHoursCurrentMeter": np.where(
            rng.random(n) < 0.40, 0.0,
            np.where(rng.random(n) < 0.25, np.nan, np.round(hours_true))),
        "UsageBand": np.where(rng.random(n) < 0.75, None, usage),
        "saledate": saledate.strftime("%m/%d/%Y %H:%M"),
        "fiModelDesc": ["M" + str(x) + v for x, v in zip(
            m["ModelID"], rng.choice(["", "C", "D", "LC", "XT", "G"], n))],
        "fiBaseModel": ["M" + str(x) for x in m["ModelID"]],
        "fiSecondaryDesc": np.where(rng.random(n) < 0.65, None,
                                    rng.choice(["C", "D", "LC", "XT", "G", "H"], n)),
        "fiModelSeries": np.where(rng.random(n) < 0.85, None,
                                  rng.choice(["II", "III", "-2", "-5"], n)),
        "fiModelDescriptor": np.where(rng.random(n) < 0.8, None,
                                      rng.choice(["LGP", "XL", "LT", "SB"], n)),
        "ProductSize": m["size"],
        "fiProductClassDesc": [f"{GROUPS[g][0]} - {rng.integers(1, 30)}.0 to "
                               f"{rng.integers(30, 90)}.0 Metric Tons" for g in m["group"]],
        "state": rng.choice(STATES, n),
        "ProductGroup": m["group"],
        "ProductGroupDesc": m["group"].map(lambda g: GROUPS[g][0]),
    })
    for c in ALL_COLUMNS[20:]:  # the 33 option columns, mostly empty
        share_missing = rng.uniform(0.55, 0.95)
        vals = rng.choice(OPTION_LEVELS[:rng.integers(2, 6)], n)
        df[c] = np.where(rng.random(n) < share_missing, None, vals)
    return df[ALL_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-valid", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    machines = make_machines(n_models=800, n_machines=max(2000, args.n_train // 2), rng=rng)

    train = sales_for(machines, args.n_train, "1994-01-01", "2011-12-31", rng, 1)
    valid = sales_for(machines, args.n_valid, "2012-01-01", "2012-04-30", rng, 4_000_001)
    test = sales_for(machines, args.n_test, "2012-05-01", "2012-11-30", rng, 6_000_001)

    train.to_csv(out / "Train.csv", index=False)
    valid.drop(columns=["SalePrice"]).to_csv(out / "Valid.csv", index=False)
    valid[["SalesID", "SalePrice"]].to_csv(out / "ValidSolution.csv", index=False)
    test.drop(columns=["SalePrice"]).to_csv(out / "Test.csv", index=False)

    appendix = machines[["MachineID", "ModelID", "year_true", "manufacturer"]].copy()
    appendix = appendix.rename(columns={"year_true": "MfgYear",
                                        "manufacturer": "fiManufacturerID"})
    appendix["fiManufacturerDesc"] = "MFG" + appendix["fiManufacturerID"].astype(str)
    appendix["PrimarySizeBasis"] = "Operating Weight"
    appendix["PrimaryLower"] = np.round(np.exp(rng.normal(9.5, 0.8, len(appendix))))
    appendix["PrimaryUpper"] = appendix["PrimaryLower"] * rng.uniform(1.1, 1.6, len(appendix))
    appendix.to_csv(out / "Machine_Appendix.csv", index=False)

    print(f"[synthetic] {args.n_train}/{args.n_valid}/{args.n_test} sales, "
          f"{len(machines)} machines -> {out}/ (53-column Kaggle schema)")


if __name__ == "__main__":
    main()
