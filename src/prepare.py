"""Read the raw Kaggle CSVs (or the synthetic copies), build the out-of-time splits,
write parquet files and the noise audit.

Usage: python -m src.prepare --raw-dir data/raw --out-dir data/prepared
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ALL_COLUMNS, APPENDIX_KEY, DATE_COL, DATE_PARSED, ID_COL, TARGET

VALID_START, VALID_END = "2012-01-01", "2012-04-30"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if DATE_COL in df.columns:
        # `saledate` stays exactly as it appears in the CSV (pre-registered: the raw
        # arm hands the original string to TabPFN). The parsed copy drives sorting,
        # splitting and the date-decomposed features, and never reaches the predictor.
        df[DATE_PARSED] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def load_raw(raw_dir: Path):
    """Return (train, valid, test, appendix); valid carries prices from ValidSolution."""
    train_p, valid_p = raw_dir / "Train.csv", raw_dir / "Valid.csv"
    if train_p.exists() and valid_p.exists():
        train, valid = _read_csv(train_p), _read_csv(valid_p)
    elif (raw_dir / "TrainAndValid.csv").exists():
        both = _read_csv(raw_dir / "TrainAndValid.csv")
        mask = (both[DATE_PARSED] >= VALID_START) & (both[DATE_PARSED] <= VALID_END)
        train, valid = both[~mask].copy(), both[mask].copy()
    else:
        raise FileNotFoundError(
            f"Neither Train.csv+Valid.csv nor TrainAndValid.csv found in {raw_dir}. "
            "See DATA_NOTICE.md for the required files.")

    sol_p = raw_dir / "ValidSolution.csv"
    if TARGET not in valid.columns or valid[TARGET].isna().all():
        if not sol_p.exists():
            raise FileNotFoundError(
                f"{sol_p} is required to score the validation split (see DATA_NOTICE.md).")
        sol = pd.read_csv(sol_p)
        price_col = TARGET if TARGET in sol.columns else sol.columns[1]
        valid = valid.drop(columns=[TARGET], errors="ignore").merge(
            sol[[ID_COL, price_col]].rename(columns={price_col: TARGET}),
            on=ID_COL, how="left", validate="one_to_one")
        if valid[TARGET].isna().any():
            raise ValueError("ValidSolution.csv did not cover every validation sale.")

    test_p = raw_dir / "Test.csv"
    test = _read_csv(test_p) if test_p.exists() else None

    app_p = raw_dir / "Machine_Appendix.csv"
    appendix = pd.read_csv(app_p, low_memory=False) if app_p.exists() else None

    train = train.sort_values(DATE_PARSED).reset_index(drop=True)
    valid = valid.sort_values(DATE_PARSED).reset_index(drop=True)
    return train, valid, test, appendix


def noise_audit(train: pd.DataFrame) -> dict:
    raw = train.drop(columns=[DATE_PARSED], errors="ignore")  # audit the table as shipped
    n = len(raw)
    audit = {
        "n_rows": int(n),
        "n_columns": int(raw.shape[1]),
        "date_min": str(train[DATE_PARSED].min().date()),
        "date_max": str(train[DATE_PARSED].max().date()),
        "missing_share_by_column": {
            c: round(float(raw[c].isna().mean()), 4) for c in raw.columns},
        "yearmade_eq_1000": int((raw["YearMade"] == 1000).sum()),
        "yearmade_lt_1900_share": round(float((raw["YearMade"] < 1900).mean()), 4),
        "hours_zero_share": round(
            float((raw["MachineHoursCurrentMeter"].fillna(-1) == 0).mean()), 4),
        "hours_missing_share": round(
            float(raw["MachineHoursCurrentMeter"].isna().mean()), 4),
        "cardinality": {
            c: int(raw[c].nunique()) for c in
            ["ModelID", "fiModelDesc", "fiBaseModel", "fiProductClassDesc", "state",
             "ProductGroup"] if c in raw.columns},
        "price_median": float(raw[TARGET].median()) if TARGET in raw.columns else None,
        "share_columns_over_50pct_missing": round(float(np.mean(
            [raw[c].isna().mean() > 0.5 for c in raw.columns])), 4),
    }
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/prepared")
    args = ap.parse_args(argv)

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    train, valid, test, appendix = load_raw(raw_dir)

    expected = set(ALL_COLUMNS)
    got = set(train.columns) - {DATE_PARSED}
    if got != expected:
        print(f"[prepare] WARNING: train columns differ from the 53-column schema "
              f"(missing: {sorted(expected - got)}; extra: {sorted(got - expected)})")

    train.to_parquet(out_dir / "train.parquet", index=False)
    valid.to_parquet(out_dir / "valid.parquet", index=False)
    if test is not None:
        test.to_parquet(out_dir / "test.parquet", index=False)
    if appendix is not None:
        appendix.to_parquet(out_dir / "appendix.parquet", index=False)

    audit = noise_audit(train)
    with open("results/metrics/noise_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(f"[prepare] train {len(train)} rows ({audit['date_min']} .. {audit['date_max']}), "
          f"valid {len(valid)}, test {0 if test is None else len(test)}, "
          f"appendix {'yes' if appendix is not None else 'no'}")
    print(f"[prepare] YearMade==1000: {audit['yearmade_eq_1000']} | zero hour meter: "
          f"{audit['hours_zero_share']:.1%} | columns >50% missing: "
          f"{audit['share_columns_over_50pct_missing']:.1%}")
    print("[prepare] audit -> results/metrics/noise_audit.json")


if __name__ == "__main__":
    main()
