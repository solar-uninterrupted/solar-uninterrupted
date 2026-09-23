"""
build_panel.py — Solar Uninterrupted (BDA 602, Fall 2026)

Reshape the two Clean Energy Regulator (CER) small-scale solar installation
files from wide (postcode x month columns) to a long postcode-month panel.

Usage
-----
    python src/build_panel.py --raw data/raw --out data/processed

Inputs (data/raw)
-----------------
    sgu-solar-installations-2001-to-2010.csv
    sgu-solar-installations-2011-to-present-and-totals.csv

Outputs (data/processed)
------------------------
    cer_solar_panel_long.csv      854,310 rows: postcode, month, installations
    cer_solar_panel_long.meta.json  dtype sidecar + provenance; load with load_panel()

Handling decisions (documented in the M2 planning summary)
----------------------------------------------------------
1. Leading zeros. 48 postcodes begin with 0 (e.g. 0800 Darwin). Postcode is
   read as a string and never converted to a number.
2. Schema drift. The 2001-2010 file names monthly columns
   "<Mon YYYY> - Installations Quantity"; the 2011-present file uses
   "<Mon YYYY> - Installation Quantity" (singular). Both are parsed with one
   regex and the two panels concatenated on a common schema.
3. Totals columns. The 2011-present file carries "Historic Total ... (2001 - 2010)"
   and "Total" columns. They are dropped from the panel and used only to
   reconcile the row total (4,501,887) against the CER published figure.
4. Postcode coverage. The 2001-2010 file has 2,809 postcodes, the 2011-present
   file 2,811. By default the panel keeps only the rows present in the source
   files (854,310 rows = 2,809 x 117 + 2,811 x 187), which reproduces the
   figures reported in the M2 planning summary. Pass --complete-grid to fill
   the 2 x 117 missing postcode-months with 0 (854,544 rows).
5. Registration lag and privacy floor are NOT applied here; they are modelling
   decisions handled downstream (see README).
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

POSTCODE_COL = "Small Unit Installation Postcode"
MONTH_RE = re.compile(r"^([A-Z][a-z]{2}) (\d{4}) - Installations? Quantity$")


def _melt(path: Path) -> pd.DataFrame:
    """Read one CER wide file and return long rows (postcode, month, installations)."""
    df = pd.read_csv(path, dtype={POSTCODE_COL: str}, thousands=",")
    month_cols = {}
    for col in df.columns:
        m = MONTH_RE.match(col)
        if m:
            month_cols[col] = pd.Timestamp(f"{m.group(2)}-{m.group(1)}-01")
    long = (
        df[[POSTCODE_COL] + list(month_cols)]
        .melt(id_vars=POSTCODE_COL, var_name="col", value_name="installations")
    )
    long["month"] = long["col"].map(month_cols)
    long = long.drop(columns="col")
    long["installations"] = pd.to_numeric(long["installations"], errors="coerce").fillna(0).astype("int64")
    return long


def build_panel(raw_dir: Path, complete_grid: bool = False) -> pd.DataFrame:
    a = _melt(raw_dir / "sgu-solar-installations-2001-to-2010.csv")
    b = _melt(raw_dir / "sgu-solar-installations-2011-to-present-and-totals.csv")
    long = pd.concat([a, b], ignore_index=True)
    if not complete_grid:
        return long.sort_values([POSTCODE_COL, "month"]).reset_index(drop=True)

    # Optional: complete the grid so every postcode has every month (fill 0).
    postcodes = sorted(long[POSTCODE_COL].unique())
    months = sorted(long["month"].unique())
    grid = pd.MultiIndex.from_product([postcodes, months], names=[POSTCODE_COL, "month"]).to_frame(index=False)
    panel = grid.merge(long, on=[POSTCODE_COL, "month"], how="left")
    panel["installations"] = panel["installations"].fillna(0).astype("int64")
    return panel.sort_values([POSTCODE_COL, "month"]).reset_index(drop=True)


def write_outputs(panel: pd.DataFrame, out_dir: Path, raw_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cer_solar_panel_long.csv"
    panel.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    meta = {
        "file": csv_path.name,
        "description": "CER small-scale solar installations, long postcode-month panel",
        "source": "Clean Energy Regulator, Small-scale installation postcode data, current as at 31 July 2026",
        "source_url": "https://cer.gov.au/markets/reports-and-data/small-scale-installation-postcode-data",
        "raw_files": [
            "sgu-solar-installations-2001-to-2010.csv",
            "sgu-solar-installations-2011-to-present-and-totals.csv",
        ],
        "columns": {
            POSTCODE_COL: {
                "dtype": "string",
                "role": "nominal identifier",
                "note": "4-character string; 48 postcodes have leading zeros. NEVER read as int.",
            },
            "month": {"dtype": "datetime64[ns]", "format": "%Y-%m-%d", "role": "time index (first day of month)"},
            "installations": {"dtype": "int64", "role": "target (monthly count of small-scale solar installations)"},
        },
        "grid": "as-in-source (see handling decision 4)",
        "shape": {"rows": int(len(panel)), "postcodes": int(panel[POSTCODE_COL].nunique()), "months": int(panel["month"].nunique())},
        "coverage": {"first_month": str(panel["month"].min().date()), "last_month": str(panel["month"].max().date())},
        "total_installations": int(panel["installations"].sum()),
        "known_issues": {
            "registration_lag": "Installers have 12 months to register; the last ~12 months undercount. Truncate or lag-adjust before modelling.",
            "privacy_floor": "CER modifies cells with fewer than 10 installations; flag rather than treat as exact.",
            "no_residential_split": "Residential and commercial systems are not distinguished; apply a capacity threshold following APVI.",
            "postcode_0000": "A '0000' postcode row exists in the CER files (unassigned/placeholder); exclude from geographic joins.",
        },
        "load_with": "src/build_panel.py: load_panel(path)",
    }
    (out_dir / "cer_solar_panel_long.meta.json").write_text(json.dumps(meta, indent=2))


def load_panel(csv_path: str | Path) -> pd.DataFrame:
    """Load the panel with dtypes restored from the sidecar. Use this, not pd.read_csv."""
    csv_path = Path(csv_path)
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text())
    df = pd.read_csv(csv_path, dtype={POSTCODE_COL: str}, parse_dates=["month"])
    df["installations"] = df["installations"].astype("int64")
    assert df[POSTCODE_COL].str.len().eq(4).all(), "postcode lost its leading zeros"
    assert len(df) == meta["shape"]["rows"], "row count does not match sidecar"
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--complete-grid", action="store_true", help="fill missing postcode-months with 0")
    args = ap.parse_args()
    panel = build_panel(Path(args.raw), complete_grid=args.complete_grid)
    write_outputs(panel, Path(args.out), Path(args.raw))
    print(f"rows={len(panel):,} postcodes={panel[POSTCODE_COL].nunique():,} "
          f"months={panel['month'].nunique()} total={panel['installations'].sum():,}")
