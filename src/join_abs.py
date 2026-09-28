#!/usr/bin/env python3
"""
join_abs.py — Solar Uninterrupted (BDA 602, Fall 2026)

Join the CER postcode-month panel to the ABS 2021 Census (General Community
Profile, Postal Areas) and build the adoption-rate target plus postcode-level
demographic features.

Usage
-----
    python src/join_abs.py
    python src/join_abs.py --panel data/processed/cer_solar_panel_long.csv \
                           --abs data/raw/abs --out data/processed

Inputs
------
    data/processed/cer_solar_panel_long.csv   (built by src/build_panel.py)
    data/raw/abs/2021Census_G01_AUST_POA.csv  persons
    data/raw/abs/2021Census_G02_AUST_POA.csv  medians and averages
    data/raw/abs/2021Census_G36_AUST_POA.csv  dwelling structure (denominator)
    data/raw/abs/2021Census_G37_AUST_POA.csv  tenure
    data/raw/abs/2021_POA_area_sqkm.csv       POA area and state, extracted from
                                               Metadata/2021Census_geog_desc_1st_2nd_3rd_release.xlsx
                                               (sheet 2021_ASGS_Non_ABS_Structures, rows ASGS_Structure == "POA")

Outputs (data/processed)
------------------------
    abs_poa_features.csv (+ .meta.json)     one row per ABS Postal Area (2,643)
    cer_abs_panel_long.csv.gz (+ .meta.json) matched postcodes only: monthly
                                             installations, cumulative installations,
                                             dwelling denominator, adoption rate
    abs_join_report.json                     match counts, unmatched postcodes, checks

Load with load_joined(), not pd.read_csv, so postcodes stay 4-character strings.

Decisions
---------
1. Join key. ABS codes are "POA" + 4 digits; the prefix is stripped and the
   postcode kept as a string. CER postcode 0000 (placeholder) is excluded.
2. Denominator. Adoption rate = cumulative installations since Apr 2001 ÷
   occupied private dwellings (G36 OPDs_Tot_OPDs_Dwellings). The denominator is
   2021 and static against a 2001–2026 numerator; rates before 2021 are relative
   to the 2021 housing stock. Rates can exceed 1 (commercial systems, replacements,
   few occupied dwellings); these rows are flagged, not dropped.
3. Medians. ABS reports 0 where a median or average is not available (small
   areas). Zeros in G02 medians/averages are set to missing.
4. State. Taken from the ABS POA name ("2000, NSW"); the 15 cross-border POAs
   ("2620 crosses NSW/ACT") take the first state listed and carry cross_border_flag.
   The two non-geographic ABS codes (no usual address, migratory) get no state.
5. Small areas. Postcodes with Tot_P_P < 100 are flagged (small_pop_flag), not
   dropped; ABS perturbs every cell, so their shares and medians are noisy.
6. Leakage. Census features describe 2021. Used as predictors of pre-2021
   adoption they are retrospective explanation (Experiment A), not forecasting
   (Experiment B, history only). Keep the two designs separate in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POSTCODE_COL = "Small Unit Installation Postcode"
ABS_KEY = "POA_CODE_2021"

G01_COLS = ["Tot_P_P"]
G02_COLS = ["Median_age_persons", "Median_tot_hhd_inc_weekly", "Average_household_size",
            "Median_mortgage_repay_monthly", "Median_rent_weekly"]
G36_COLS = ["OPDs_Separate_house_Dwellings", "OPDs_SD_r_t_h_th_Tot_Dwgs", "OPDs_Flt_apart_Tot_Dwgs",
            "OPDs_Tot_OPDs_Dwellings", "Total_PDs_Dwellings"]
G37_COLS = ["O_OR_Total", "O_MTG_Total", "R_Tot_Total", "Total_Total"]
DENOM = "OPDs_Tot_OPDs_Dwellings"


def _read_abs(path: Path, cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in cols if c not in header]
    if missing:
        raise KeyError(f"{path.name}: columns not found {missing}; check the short-header DataPack")
    df = pd.read_csv(path, dtype={ABS_KEY: str}, usecols=[ABS_KEY] + cols)
    df[POSTCODE_COL] = df[ABS_KEY].str[3:]
    return df[[POSTCODE_COL] + cols].set_index(POSTCODE_COL)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).astype("float64")


def build_features(abs_dir: Path) -> pd.DataFrame:
    g01 = _read_abs(abs_dir / "2021Census_G01_AUST_POA.csv", G01_COLS)
    g02 = _read_abs(abs_dir / "2021Census_G02_AUST_POA.csv", G02_COLS)
    g36 = _read_abs(abs_dir / "2021Census_G36_AUST_POA.csv", G36_COLS)
    g37 = _read_abs(abs_dir / "2021Census_G37_AUST_POA.csv", G37_COLS).rename(
        columns={"Total_Total": "G37_Total_dwellings"})

    area = pd.read_csv(abs_dir / "2021_POA_area_sqkm.csv", dtype={"Census_Code_2021": str})
    area[POSTCODE_COL] = area["Census_Code_2021"].str[3:]
    # Names are "2000, NSW" or "2620 crosses NSW/ACT"; cross-border POAs take the first state listed.
    name = area["Census_Name_2021"]
    area["state_abs"] = name.str.extract(r"^\d{4}(?:, | crosses )([A-Z]+)", expand=False)
    area["cross_border_flag"] = name.str.contains(" crosses ", regex=False).astype("int8")
    area = area.set_index(POSTCODE_COL)[["state_abs", "cross_border_flag", "Area sqkm"]].rename(
        columns={"Area sqkm": "area_sqkm"})

    f = g01.join([g02, g36, g37, area], how="outer")
    f[G02_COLS] = f[G02_COLS].replace(0, np.nan)

    f["pop_density_per_sqkm"] = _safe_div(f["Tot_P_P"], f["area_sqkm"])
    f["detached_share"] = _safe_div(f["OPDs_Separate_house_Dwellings"], f[DENOM])
    f["semidetached_share"] = _safe_div(f["OPDs_SD_r_t_h_th_Tot_Dwgs"], f[DENOM])
    f["apartment_share"] = _safe_div(f["OPDs_Flt_apart_Tot_Dwgs"], f[DENOM])
    f["unoccupied_share"] = 1 - _safe_div(f[DENOM], f["Total_PDs_Dwellings"])
    f["owner_occupied_share"] = _safe_div(f["O_OR_Total"] + f["O_MTG_Total"], f["G37_Total_dwellings"])
    f["rented_share"] = _safe_div(f["R_Tot_Total"], f["G37_Total_dwellings"])
    f["small_pop_flag"] = (f["Tot_P_P"] < 100).astype("int8")
    return f.sort_index().reset_index()


def build_joined(panel: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    panel = panel[panel[POSTCODE_COL] != "0000"].copy()
    cer_pc = set(panel[POSTCODE_COL])
    abs_pc = set(feats[POSTCODE_COL])
    matched = sorted(cer_pc & abs_pc)
    unmatched = sorted(cer_pc - abs_pc)

    totals = panel.groupby(POSTCODE_COL)["installations"].sum()
    report = {
        "cer_postcodes_excl_0000": len(cer_pc),
        "abs_postal_areas": len(abs_pc),
        "matched_postcodes": len(matched),
        "unmatched_cer_postcodes": len(unmatched),
        "unmatched_installations": int(totals.loc[unmatched].sum()),
        "unmatched_installation_share": round(float(totals.loc[unmatched].sum() / totals.sum()), 6),
        "abs_areas_without_cer_rows": len(abs_pc - cer_pc),
        "unmatched_postcode_list": unmatched,
    }

    j = panel[panel[POSTCODE_COL].isin(matched)].sort_values([POSTCODE_COL, "month"]).copy()
    j["cum_installations"] = j.groupby(POSTCODE_COL)["installations"].cumsum().astype("int64")
    denom = feats.set_index(POSTCODE_COL)[DENOM]
    j["dwellings_opd_2021"] = j[POSTCODE_COL].map(denom)
    j["adoption_rate"] = _safe_div(j["cum_installations"], j["dwellings_opd_2021"])
    j["rate_gt_1_flag"] = (j["adoption_rate"] > 1).astype("int8")

    last = j[j["month"] == j["month"].max()]
    report.update({
        "matched_postcodes_zero_dwellings": int((denom.reindex(matched).fillna(0) <= 0).sum()),
        "latest_month": str(j["month"].max().date()),
        "postcodes_rate_gt_1_at_latest_month": int(last["rate_gt_1_flag"].sum()),
        "median_adoption_rate_at_latest_month": round(float(last["adoption_rate"].median()), 4),
    })
    return j.reset_index(drop=True), report


def _sidecar(path: Path, description: str, df: pd.DataFrame, extra: dict) -> None:
    meta = {
        "file": path.name,
        "description": description,
        "source": "ABS 2021 Census General Community Profile DataPack, Postal Areas, Australia (short header); "
                  "CER small-scale installation postcode data, current as at 31 July 2026",
        "string_columns": [POSTCODE_COL] + (["state_abs"] if "state_abs" in df.columns else []),
        "date_columns": ["month"] if "month" in df.columns else [],
        "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "built_by": "src/join_abs.py",
        "load_with": "src/join_abs.py: load_joined() / load_features()",
        **extra,
    }
    path.with_name(path.name.split(".")[0] + ".meta.json").write_text(json.dumps(meta, indent=2))


def load_features(path: str | Path = "data/processed/abs_poa_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={POSTCODE_COL: str, "state_abs": str})
    assert df[POSTCODE_COL].str.len().eq(4).all(), "postcode lost its leading zeros"
    return df


def load_joined(path: str | Path = "data/processed/cer_abs_panel_long.csv.gz",
                features: str | Path | None = None) -> pd.DataFrame:
    """Load the joined panel; pass features=<abs_poa_features.csv> to attach Census features."""
    path = Path(path)
    meta = json.loads(path.with_name(path.name.split(".")[0] + ".meta.json").read_text())
    df = pd.read_csv(path, dtype={POSTCODE_COL: str}, parse_dates=["month"])
    assert df[POSTCODE_COL].str.len().eq(4).all(), "postcode lost its leading zeros"
    assert len(df) == meta["shape"]["rows"], "row count does not match sidecar"
    if features is not None:
        df = df.merge(load_features(features).drop(columns=[DENOM]), on=POSTCODE_COL, how="left")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/cer_solar_panel_long.csv")
    ap.add_argument("--abs", default="data/raw/abs")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_panel import load_panel

    panel = load_panel(args.panel)
    feats = build_features(Path(args.abs))
    joined, report = build_joined(panel, feats)

    fpath = out / "abs_poa_features.csv"
    feats.to_csv(fpath, index=False)
    _sidecar(fpath, "ABS 2021 Census features, one row per Postal Area", feats,
             {"denominator": DENOM, "zero_medians_set_missing": G02_COLS,
              "caution": "2021 snapshot; retrospective explanation only when used for pre-2021 targets"})

    jpath = out / "cer_abs_panel_long.csv.gz"
    joined.to_csv(jpath, index=False, date_format="%Y-%m-%d", compression="gzip")
    _sidecar(jpath, "CER panel joined to ABS 2021 dwellings: adoption-rate target, matched postcodes only", joined,
             {"target": "adoption_rate = cum_installations / dwellings_opd_2021",
              "known_issues": {
                  "static_denominator": "2021 occupied private dwellings used for every month 2001-2026",
                  "rate_above_1": "flagged in rate_gt_1_flag; commercial systems and replacements are not separable",
                  "registration_lag": "last ~12 months undercount (see cer_solar_panel_long.meta.json)",
                  "cumulative_start": "cumulative counts start Apr 2001 (CER coverage start)"}})

    (out / "abs_join_report.json").write_text(json.dumps(report, indent=2))
    print(f"matched {report['matched_postcodes']:,} of {report['cer_postcodes_excl_0000']:,} CER postcodes; "
          f"unmatched {report['unmatched_cer_postcodes']} holding {report['unmatched_installations']:,} installations "
          f"({report['unmatched_installation_share']:.2%})")
    print(f"joined panel rows={len(joined):,}; median adoption rate at {report['latest_month']}: "
          f"{report['median_adoption_rate_at_latest_month']:.3f}; "
          f"postcodes above 1: {report['postcodes_rate_gt_1_at_latest_month']}")


if __name__ == "__main__":
    main()
