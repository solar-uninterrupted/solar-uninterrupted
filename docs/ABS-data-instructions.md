# ABS Census data — what to download and how to join it

Owner: Abraham (Data Lead). Needed before M7; not required for the M5 first draft.
Estimated time: 15–20 minutes to download, ~1 hour to join and check.

## 1. Where

ABS Census DataPacks: https://www.abs.gov.au/census/find-census-data/datapacks

Choose:
- Census year: **2021**
- DataPack type: **General Community Profile (GCP)**
- Geography: **Postal Areas (POA)**
- Area: **Australia** (one zip, all states)

Free, no registration. The zip contains one CSV per table plus a `Metadata/` folder with
the long column descriptors.

## 2. Which tables

| Table | File in the zip | What we take from it | Used for |
|---|---|---|---|
| G01 | `2021Census_G01_AUST_POA.csv` | `Tot_P_P` (total persons) | population density (with area) |
| G02 | `2021Census_G02_AUST_POA.csv` | `Median_age_persons`, `Median_tot_hhd_inc_weekly`, `Average_household_size`, `Median_mortgage_repay_monthly`, `Median_rent_weekly` | income, age, household size features |
| G36 | `2021Census_G36_AUST_POA.csv` | `OPDs_Separate_house_Dwellings`, `OPDs_SD_r_t_h_th_Tot_Dwgs` (semi-detached/terrace), `OPDs_Flt_apart_Tot_Dwgs` (flats/apartments), `OPDs_Tot_OPDs_Dwellings` (occupied private dwellings), `Total_PDs_Dwellings` (all private dwellings incl. unoccupied) | **dwelling denominator**; detached share; apartment share |
| G37 | `2021Census_G37_AUST_POA.csv` | tenure totals: `O_OR_Total` (owned outright), `O_MTG_Total` (owned with mortgage), `R_Tot_Total` (rented) | owner-occupied share |

Column names above were verified against the downloaded 2021 short-header pack on 2026-09-23 (G01, G02, G37 as listed; G36 uses the abbreviated labels shown). The general rule: confirm against
`Metadata/2021Census_geog_desc_1st_2nd_3rd_release.xlsx` and the G-table metadata sheet,
since ABS abbreviates differently table to table.

Optional, for the "static census" limitation: the **Time Series Profile (TSP)** DataPack at
POA level gives 2011 / 2016 / 2021 on a consistent basis for dwelling structure (T-tables),
which lets us test whether the denominator moved between censuses.

## 3. Join key

- ABS codes are `POA` + four digits, e.g. `POA0800`. Strip the prefix: `code.str[3:]`.
- Keep it as a **string** so it matches the panel's 4-character postcode.
- ABS Postal Areas approximate Australia Post postcodes (built from mesh blocks). Verified 2026-09-23:
  2,634 of 2,810 CER postcodes match a POA; the 176 unmatched are PO-box / delivery-centre
  postcodes holding 2,179 installations (0.05% of the total). Report this in the paper.
- Exclude CER postcode `0000` before joining.

```python
import pandas as pd
g36 = pd.read_csv("2021Census_G36_AUST_POA.csv", dtype={"POA_CODE_2021": str})
g36["postcode"] = g36["POA_CODE_2021"].str[3:]
```

## 4. Denominator decision (write this down in Section 4 of the paper)

Adoption rate = cumulative installations ÷ occupied private dwellings (G36 `Total_Total`).
State that the denominator is 2021 and therefore static against a 25-year numerator;
if the TSP tables are pulled, use 2011 / 2016 / 2021 values by period as a sensitivity check.

## 5. Privacy note

ABS applies small random adjustments to every cell. Totals may not sum exactly; small
postcodes will have noisy medians. Flag postcodes with `Tot_P_P < 100` rather than dropping them.
