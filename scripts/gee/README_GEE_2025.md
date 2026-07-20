# GEE 2025 Topup Scripts — TOP Digital Twin

Five Google Earth Engine scripts extending satellite and climate data from 2024 to 2025.
Run each script in the [GEE Code Editor](https://code.earthengine.google.com).

## Scripts

| Script | Data source | Output per zone | GEE collection |
|--------|-------------|-----------------|----------------|
| `gee_01_ERA5_topup_2025.js` | ERA5-Land daily temperature | 365 rows | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| `gee_02_CHIRPS_2025.js`     | CHIRPS pentad rainfall | ~73 rows | `UCSB-CHG/CHIRPS/PENTAD` |
| `gee_03_S2_NDVI_2025.js`   | Sentinel-2 NDVI/EVI | ~24 composites | `COPERNICUS/S2_SR_HARMONIZED` |
| `gee_04_MODIS_NDVI_2025.js`| MODIS NDVI/EVI (16-day) | ~46 composites | `MOD13Q1` + `MYD13Q1` |
| `gee_05_MODIS_LST_2025.js` | MODIS LST (8-day) | ~46 composites | `MOD11A2` |

All scripts export to Google Drive folder: **`TOP_Digital_Twin_GEE_2025`**  
One CSV per zone × per script = 17 × 5 = **85 CSV files** total.

---

## Running the scripts

1. Open [code.earthengine.google.com](https://code.earthengine.google.com)
2. Copy-paste each script → click **Run**
3. Click the **Tasks** tab (top-right panel)
4. Click **Run** on each queued export task (17 tasks per script)
5. Wait for exports to complete (~5–20 min per script depending on zone)

---

## Downloading and organizing

After GEE exports complete:

1. Open [Google Drive](https://drive.google.com) → find `TOP_Digital_Twin_GEE_2025` folder
2. Download all 85 CSVs
3. Organize into local subfolders exactly as shown:

```
C:\Users\masro\Downloads\GEE_2025\
  era5\       ← 17 files named *_ERA5topup_2025.csv
  chirps\     ← 17 files named *_CHIRPS_2025.csv
  s2\         ← 17 files named *_S2_NDVI_2025.csv
  modis\      ← 34 files: *_MODIS_NDVI_2025.csv + *_MODIS_LST_2025.csv
```

---

## Running Script 14 after downloading

```powershell
cd C:\Users\masro\Documents\TOP_Digital_Twin
python scripts/14_Satellite_Climate_Features.py
```

Script 14 will automatically detect the `GEE_2025` folders and append 2025 data
to the existing 2017-2024 archive before processing. If a folder is missing, it
prints a warning and continues with historical data only.

---

## Zone reference (17 production zones)

| Zone ID | Crop | Market | State |
|---------|------|--------|-------|
| T1_Kolar | Tomato | Kolar APMC | Karnataka |
| T2_Madanapalle | Tomato | Madanapalle APMC | Andhra Pradesh |
| T3_Nashik_Tomato | Tomato | Nashik APMC | Maharashtra |
| T4_Solan | Tomato | Solan APMC | Himachal Pradesh |
| T5_Navsari | Tomato | Navsari APMC | Gujarat |
| O1_Lasalgaon | Onion | Lasalgaon APMC | Maharashtra |
| O2_Pimpalgaon | Onion | Pimpalgaon APMC | Maharashtra |
| O3_Mahuva | Onion | Mahuva APMC | Gujarat |
| O6_Hubli | Onion | Hubli APMC | Karnataka |
| O7_Solapur | Onion | Solapur APMC | Maharashtra |
| O8_Manmad | Onion | Manmad APMC | Maharashtra |
| O9_Kurnool | Onion | Kurnool APMC | Andhra Pradesh |
| O10_Gondal | Onion | Gondal APMC | Gujarat |
| P1_Agra | Potato | Agra APMC | Uttar Pradesh |
| P2_Farrukhabad | Potato | Farrukhabad APMC | Uttar Pradesh |
| P3_Jalandhar | Potato | Jalandhar APMC | Punjab |
| P4_Bardhaman | Potato | Bardhaman APMC | West Bengal |

Zone geometry: 30 km radius buffer around each APMC market centroid.

---

## Column formats (must match for Script 14 compatibility)

**ERA5** (`*_ERA5topup_2025.csv`):
`zone_id, crop, market, state, date, year, month, doy, Tmax_C, Tmin_C, Tmean_C, flag_above30, flag_above35, flag_above38`

**CHIRPS** (`*_CHIRPS_2025.csv`):
`zone_id, crop, market, state, date, year, month, doy_pentad_start, rain_mean_mm, rain_sum_mm, rain_max_mm, frac_excess_rain`

**S2** (`*_S2_NDVI_2025.csv`):
`zone_id, crop, market, state, date_start, date_end, year, doy_start, n_scenes, NDVI, EVI, valid_px_frac`

**MODIS NDVI** (`*_MODIS_NDVI_2025.csv`):
`zone_id, crop, market, state, date, year, month, doy, NDVI, EVI, n_valid_px`

**MODIS LST** (`*_MODIS_LST_2025.csv`):
`zone_id, crop, market, state, date, year, month, doy, LST_mean_C, LST_max_C, frac_above30, frac_above35, frac_above38, n_valid_px`
