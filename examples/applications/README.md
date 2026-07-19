# Applications

marEx applied outside marine heatwaves. Each example here keeps the detect → track chain of the tutorials and changes the
variable, the tail or the domain.

## European Wind Droughts

`wind_drought_europe.ipynb` builds a 45-year (1980-2024) catalogue of low-wind spells over Europe from ERA5. Extremes are
days on which the daily capacity factor falls below its day-of-year 10th percentile (`tail="lower"`, 1991-2020 normal),
and `marEx.regional_tracker` follows each spell through space and time, including merges and splits.

### Input

`prepare_era5_wind.sh` converts ERA5 hourly 10 m wind (one GRIB file per day and component, as in the DKRZ
pool) into the daily zarr store the notebook reads. The capacity factor is computed hourly from a generic 100 m turbine
and then averaged, because the power curve is non-linear. It needs CDO (>= 2.0) and is resumable.

```bash
./prepare_era5_wind.sh /pool/data/ERA5/E5/sf/an/1H /path/to/out   # ERA5_DIR OUT_DIR [FIRST_YEAR LAST_YEAR NPAR]
export MAREX_ERA5_WIND=/path/to/out/era5_wind_europe_daily.zarr
```

### Run

One 64 GB shared node (8 workers × 4 threads × 6 GB). Detection took 1.6 min and tracking 4.5 min, as reported by
the notebook's stage summaries. Set `SMOKE = True` in the parameters cell to run on 1989-1996 only.
