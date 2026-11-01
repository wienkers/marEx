#!/bin/bash
# Prepare the input for wind_drought_europe.ipynb from ERA5 hourly 10 m wind.
#
#   ws10 : daily mean 10 m wind speed (m s-1)
#   cf   : daily mean capacity factor of a generic turbine at 100 m hub height, computed HOURLY:
#          power-law extrapolation U100 = U10 * (100/10)^(1/7), then a power curve with cut-in 3 m/s,
#          a cubic ramp to rated power at 12 m/s, and cut-out at 25 m/s.
#
# Input: one GRIB file per day and variable, as in the DKRZ ERA5 pool
#   <ERA5_DIR>/165/E5sf00_1H_<YYYY-MM-DD>_165.grb  (10 m u)
#   <ERA5_DIR>/166/E5sf00_1H_<YYYY-MM-DD>_166.grb  (10 m v)
# e.g. ERA5_DIR=/pool/data/ERA5/E5/sf/an/1H on Levante. Adapt the two paths in `one()` for another layout.
#
# Requires CDO (>= 2.0) and a Python environment with xarray, netCDF4 and zarr.
# Resumable: one NetCDF per day is kept and skipped on a rerun.
#
# Usage: prepare_era5_wind.sh ERA5_DIR OUT_DIR [FIRST_YEAR LAST_YEAR NPAR]
set -euo pipefail
ERA5_DIR=${1:?ERA5 directory}
OUT=${2:?output directory}
Y0=${3:-1980}
Y1=${4:-2024}
NPAR=${5:-8}
DAYS=$OUT/days
YRS=$OUT/years
mkdir -p "$DAYS" "$YRS"

d=$Y0-01-01
: > "$OUT/todo_all.txt"
while [ "$d" != "$((Y1 + 1))-01-01" ]; do echo "$d" >> "$OUT/todo_all.txt"; d=$(date -d "$d +1 day" +%F); done
while read -r d; do [ -s "$DAYS/$d.nc" ] || echo "$d"; done < "$OUT/todo_all.txt" > "$OUT/todo.txt"
echo "days: $(wc -l < "$OUT/todo_all.txt") in total, $(wc -l < "$OUT/todo.txt") to process"

# Hourly speed and power curve first, daily mean second (the power curve is non-linear).
# Box 12W-35E, 34N-72N; the reduced Gaussian grid is converted to its regular equivalent.
EXPR='ws10=sqrt(var165*var165+var166*var166);_h=ws10*1.3895;cf=(_h<3.0)?0.0:((_h<12.0)?(_h*_h*_h-27.0)/1701.0:((_h<25.0)?1.0:0.0));'
export ERA5_DIR DAYS EXPR
one() {
  d=$1
  cdo -s -f nc4 -z zip_1 -sellonlatbox,-12,35,34,72 -setgridtype,regular -daymean -expr,"$EXPR" \
      -merge "$ERA5_DIR/165/E5sf00_1H_${d}_165.grb" "$ERA5_DIR/166/E5sf00_1H_${d}_166.grb" "$DAYS/.$d.nc" \
    && mv "$DAYS/.$d.nc" "$DAYS/$d.nc"
}
export -f one
xargs -a "$OUT/todo.txt" -d '\n' -P "$NPAR" -I{} bash -c 'one {}'

missing=$(while read -r d; do [ -s "$DAYS/$d.nc" ] || echo "$d"; done < "$OUT/todo_all.txt" | wc -l)
if [ "$missing" != 0 ]; then echo "ERROR: $missing days missing, rerun to resume"; exit 1; fi

for y in $(seq "$Y0" "$Y1"); do [ -s "$YRS/$y.nc" ] || echo "$y"; done |
  xargs -P "$NPAR" -I{} bash -c 'cdo -s -O mergetime '"$DAYS"'/{}-*.nc '"$YRS"'/.{}.nc && mv '"$YRS"'/.{}.nc '"$YRS"'/{}.nc'

python - "$OUT" <<'EOF'
import glob
import sys

import xarray as xr

out = sys.argv[1]
ds = xr.open_mfdataset(sorted(glob.glob(f"{out}/years/[0-9]*.nc")), combine="by_coords")
ds = ds.drop_vars("time_bnds", errors="ignore")
ds["time"] = ds.time.dt.floor("D")
ds = ds.sortby("lat")
ds.ws10.attrs.update(long_name="daily mean 10 m wind speed", units="m s-1")
ds.cf.attrs.update(long_name="daily mean capacity factor, generic turbine, 100 m hub (alpha=1/7)", units="1")
ds = ds.chunk({"time": 365, "lat": -1, "lon": -1})
for v in ds.data_vars:
    ds[v].encoding = {}
store = f"{out}/era5_wind_europe_daily.zarr"
ds.to_zarr(store, mode="w", consolidated=True)
print("wrote", store, dict(ds.sizes))
EOF
