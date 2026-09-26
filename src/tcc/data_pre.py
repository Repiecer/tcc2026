import xarray as xr
import matplotlib.pyplot as plt
ds = xr.open_dataset('data/raw/tmax/tmax_1982_06.nc', engine='netcdf4')

print(ds.dims)
ds['t2m'].sel(time='1982-06-27T00:00:00.000000000').plot(cmap='RdBu_r') # type: ignore
plt.show()



