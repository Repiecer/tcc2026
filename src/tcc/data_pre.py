import xarray as xr
import matplotlib.pyplot as plt
ds = xr.open_dataset('e5-sp-201008.grib')
print(ds.dims)




