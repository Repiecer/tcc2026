import cdsapi

client = cdsapi.Client()

dataset = 'reanalysis-era5-pressure-levels'
request={
    'product_type': ['reanalysis'],
    'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind'],
    'pressure_level': ['500', '850'],
    'year': ['1990'],
    'month': ['03'],
    'day': ['05'],
    'time': ['00:00'],
    'area': [55, 70, 15, 140],
    'data_format': 'netcdf',
}
target = 'test.nc'

client.retrieve(name=dataset, request=request, target=target)



