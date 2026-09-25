import cdsapi

client = cdsapi.Client()
dataset = 'derived-era5-land-daily-statistics'

variables = ['2m_temperature']
request = {
    'variable': variables,
    'year': ['2024'],
    'month': ['07'],
    'day': [str(i).zfill(2) for i in range(1, 32)],
    'daily_statistic': 'daily_maximum',
    'time_zone':'utc+08:00',
    'frequency':'1_hourly',
    'data_format': 'netcdf',
}

client.retrieve(dataset, request, './data/daily_tmax_202407.nc')

