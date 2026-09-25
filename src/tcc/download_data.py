import cdsapi

client = cdsapi.Client()

dataset = 'reanalysis-era5-single-levels'
variables = ['2m_temperature']
request = {
    'product_type': ['reanalysis'],
    'variable': variables,
    'year': [str(y) for y in range(1981, 2026)],
    'month': ['5', '6', '7', '8'],
    'day': [str(i).zfill(2) for i in range(1, 32)],  # 01-31
    'time': ['12:00'],       # f'{i:02d}:00' for i in range(24)
    'data_format': 'grib',
}

client.retrieve(dataset, request, '../data/e5-t2-201008.grib')
