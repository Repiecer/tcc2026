import cdsapi
import cdsswarm

dataset = 'reanalysis-era5-single-levels'
variables = ['2m_temperature']
tasks = []
years = [str(y) for y in range(1981, 2026)]
months = ['5', '6', '7', '8']
for year in years:
    for month in months:
        tasks.append(
            cdsswarm.Task(
                dataset=dataset,
                request={
                    'product_type': ['reanalysis'],
                    'variable': variables,
                    'year': [f'{year}'],
                    'month': [f'{month}'],
                    'day': [str(i).zfill(2) for i in range(1, 32)],
                    'time': [f'{i:02d}:00' for i in range(24)],
                    'data_format': 'grib',
                },
                target=f'../data/era6_t2m_{year}_{month}.grib'
            )
        )

results = cdsswarm.download(tasks, num_workers=4)

def build_req(year):
    request = {
        'product_type': ['reanalysis'],
        'variable': variables,
        'year': [str(year)],
        'month': ['5', '6', '7', '8'],
        'day': [str(i).zfill(2) for i in range(1, 32)],  # 01-31
        'time': ['12:00'],       # 00:00-23:00 f'{i:02d}:00' for i in range(24)
        'data_format': 'grib',
    }
    return request
