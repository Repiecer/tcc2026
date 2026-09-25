import cdsapi
import cdsswarm
tasks = []
years = [str(y) for y in range(1981, 2026)]
months = ['5', '6', '7', '8']

for year in years:
    for month in months:
        tasks.append(
            cdsswarm.Task(
                dataset='reanalysis-era5-single-levels',
                request={
                    'product_type': ['reanalysis'],
                    'variable': ['sea_surface_temperature', 'mean_sea_level_pressure'],
                    'year': [year],
                    'month': [month],
                    'day': [str(i).zfill(2) for i in range(1, 32)],
                    'time': [f'{i:02d}:00' for i in range(24)],
                    'area': [55, 70, 15, 140],
                    'data_format': 'netcdf',
                },
                target=f'../data/era5_sst_mslp_{year}_{month}.nc'
            )
        )
        tasks.append(
            cdsswarm.Task(
                dataset='reanalysis-era5-pressure-levels',
                request={
                    'product_type': ['reanalysis'],
                    'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind'],
                    'pressure_level': ['500', '850'],
                    'year': [year],
                    'month': [month],
                    'day': [str(i).zfill(2) for i in range(1, 32)],
                    'time': [f'{i:02d}:00' for i in range(24)],
                    'area': [55, 70, 15, 140],
                    'data_format': 'netcdf',
                },
                target=f'../data/era5_geo_uv_{year}_{month}.nc'
            )
        )
        tasks.append(
            cdsswarm.Task(
                dataset='derived-era5-land-daily-statistics',
                request = {
                    'variable': '2m_temperature',
                    'year': [year],
                    'month': [month],
                    'day': [str(i).zfill(2) for i in range(1, 32)],
                    'daily_statistic': 'daily_maximum',
                    'time_zone':'utc+08:00',
                    'frequency':'1_hourly',
                    'area': [55, 70, 15, 140],
                    'data_format': 'netcdf',
                },

                target=f'../data/era5_t2m_{year}_{month}.nc'
            )
        )

results = cdsswarm.download(tasks, num_workers=2)
