import numpy as np
import iris
import os
import pandas as pd
import sys

# CMORIZE ERSST: 
# wget 

obs_path=sys.argv[2] #"/gpfs/scratch/bsc32/bsc032259/climate_data/OBS/Tier2/ERSSTv5/"
cube=iris.load_cube(os.path.join(obs_path,"sst.mnmean.nc"))

coord = cube.coord(axis='T')
end = []
for cell in coord.cells():
    month = cell.point.month + 1
    year = cell.point.year
    if month == 13:
        month = 1
        year = year + 1
    end.append(cell.point.replace(month=month, year=year))
end = coord.units.date2num(end)
start = coord.points
coord.points = 0.5 * (start + end)
coord.bounds = np.column_stack([start, end])
dates=coord.units.num2date(coord.points)
start_date = f"{dates[0].year}{dates[0].month:02}"
end_date = f"{dates[-1].year}{dates[-1].month:02}"
iris.save(cube, os.path.join(obs_path, f"OBS_ERSSTv5_reanaly_v5_Omon_tos_{start_date}-{end_date}.nc"))