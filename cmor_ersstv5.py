import numpy as np
import iris
import os
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)

# CMORIZE ERSST: 

def runcmd(cmd, verbose = False):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def main():
    if len(sys.argv) < 2:
        raise ValueError("A path to the observations directory is needed")

    obs_path=sys.argv[1] #"/gpfs/scratch/bsc32/bsc032259/climate_data/OBS/Tier2/ERSSTv5/"

    if not os.path.isdir(obs_path):
        logger.info(f"creating directory {obs_path}")
        os.makedirs(obs_path)

    if not os.path.isfile(os.path.join(obs_path, "sst.mnmean.nc")):
        logger.info("downloading ERSST v5")
        runcmd(f"wget -P {obs_path} https://downloads.psl.noaa.gov//Datasets/noaa.ersst.v5/sst.mnmean.nc", verbose = True)

    cubes=iris.load(os.path.join(obs_path,"sst.mnmean.nc"))
    cube = [c for c in cubes if len(c.shape) == 3][0]

    cmorize(cube, obs_path)

def cmorize(cube, obs_path):
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
    logger.info(f"saving cube {os.path.join(obs_path, f'OBS_ERSSTv5_reanaly_v5_Omon_tos_{start_date}-{end_date}.nc')}")
    iris.save(cube, os.path.join(obs_path, f"OBS_ERSSTv5_reanaly_v5_Omon_tos_{start_date}-{end_date}.nc"))

if __name__ == "__main__":
    main()
