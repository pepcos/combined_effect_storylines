import iris
import iris.analysis
import iris.analysis.cartography
import numpy as np
import os    
from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
)
import json
import xarray as xr
import iris.analysis.stats

from esmvalcore.preprocessor import extract_region
import cartopy.crs as ccrs

import matplotlib.pyplot as plt

def eval(config):
    with open(os.path.join(config["work_dir"], "correlations.json"), "r") as f:
        correlation_dict = json.load(f)
    corrs=np.array(list(correlation_dict.values()))
    mn=np.mean(corrs)
    std=np.std(corrs)
    best={k:v for k,v in correlation_dict.items() if v>mn+std}
    worst={k:v for k,v in correlation_dict.items() if v<mn+std}

def compute(config):
    dataset_dict=group_metadata(config["input_data"].values(), "alias")
    ref = "OBS"
    ref_metadata=group_metadata(dataset_dict[ref], "variable_group")
    ref_trend_cube = iris.load_cube(ref_metadata["tos"][0]["filename"])
    ref_trend_cube.convert_units("year**-1 K")

    ref_oni_20102014_cube, ref_oni_20152019_cube, ref_oni_20202024_cube = load_oni(ref_metadata)

    correlation_dict = {}
    for alias, alias_dict in dataset_dict.items():
        if "OBS" in alias:
            plot_trend(ref_trend_cube, alias, config, 1.)
            plot_oni(ref_oni_20102014_cube, ref_oni_20152019_cube, ref_oni_20202024_cube, alias, config)
            continue
        
        var_metadata=group_metadata(alias_dict, "variable_group")
        trend_cube = iris.load_cube(var_metadata["tos"][0]["filename"])
        trend_cube.convert_units("year**-1 K")

        region=[120,290,-70,70]
        # pattern_correlation = compute_pattern_correlation(trend_cube, ref_trend_cube, region)
        pattern_correlation = call_pearson(trend_cube, ref_trend_cube, region)

        plot_trend(trend_cube, alias, config, pattern_correlation)

        oni_20102014_cube, oni_20152019_cube, oni_20202024_cube = load_oni(var_metadata)
        plot_oni(oni_20102014_cube, oni_20152019_cube, oni_20202024_cube, alias, config)
        print(alias)

        correlation_dict[alias] = float(pattern_correlation)
        print(f"cor({alias}, {ref}): {pattern_correlation}")
    plot_correlations(correlation_dict, config)
    with open(os.path.join(config["work_dir"], "correlations.json"), "w") as f:
        json.dump(correlation_dict, f)

def load_oni(metadata):
    oni_20102014_cube = iris.load_cube(metadata["oni_2010-2014"][0]["filename"])
    oni_20152019_cube = iris.load_cube(metadata["oni_2015-2019"][0]["filename"])
    oni_20202024_cube = iris.load_cube(metadata["oni_2020-2024"][0]["filename"])
    return oni_20102014_cube, oni_20152019_cube, oni_20202024_cube

def compute_pattern_correlation(trend_cube, ref_trend_cube, region):
    cube_a = extract_region(trend_cube, region[0], region[1], region[2], region[3])
    cube_b = extract_region(ref_trend_cube, region[0], region[1], region[2], region[3])
    r=iris.analysis.stats.pearsonr(cube_a, 
                                    cube_b,
                                    corr_coords=["latitude", "longitude"],
                                    weights=iris.analysis.cartography.area_weights(cube_a), 
                                    common_mask=True)
    return r.data

def correlation_scheme(x, y, weights):
    # UNCENTERED
    weighted_sum_xy = (x * y * weights).sum(dim=("lat", "lon"))
    weighted_sum_x2 = ((x ** 2) * weights).sum(dim=("lat", "lon"))
    weighted_sum_y2 = ((y ** 2) * weights).sum(dim=("lat", "lon"))

    return weighted_sum_xy, weighted_sum_x2, weighted_sum_y2

# Function to calculate uncentered Pearson correlation
def uncentered_pearson_correlation(x, y, weights, scheme):
    # Compute the spatial mean of ds1 for each time step and ds2 (which is time-independent)
    if scheme == "centred":
        mean_x = (x * weights).sum(dim=("lat", "lon")) / weights.sum(dim=("lat", "lon"))
        mean_y = (y * weights).sum(dim=("lat", "lon")) / weights.sum(dim=("lat", "lon"))

        # Center the datasets by subtracting the mean
        x = x - mean_x
        y = y - mean_y
    elif scheme != "uncentred":
        raise ValueError(f"scheme '{scheme}' is not defined. Options are 'uncentred' and 'centred'")
    
    # Weighted sums for numerator and denominator
    weighted_sum_xy, weighted_sum_x2, weighted_sum_y2 = correlation_scheme(x, y, weights)
    
    # Uncentered Pearson correlation for each time step
    correlation = weighted_sum_xy / np.sqrt(weighted_sum_x2 * weighted_sum_y2)
    
    return correlation

def call_pearson(array, reference, region):
    cube_a = extract_region(array, region[0], region[1], region[2], region[3])
    cube_b = extract_region(reference, region[0], region[1], region[2], region[3])

    ds1 = xr.DataArray.from_iris(cube_a) 
    ds2 = xr.DataArray.from_iris(cube_b)

    # Compute the weights based on latitude
    lat_radians = np.deg2rad(ds1.lat)  # Convert latitude to radians
    weights = np.cos(lat_radians)
    # Broadcast weights to match lat, lon dimensions
    weights = weights.broadcast_like(ds1)
    # Apply the function
    correlation = uncentered_pearson_correlation(ds1, ds2, weights, "uncentred").values

    return correlation

def plot_correlations(correlation_dict, config):
    plt.figure(figsize=(25,7))
    plt.scatter(np.arange(len(list(correlation_dict.values()))), list(correlation_dict.values()))
    # plt.scatter(np.arange(len(list(correlation_dict.values()))), list(correlation_dict.values()))
    plt.xticks(np.arange(len(list(correlation_dict.values()))), rotation=90)
    plt.gca().set_xticklabels(list(correlation_dict.keys()))
    plt.tight_layout()
    plt.xlim(-1,len(list(correlation_dict.values())))
    plt.hlines(y=0,xmin=plt.xlim()[0],xmax=plt.xlim()[-1], color="black")
    plt.grid(True)
    plt.savefig(os.path.join(config["plot_dir"], f"AAA_correlations.png"))

def plot_trend(trend_cube, alias, config, corr):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    xr.DataArray.from_iris(trend_cube).plot(transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title(f"{alias}\nSST trend 1980-2023. Corr={corr:.04}")
    plt.savefig(os.path.join(config["plot_dir"], f"{alias}_trend.png"))

def plot_oni(oni_20102014_cube, oni_20152019_cube, oni_20202024_cube, alias, config):
    plt.figure()
    xr.DataArray.from_iris(oni_20102014_cube).plot(label=f"{alias} 2010-2014")
    xr.DataArray.from_iris(oni_20152019_cube).plot(label=f"{alias} 2015-2019")
    xr.DataArray.from_iris(oni_20202024_cube).plot(label=f"{alias} 2020-2024")
    plt.xlim(plt.xlim()[0],plt.xlim()[-1])
    plt.legend()
    plt.hlines(y=0,xmin=plt.xlim()[0],xmax=plt.xlim()[-1], color="black")
    plt.hlines(y=-0.5,xmin=plt.xlim()[0],xmax=plt.xlim()[-1], color="black", linestyles="--", alpha=0.7)
    plt.hlines(y=0.5,xmin=plt.xlim()[0],xmax=plt.xlim()[-1], color="black", linestyles="--", alpha=0.7)
    plt.savefig(os.path.join(config["plot_dir"], f"{alias}_oni.png"))

def regrid_longitude_coord(cube):
    """Sorts the longitudes of the cubes from 0/360 degrees to -180/180"""
    coord = cube.coord("longitude")
    lon_extent = iris.coords.CoordExtent(coord, -180., 180., True, False)
    cube = cube.intersection(lon_extent)
    return cube

def main():
    with run_diagnostic() as config:
        compute(config)

if __name__ == "__main__":
    main()
