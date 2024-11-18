***Recipe, diagnostic and ERSST cmorizer***

recipe: recipe_nino_trends.yml --> computes trends and ONI index for CMIP models and OBS

diagnostic: compares OBS and CMIP trends and ranks members

**DATASETS**

`GCMs`: all CMIP datasets are available from ESGF. Some are already included in the recipe defined through institution, experiment id (`historical/ssp245`) and wildcards (`"*"`). Some available datasets in ESGF might not be defined in the recipe, check https://aims2.llnl.gov/search/cmip6/.

`OBS`: the ERSST obs can be downloaded and cmorized through the `cmor_ersstv5.py` script. Usage:

``` bash
python cmor_ersst.py <path to OBS directory>
```