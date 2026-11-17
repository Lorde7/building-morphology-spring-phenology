# Original experiment entries

The original numerical algorithms are retained. Paths are configurable; preparation checked syntax and help only.

The packaged 636-city environmental input uses the finite-value mean of the five GDP snapshots for 2000, 2005, 2010, 2015 and 2020. The historical 638-city and 2,634-row city/rural tables retain the earlier eight-snapshot GDP aggregation. Archived Figure 2 OOF feature values, SHAP values and predictions retain the original model results and have not been retrained with the updated GDP input. The package starts from prepared summary inputs; GDP and zonal preprocessing code is not included.

| Result | Entry | Main inputs |
|---|---|---|
| Pixel OPL and climate windows | run_preseason_pipeline_wind_grid.py --steps opl preseason | Aligned monthly climate and phenology rasters |
| Pixel signed OLS ST | run_ols_sensitivity_raw.py --pheno sos --min-valid-years 11 | Existing phenology and preseason stacks; original 11-year threshold, not min13 |
| Fig. 1 partial correlations | experiment_figure1_partial_spearman.py | Original pixel NPZ, ST/OPL rasters, radiation pixels and 638-city metadata |
| Fig. 2 XGBoost/CV/OOF SHAP | experiment_figure2_xgboost_support.py | Original 636-city environmental table, 638-city support metadata, frozen folds and original settings in figure2_parameters.py; manuscript branch n_ge_50/climate24 |
| Fig. 3c/d pixel PSM | experiment_figure3_pixel_psm.py | Original full_pixels.csv.gz and full_pixels_LST.csv.gz workspace inputs |
| Fig. 3e station PSM | experiment_figure3_station_matching.py | 1177 candidates and covariates; root Python.py uses the included 373 matched pairs |
| Fig. 3f GAM | experiment_figure3_gam.py | Original 638-city primary input and archived comparison input; fixed lambda, 300 bootstrap replicates |
| Fig. 3b SEM | experiment_figure3_sem.py | R/lavaan, original 622-city input, model and reference parameters; final plot uses full/ML, 11 fitted regression paths, eight displayed |
| Fig. 4 model ST | experiment_figure4_city_annual.py | Original annual model/climate/phenology inputs and supporting modules |

`python code/experiments.py` checks entry syntax. Example execution: `python code/experiments.py --experiment 2 --config code/example_inputs.json --run`. Replace example paths with actual inputs. Other archived path keys are in EXPERIMENT_PATH_KEYS.json.

ST paths: set MORPH_CLIMATE_DIR and MORPH_ST_WORKSPACE to existing climate and a separate prepared workspace. The preserved current OLS script adds _min11 to its output name; the manuscript input lacks this suffix.

Pixel PSM uses the original Linux worker: `python code/experiment_figure3_pixel_psm.py --workspace /prepared/workspace --job code/example_pixel_psm_job.json`. Run psm_prepare for e/f, generated per-window match jobs, then psm_summary. It retains window-only strata, cross-city matching, logit caliper 0.1, optimal pairing and dyadic city-cluster inference (246300/63420 archived pairs).

SEM: `python code/experiment_figure3_sem.py --workdir /outputs/sem --archive data/figure3/sem_archive`. The archive needs source_file.txt (for example `input.csv`, resolved relative to the archive; absolute paths also work), models/original_paths__original.txt and standardized_parameters.csv. It calls original R/lavaan, retaining ML/MLR estimation and the full/reduced model definitions.

Typical dependencies: numpy, pandas, scipy, statsmodels, scikit-learn, xgboost, rasterio, numba, netCDF4, pygam and joblib. Raw external inputs are not supplied in the compact source-data tables. Model calculations use existing prepared inputs; collection and internal-record helpers are not included.
