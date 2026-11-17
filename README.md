# Building morphology and spring phenological temperature sensitivity

This collection contains city-level zonal data, analysis scripts, and source tables for the figures. The original rasters and the complete pixel-level NPZ are external inputs; city summaries cannot replace pixel data in pixel-level matching.

## Data

- `data/zonal/clean_zonal_sos_city_rural.csv`: 2,634 historical city–zone summaries for 664 city IDs; each row is one city and one zone (0 city; 1, 2, 3 rural rings). This is the wider historical source, not the final manuscript sample; its GDP values retain the historical eight-snapshot aggregation. All 638 study city IDs occur in its city zone. Its `Sensitivity_sos_Ridge_T_*` fields are Ridge summaries, not the raw-unit OLS ST outcome.
- `data/zonal/raw_model_input_638.csv`: 638 city-level analysis inputs; one row per city. `ST_OLS_days_per_C` is signed spring ST in days per °C. GDP in this 638-city table retains the historical eight-snapshot aggregation.
- `data/zonal/raw_model_input_636_environment.csv`: 636 city-level prediction inputs after excluding city IDs 35935 and 39895. Its GDP column now uses the finite-value five-snapshot mean for 2000, 2005, 2010, 2015 and 2020. The Figure 2 workflow applies at least 50 valid ST pixels, yielding 611 cities; positive ST values are retained.
- `data/zonal/GDP_2000_2020_five_snapshot_mean.csv`: the same 636 city IDs and their five-snapshot GDP mean. These are five discrete snapshots, not 21 annual observations.
- `data/zonal/experimental_cities_raw_building_height_quantiles.csv`: 17 quantile rows of original building-vector heights within study cities, in metres; the 1/3 and 2/3 quantiles are 7.456394 and 11.096033 m.
- `data/figure1/`: 638 city records and derived city partial-correlation categories and figure summaries; four categories contain 590 estimable cities.
- `data/figure2/`: 611 archived city out-of-fold feature rows, SHAP values and predictions, plus derived figure summaries. These retain the earlier GDP values and have not been retrained with the updated 636-city GDP input. ST is signed days per °C.
- `data/figure3/`: 622-city SEM input, 638-city GAM input, results and two pixel-pair tables (246,300 and 63,420 matched pairs). Each pair row is one matched pair, not a city summary.
- `data/figure3e_station_pairs.csv`: 373 matched station pairs; one row per pair.
- `data/figure4/`: model comparison inputs and results; `city_observed_ST.csv` has 586 city rows, and `city_ensemble_ST_paired.csv` has 1,172 city–ensemble rows.
- `data/supplementary_figure1/`, `supplementary_figure2/`, `supplementary_figure3/`: figure tables for city distributions, matching balance and stations; `retained_station_inventory.csv` lists 746 stations.

## Code and inputs

Install the Python packages in `requirements.txt`; SEM additionally requires R and `lavaan`. `code/ANALYSIS_NOTES.md` identifies the input paths for each analysis. The figure tables include both analysis inputs and derived results, as indicated above.

`code/building_height_quantiles.py` computes the original building-vector height quantiles; set `MORPH_CITY_GPKG`, `MORPH_BUILDING_VECTOR_DIR`, and `MORPH_HEIGHT_OUTPUT_DIR` before running. It aggregates building heights within the city polygons, not city mean heights or 4×4 subpixels.

The package starts from prepared summary inputs; GDP and zonal preprocessing code is not included. Pixel-level analyses require their original external inputs; the approximately 157 MiB pixel NPZ is not included.

```bash
python code/experiments.py
python Python.py --output output/figure3e_check.png
```

The first command checks analysis entries; the second recreates the Figure 3e paired wind contrast from the included station pairs.
