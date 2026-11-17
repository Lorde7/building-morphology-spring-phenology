


   
                      

     
                                
                          
               
                                         
                          
      
                    
                 
                    
   

import os
import re
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm







GUB_CITY_GPKG = os.environ.get("MORPH_CITY_GPKG", "")



BUILDING_VECTOR_DIR = os.environ.get("MORPH_BUILDING_VECTOR_DIR", "")

OUT_DIR = os.environ.get("MORPH_HEIGHT_OUTPUT_DIR", "")

OUT_QUANTILES_CSV = os.path.join(
    OUT_DIR,
    "experimental_cities_raw_building_height_quantiles.csv"
)

OUT_CITY_STATS_CSV = os.path.join(
    OUT_DIR,
    "experimental_cities_raw_building_height_city_stats.csv"
)

OUT_FILE_STATS_CSV = os.path.join(
    OUT_DIR,
    "experimental_cities_raw_building_height_file_stats.csv"
)



HEIGHT_FIELD = None

HEIGHT_FIELD_CANDIDATES = [
    "height", "HEIGHT", "Height",
    "h", "H",
    "mean_h", "Mean_H", "MEAN_H",
    "bldg_h", "Bldg_H", "BLDG_H",
    "building_h", "Building_H", "BUILDING_H",
    "height_m", "HEIGHT_M", "Height_m",
    "elevation", "Elevation",
]


VECTOR_PATTERNS = [
    "*.shp",
    "*.gpkg",
    "*.geojson",
]


MIN_HEIGHT = 0.0
MAX_HEIGHT = 1000.0




ASSIGN_METHOD = "representative_point_within"


ASSUME_BUILDING_CRS_IF_MISSING = "EPSG:4326"






def get_city_uid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
       
                    
                                                          
       
    if "ORIG_FID" in gdf.columns:
        gdf["city_uid"] = gdf["ORIG_FID"].astype(np.int32)
        print("城市使用 ORIG_FID 作为 city_uid")

    elif "orig_fid" in gdf.columns:
        gdf["city_uid"] = gdf["orig_fid"].astype(np.int32)
        print("城市使用 orig_fid 作为 city_uid")

    elif "FID" in gdf.columns:
        gdf["city_uid"] = gdf["FID"].astype(np.int32)
        print("城市使用 FID 作为 city_uid")

    elif "OBJECTID" in gdf.columns:
        gdf["city_uid"] = gdf["OBJECTID"].astype(np.int32)
        print("城市使用 OBJECTID 作为 city_uid")

    else:
        gdf["city_uid"] = np.arange(len(gdf), dtype=np.int32)
        print("警告：城市没有 ORIG_FID/orig_fid/FID/OBJECTID，只能临时使用 range 编号。")

    return gdf


def list_vector_files(root_dir: str):
    root = Path(root_dir)

    files = []
    for pat in VECTOR_PATTERNS:
        files.extend(root.rglob(pat))

    files = sorted([str(f) for f in files])

    return files


def try_import_pyogrio():
    try:
        import pyogrio
        return pyogrio
    except Exception:
        return None


PYOGRIO = try_import_pyogrio()


def read_vector_info(fp):
       
                               
       
    if PYOGRIO is not None:
        try:
            info = PYOGRIO.read_info(fp)
            fields = list(info.get("fields", []))
            crs = info.get("crs", None)

            bounds = info.get("total_bounds", None)
            if bounds is None:
                bounds = info.get("bounds", None)

            if bounds is not None:
                bounds = tuple(float(x) for x in bounds)

            return {
                "fields": fields,
                "crs": crs,
                "bounds": bounds
            }

        except Exception:
            pass

    
    try:
        gdf_head = gpd.read_file(fp, rows=1)
        fields = [c for c in gdf_head.columns if c != "geometry"]
        crs = gdf_head.crs
        bounds = None

        return {
            "fields": fields,
            "crs": crs,
            "bounds": bounds
        }

    except Exception as e:
        return {
            "fields": [],
            "crs": None,
            "bounds": None,
            "error": str(e)
        }


def detect_height_field(fields):
    if HEIGHT_FIELD is not None:
        if HEIGHT_FIELD in fields:
            return HEIGHT_FIELD
        else:
            raise ValueError(
                f"手动指定的 HEIGHT_FIELD={HEIGHT_FIELD} 不在字段中。"
                f"该文件字段为: {fields}"
            )

    lower_map = {str(c).lower(): c for c in fields}

    for cand in HEIGHT_FIELD_CANDIDATES:
        if cand in fields:
            return cand

        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    
    for c in fields:
        cl = str(c).lower()
        if "height" in cl or cl in ["h", "meanh", "bldgh"]:
            return c

    return None


def read_buildings_with_bbox(fp, bbox, columns):
       
                     
                     
       
    if PYOGRIO is not None:
        try:
            return PYOGRIO.read_dataframe(
                fp,
                bbox=bbox,
                columns=columns
            )
        except Exception:
            pass

    try:
        return gpd.read_file(
            fp,
            bbox=bbox,
            columns=columns
        )

    except TypeError:
        
        return gpd.read_file(
            fp,
            bbox=bbox
        )


def clean_height_series(s):
    vals = pd.to_numeric(s, errors="coerce").astype("float32").values

    valid = np.isfinite(vals)
    valid &= vals > MIN_HEIGHT
    valid &= vals < MAX_HEIGHT

    return vals, valid


def calc_quantiles(vals):
    qs = [
        0.001, 0.005, 0.01, 0.02, 0.05,
        0.10, 0.25, 1/3, 0.50, 2/3,
        0.75, 0.90, 0.95, 0.98, 0.99,
        0.995, 0.999
    ]

    rows = []
    for q in qs:
        rows.append({
            "quantile": q,
            "percent": q * 100,
            "height_m": float(np.nanquantile(vals, q))
        })

    return pd.DataFrame(rows)


def calc_basic_stats(vals):
    vals = np.asarray(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "q33": np.nan,
            "median": np.nan,
            "q66": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }

    return {
        "n_valid": int(vals.size),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "q25": float(np.nanquantile(vals, 0.25)),
        "q33": float(np.nanquantile(vals, 1/3)),
        "median": float(np.nanquantile(vals, 0.50)),
        "q66": float(np.nanquantile(vals, 2/3)),
        "q75": float(np.nanquantile(vals, 0.75)),
        "q90": float(np.nanquantile(vals, 0.90)),
        "q95": float(np.nanquantile(vals, 0.95)),
        "q99": float(np.nanquantile(vals, 0.99)),
        "max": float(np.nanmax(vals)),
    }






def main():
    print("========== 读取实验城市矢量面 ==========")
    print(GUB_CITY_GPKG)

    cities = gpd.read_file(GUB_CITY_GPKG)
    cities = cities[cities.geometry.notna() & (~cities.geometry.is_empty)].copy()
    cities["geometry"] = cities.geometry.buffer(0)

    cities = get_city_uid(cities)

    print(f"原始城市面数量: {len(cities)}")
    print(f"唯一 city_uid 数: {cities['city_uid'].nunique()}")

    if cities["city_uid"].duplicated().any():
        print("检测到重复 city_uid，按 city_uid dissolve 成城市整体。")
        cities = cities.dissolve(by="city_uid", as_index=False)
        cities["geometry"] = cities.geometry.buffer(0)

    print(f"dissolve 后城市数量: {len(cities)}")
    print(f"城市 CRS: {cities.crs}")

    if cities.crs is None:
        raise ValueError("城市矢量没有 CRS，请先为 GUB_north30_clean.gpkg 定义 CRS。")

    print("\n========== 扫描建筑矢量文件 ==========")
    building_files = list_vector_files(BUILDING_VECTOR_DIR)

    print(f"建筑矢量根目录: {BUILDING_VECTOR_DIR}")
    print(f"找到建筑矢量文件数: {len(building_files)}")

    if len(building_files) == 0:
        raise FileNotFoundError(
            f"没有在 {BUILDING_VECTOR_DIR} 下找到建筑矢量文件。"
            f"请检查 BUILDING_VECTOR_DIR。"
        )

    global_height_chunks = []
    city_height_dict = defaultdict(list)
    file_records = []

    
    cities_by_crs_cache = {}

    for fp in tqdm(building_files, desc="处理建筑文件", unit="file"):
        info = read_vector_info(fp)

        fields = info.get("fields", [])
        bcrs = info.get("crs", None)
        bounds = info.get("bounds", None)

        if len(fields) == 0:
            file_records.append({
                "file": fp,
                "status": "failed_read_info",
                "height_field": None,
                "n_read": 0,
                "n_inside_city": 0
            })
            continue

        hfield = detect_height_field(fields)

        if hfield is None:
            file_records.append({
                "file": fp,
                "status": "height_field_not_found",
                "height_field": None,
                "n_read": 0,
                "n_inside_city": 0,
                "fields": "|".join(map(str, fields))
            })
            continue

        if bcrs is None:
            bcrs = ASSUME_BUILDING_CRS_IF_MISSING

        crs_key = str(bcrs)

        if crs_key not in cities_by_crs_cache:
            cities_src = cities[["city_uid", "geometry"]].copy().to_crs(bcrs)
            cities_src["geometry"] = cities_src.geometry.buffer(0)
            cities_by_crs_cache[crs_key] = cities_src
        else:
            cities_src = cities_by_crs_cache[crs_key]

        
        if bounds is not None:
            try:
                file_box = box(*bounds)
                cand_idx = list(cities_src.sindex.query(file_box, predicate="intersects"))

                if len(cand_idx) == 0:
                    file_records.append({
                        "file": fp,
                        "status": "no_city_overlap_by_bounds",
                        "height_field": hfield,
                        "n_read": 0,
                        "n_inside_city": 0
                    })
                    continue

                cand_cities = cities_src.iloc[cand_idx].copy()

            except Exception:
                cand_cities = cities_src

        else:
            cand_cities = cities_src

        
        read_bbox = tuple(float(x) for x in cand_cities.total_bounds)

        try:
            bldg = read_buildings_with_bbox(
                fp,
                bbox=read_bbox,
                columns=[hfield, "geometry"]
            )

        except Exception as e:
            file_records.append({
                "file": fp,
                "status": f"failed_read_buildings: {e}",
                "height_field": hfield,
                "n_read": 0,
                "n_inside_city": 0
            })
            continue

        if bldg is None or len(bldg) == 0:
            file_records.append({
                "file": fp,
                "status": "empty_after_bbox",
                "height_field": hfield,
                "n_read": 0,
                "n_inside_city": 0
            })
            continue

        if bldg.crs is None:
            bldg = bldg.set_crs(bcrs, allow_override=True)

        
        vals, valid = clean_height_series(bldg[hfield])

        bldg = bldg.loc[valid, [hfield, "geometry"]].copy()
        bldg["_height_m"] = vals[valid].astype(np.float32)

        bldg = bldg[bldg.geometry.notna() & (~bldg.geometry.is_empty)].copy()

        if len(bldg) == 0:
            file_records.append({
                "file": fp,
                "status": "no_valid_height",
                "height_field": hfield,
                "n_read": int(len(vals)),
                "n_inside_city": 0
            })
            continue

        
        try:
            if ASSIGN_METHOD == "intersects":
                join_left = bldg[["_height_m", "geometry"]].copy()
                matched = gpd.sjoin(
                    join_left,
                    cand_cities[["city_uid", "geometry"]],
                    how="inner",
                    predicate="intersects"
                )

            else:
                
                
                points = gpd.GeoDataFrame(
                    {
                        "_height_m": bldg["_height_m"].values
                    },
                    geometry=bldg.geometry.representative_point(),
                    crs=bldg.crs
                )

                matched = gpd.sjoin(
                    points,
                    cand_cities[["city_uid", "geometry"]],
                    how="inner",
                    predicate="within"
                )

        except Exception as e:
            file_records.append({
                "file": fp,
                "status": f"failed_spatial_join: {e}",
                "height_field": hfield,
                "n_read": int(len(bldg)),
                "n_inside_city": 0
            })
            continue

        if matched is None or len(matched) == 0:
            file_records.append({
                "file": fp,
                "status": "no_building_inside_city",
                "height_field": hfield,
                "n_read": int(len(bldg)),
                "n_inside_city": 0
            })
            continue

        matched["_height_m"] = matched["_height_m"].astype(np.float32)

        
        hvals_all = matched["_height_m"].values.astype(np.float32)
        global_height_chunks.append(hvals_all)

        
        for cid, sub in matched.groupby("city_uid"):
            city_height_dict[int(cid)].append(
                sub["_height_m"].values.astype(np.float32)
            )

        file_records.append({
            "file": fp,
            "status": "ok",
            "height_field": hfield,
            "n_read": int(len(bldg)),
            "n_inside_city": int(len(matched))
        })

    
    
    

    file_stats = pd.DataFrame(file_records)
    file_stats.to_csv(OUT_FILE_STATS_CSV, index=False, encoding="utf-8-sig")

    print("\n文件级统计已保存:")
    print(OUT_FILE_STATS_CSV)

    if len(global_height_chunks) == 0:
        raise ValueError(
            "没有在实验城市内部统计到任何有效建筑高度。"
            "请检查：建筑矢量目录、height 字段名、CRS、城市面是否重叠。"
        )

    
    
    

    all_heights = np.concatenate(global_height_chunks).astype(np.float32)
    all_heights = all_heights[np.isfinite(all_heights)]
    all_heights = all_heights[(all_heights > MIN_HEIGHT) & (all_heights < MAX_HEIGHT)]

    print("\n========== 实验城市内部原始建筑高度全样本统计 ==========")
    print(f"有效建筑数量: {all_heights.size}")
    print(f"mean: {np.nanmean(all_heights):.4f}")
    print(f"std : {np.nanstd(all_heights):.4f}")
    print(f"min : {np.nanmin(all_heights):.4f}")
    print(f"max : {np.nanmax(all_heights):.4f}")

    qdf = calc_quantiles(all_heights)
    qdf.to_csv(OUT_QUANTILES_CSV, index=False, encoding="utf-8-sig")

    print("\n========== 实验城市内部原始建筑高度分位数 ==========")
    print(qdf.to_string(index=False))

    q33 = float(np.nanquantile(all_heights, 1/3))
    q66 = float(np.nanquantile(all_heights, 2/3))
    q25 = float(np.nanquantile(all_heights, 0.25))
    q50 = float(np.nanquantile(all_heights, 0.50))
    q75 = float(np.nanquantile(all_heights, 0.75))

    print("\n========== 建议的城市内部建筑高度分层阈值 ==========")

    print("\n三层 height strata:")
    print(f"  S1: 0 < H <= Q33 = {q33:.2f} m")
    print(f"  S2: Q33 < H <= Q66 = {q66:.2f} m")
    print(f"  S3: H > Q66 = {q66:.2f} m")

    print("\n四层 height strata:")
    print(f"  S1: 0 < H <= Q25 = {q25:.2f} m")
    print(f"  S2: Q25 < H <= Q50 = {q50:.2f} m")
    print(f"  S3: Q50 < H <= Q75 = {q75:.2f} m")
    print(f"  S4: H > Q75 = {q75:.2f} m")

    print("\nLCZ-like 固定阈值，可作为语义稳健性检验:")
    print("  low-rise : 0 < H <= 10 m")
    print("  mid-rise : 10 < H <= 25 m")
    print("  high-rise: H > 25 m")

    
    
    

    city_rows = []

    all_city_ids = sorted(cities["city_uid"].astype(int).unique())

    for cid in all_city_ids:
        chunks = city_height_dict.get(int(cid), [])

        if len(chunks) == 0:
            stats = calc_basic_stats(np.array([], dtype=np.float32))
        else:
            vals = np.concatenate(chunks).astype(np.float32)
            stats = calc_basic_stats(vals)

        row = {"city_id": int(cid)}
        row.update(stats)
        city_rows.append(row)

    city_stats = pd.DataFrame(city_rows)
    city_stats.to_csv(OUT_CITY_STATS_CSV, index=False, encoding="utf-8-sig")

    print("\n城市级建筑高度统计已保存:")
    print(OUT_CITY_STATS_CSV)

    print("\n========== 城市级统计概览 ==========")
    print(f"城市总数: {len(city_stats)}")
    print(f"有有效建筑高度的城市数: {(city_stats['n_valid'] > 0).sum()}")
    print(f"无有效建筑高度的城市数: {(city_stats['n_valid'] == 0).sum()}")

    print("\n完成。")


if __name__ == "__main__":
    if not all((GUB_CITY_GPKG, BUILDING_VECTOR_DIR, OUT_DIR)):
        raise SystemExit("Set MORPH_CITY_GPKG, MORPH_BUILDING_VECTOR_DIR, and MORPH_HEIGHT_OUTPUT_DIR")
    os.makedirs(OUT_DIR, exist_ok=True)
    main()