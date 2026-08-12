# MARS Images BA

A HiRISE stereo image processing and bundle-adjustment (BA) pipeline. The main entry point is the C++ binary `Mars`, controlled by a YAML config that enables a nine-step workflow from PDS preprocessing through feature/grid matching, inter-CCD ties, and BA / forward-intersection DEM.

## Overview

- PDS half-frame preprocessing and intra-CCD mosaicking
- Pyramid downsampling and hierarchical feature matching (fenfu)
- Semi-dense grid matching and cross-CCD tie-point generation
- Bundle adjustment (Ceres) and forward-intersection DEM (`*_FI.txt`)
- Supporting scripts for batch runs, disparity figures, FI point-cloud denoise, DTM evaluation, etc.

## Repository layout

```
mars_images_ba/
├── Mars/                 # Entry: Mars.cpp, draw_match.cpp
├── src/                  # Observation / ImageMatch / ImageProcess / BA / EO / …
├── include/myinc/        # PipelineConfig.h and other headers
├── data/                 # EO, IO, observations, results (relative to CWD)
│   ├── EO/
│   ├── IO/
│   ├── observedata/
│   └── result/
├── script/               # Batch, disparity plots, evaluation, SPICE/EO helpers
├── cfg.yaml              # Default pipeline config
├── cfg_*.yaml            # Per stereo-pair configs
├── CMakeLists.txt
└── out/                  # Debug visualization (e.g. fenfu_match*.tif)
```

## Dependencies

| Library | Role |
|---------|------|
| OpenCV | Image I/O, match visualization |
| Eigen3 | Linear algebra |
| Ceres Solver | BA / forward intersection |
| GDAL | Raster I/O |
| CSPICE (+ csupport) | Exterior orientation / SPICE |
| glog (optional gflags) | Logging |
| OpenMP | Parallel matching |
| PROJ (optional) | Projection-related ops |

CMake defaults for OpenCV / CSPICE point at local paths (e.g. `~/pkgs/opencv455`, `~/pkgs/cspice`); adjust `CMakeLists.txt` for your machine.

Python scripts additionally need numpy, matplotlib, GDAL/OpenCV, etc. There is no root `requirements.txt`.

## Build and run

```bash
mkdir -p build && cd build
cmake ..                          # default Release; override with -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
# binary: ../bin/Mars

cd ../bin
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/pkgs/opencv455/opencv/install/lib"
./Mars ../cfg.yaml                # optional; defaults to cfg.yaml
```

Run from `bin/` so relative paths resolve to `../data`, `../cfg.yaml`, and `../out`.

Batch stereo pairs:

```bash
bash script/run_mars_batch.sh
bash script/run_mars_batch.sh --dry-run
bash script/run_mars_batch.sh --continue
```

## Pipeline steps (`steps.*`)

Controlled by booleans in `cfg.yaml`, always in this order:

| Switch | Role |
|--------|------|
| `preprocess` | PDS `.img` → half-frame TIF |
| `intra_ccd_mosaic` | Merge `_0/_1` into `RED{j}.tif` |
| `downsample` | Pyramid levels L1–L4 |
| `feature_mosaic` | Build/update `mosaic.txt` and mosaic params |
| `feature_extract` | Features per pyramid level |
| `feature_match` | Pyramid fenfu match + optional densify + Fea IR |
| `grid_match` | Semi-dense grid matching |
| `inter_ccd_match` | Intra- / cross-CCD ties |
| `ba` | Bundle adjustment and forward-intersection DEM |

Disabled steps print `STEP_SKIP`. Per-run logs go under `logs/p/<timestamp>/`.

## Configuration

Primary config: `cfg.yaml` (or `cfg_<ID1>_<ID2>.yaml`). Main groups:

- **`dataset`**: `srcfilepath` (PDS), `filepath` (working tree), stereo IDs, `rows` / `cols` / `CCD_*`
- **`steps`**: the nine switches above
- **`mosaic`**: overlap samples, etc.
- **`feature_extract` / `feature_match`**: windows, search, CC, RANSAC, local-affine keep, densify, iterative refinement
- **`grid_match`**: grid window/search/thresholds, affine gates, `affine_pred_as_match`, etc.
- **`inter_ccd_match`**: intra/cross params, `control_source` (`feature` \| `grid`)
- **`ba`**: Tichu outlier gates, `fi_source`, Ceres iterations and Cauchy losses

See `script/gen_cfg.py` and `script/gen_pair_configs.sh` to generate pair configs.

## Data layout

**PDS source (`dataset.srcfilepath`)**

```text
{src}/{ID}/{ID}_RED{ccd}_{0|1}.img
```

**Working tree (`dataset.filepath`)**

```text
{filepath}/{ID}/downsample/{0..4}/
  {ID}_RED{j}.tif
  {ID}_RED{j}.txt / _match.txt / _grid.txt
  mosaic.txt / mosaic*.tif
```

**Repo `data/` (from `bin/`, typically `../data/`)**

| Path | Contents |
|------|----------|
| `EO/` | Exterior orientation `{ID}.txt`, `{ID}/`, often SPICE poses |
| `IO/` | Interior orientation `IO.txt` |
| `observedata/` | Observations such as `{ID1}_{ID2}.txt`, `_LCfea` / `_LCgrid` |
| `result/` | BA outputs `{ID}_EOre.txt`, FI DEM `{ID1}_{ID2}_FI.txt`, etc. |

## Useful scripts

| Script | Purpose |
|--------|---------|
| `script/run_mars_batch.sh` | Batch-run `bin/Mars` |
| `script/gen_cfg.py` / `gen_pair_configs.sh` | Generate per-pair cfg |
| `script/gen_eo_inputfile.py` | Write EO inputs |
| `script/update_all_result.sh` | Collect match and BA products |
| `script/grid_to_mosaic_disparity*.py` | Mosaic-frame disparity / paper panels (incl. `*_points.txt`) |
| `script/update_all_grid_to_mosaic_disparity.sh` | Batch disparity figures |
| `script/draw_fenfu_match0_inliers.py` | Fenfu correct / incorrect / unmatched viz |
| `script/denoise_fi_dem_pointcloud.py` | Denoise FI point clouds |
| `script/mro_spice/` | HiRISE SPICE / EO helpers |
| `script/matlab/` | DTM evaluation, rendering, coordinate transforms, etc. |

## Fenfu visualization

`out/fenfu_match0.tif` (layer 0) is colored from files:

- **Green**: matches kept in `*_match.txt`
- **Red**: `bj==1` in `*.txt` but not present in `*_match.txt`
- **Yellow**: unmatched features (`bj==0`)

With `use_local_affine: true`, keep uses local-affine residual (+ optional score rescue). With `false`, keep uses global affine thresholds `global_keep_radius` / `global_keep_abs`.

## License

No license file is included in this repository. Use and redistribution must comply with the licenses of dependencies (OpenCV, Ceres, CSPICE, etc.).
