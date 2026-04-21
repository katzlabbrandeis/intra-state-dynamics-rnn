
import os
import sys
from pprint import pprint as pp
from matplotlib import pyplot as plt
import xarray as xr

base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

from core.utils.read_parquets import read_parquet_files_into_dict
# /media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/july_25_rnn_with_fr/pred_latent
rel_data_path = 'output/july_25_rnn_with_fr/pred_latent'
abs_data_path = f"{base_dir}/{rel_data_path}"

# make sure path exists
if not os.path.exists(abs_data_path):
    raise FileNotFoundError(f"Data path does not exist: {abs_data_path}")

# Load previously saved PCA results

# Help on function read_parquet_files_into_dict in module core.utils.read_parquets:
#
# read_parquet_files_into_dict(parquet_path: str | Path) -> Dict[str, pl.DataFrame]
#     Reads all .parquet files in a directory into a dictionary.
#
#     Parameters
#     ----------
#     parquet_path : str or Path
#         Path to the folder containing parquet files.
#
#     Returns
#     -------
#     dict
#         Dictionary where keys are filenames (no extension) and values are Polars DataFrames.

latents = read_parquet_files_into_dict(
    abs_data_path,
)

# latents: structure

# >>> pp(list(latents.keys()))
# ['AM11_4Tastes_191030_114043_repacked_raw_latent_vectors',
#  'AM11_4Tastes_191031_083633_repacked_raw_latent_vectors',
#  'AM11_4Tastes_191101_084921_repacked_raw_latent_vectors',
#  'AM12_4Tastes_191105_083246_repacked_raw_latent_vectors',
#  'AM12_4Tastes_191106_085215_repacked_raw_latent_vectors',
#  'AM12_4Tastes_191107_084707_repacked_raw_latent_vectors',
#  'AM17_4Tastes_191125_084206_repacked_raw_latent_vectors',
#  'AM17_4Tastes_191126_084934_repacked_raw_latent_vectors',
#  'AM25_4Tastes_200806_094914_repacked_raw_latent_vectors',
#  'AM25_4Tastes_200807_092703_repacked_raw_latent_vectors',
#  'AM25_4Tastes_200808_095124_repacked_raw_latent_vectors',
#  'AM26_4Tastes_200826_101430_repacked_raw_latent_vectors',
#  'AM26_4Tastes_200827_094829_repacked_raw_latent_vectors',
#  'AM26_4Tastes_200829_100535_repacked_raw_latent_vectors',
#  'AM35_4Tastes_201228_124547_repacked_raw_latent_vectors',
#  'AM35_4Tastes_201229_150307_repacked_raw_latent_vectors',
#  'AM35_4Tastes_201230_115322_repacked_raw_latent_vectors',
#  'AM35_4Tastes_201231_105700_repacked_repacked_raw_latent_vectors']


# >>> latents['AM11_4Tastes_191030_114043_repacked_raw_latent_vectors']
# shape: (14_280, 11)
# ┌──────────────┬──────────────┬──────────────┬──────────────┬───┬──────────────┬───────┬───────┬──────┐
# │ latent_dim_0 ┆ latent_dim_1 ┆ latent_dim_2 ┆ latent_dim_3 ┆ … ┆ latent_dim_7 ┆ taste ┆ trial ┆ time │
# │ ---          ┆ ---          ┆ ---          ┆ ---          ┆   ┆ ---          ┆ ---   ┆ ---   ┆ ---  │
# │ f32          ┆ f32          ┆ f32          ┆ f32          ┆   ┆ f32          ┆ i64   ┆ i64   ┆ i64  │
# ╞══════════════╪══════════════╪══════════════╪══════════════╪═══╪══════════════╪═══════╪═══════╪══════╡
# │ 0.966254     ┆ 0.014346     ┆ -0.218617    ┆ -0.421086    ┆ … ┆ 0.308383     ┆ 0     ┆ 0     ┆ 0    │
# │ 0.728934     ┆ 0.025451     ┆ -0.203998    ┆ -0.682993    ┆ … ┆ 0.175007     ┆ 0     ┆ 0     ┆ 1    │
# │ 0.923097     ┆ 0.31882      ┆ -0.529579    ┆ -0.23832     ┆ … ┆ -0.125736    ┆ 0     ┆ 0     ┆ 2    │
# │ 0.817191     ┆ 0.184952     ┆ -0.374975    ┆ -0.608398    ┆ … ┆ 0.143324     ┆ 0     ┆ 0     ┆ 3    │
# │ 0.663023     ┆ -0.049147    ┆ -0.106354    ┆ -0.765761    ┆ … ┆ 0.267793     ┆ 0     ┆ 0     ┆ 4    │
# │ …            ┆ …            ┆ …            ┆ …            ┆ … ┆ …            ┆ …     ┆ …     ┆ …    │
# │ 0.04207      ┆ -0.295765    ┆ 0.847281     ┆ -0.027463    ┆ … ┆ 0.959085     ┆ 3     ┆ 118   ┆ 25   │
# │ -0.336883    ┆ -0.720598    ┆ 0.959569     ┆ -0.238973    ┆ … ┆ 0.806027     ┆ 3     ┆ 118   ┆ 26   │
# │ -0.642317    ┆ -0.750802    ┆ 0.922148     ┆ 0.484366     ┆ … ┆ 0.997493     ┆ 3     ┆ 118   ┆ 27   │
# │ -0.269516    ┆ -0.332779    ┆ 0.966662     ┆ 0.499759     ┆ … ┆ 0.997414     ┆ 3     ┆ 118   ┆ 28   │
# │ -0.05679     ┆ -0.307341    ┆ 0.914504     ┆ -0.134278    ┆ … ┆ 0.975255     ┆ 3     ┆ 118   ┆ 29   │
# └──────────────┴──────────────┴──────────────┴──────────────┴───┴──────────────┴───────┴───────┴──────┘

# >>> pp(latents['AM11_4Tastes_191030_114043_repacked_raw_latent_vectors'].columns)
# ['latent_dim_0',
#  'latent_dim_1',
#  'latent_dim_2',
#  'latent_dim_3',
#  'latent_dim_4',
#  'latent_dim_5',
#  'latent_dim_6',
#  'latent_dim_7',
#  'taste',
#  'trial',
#  'time']

##############################
# Testing
##############################

# Load latents for one session and plot n random trials 
session_key = 'AM11_4Tastes_191030_114043_repacked_raw_latent_vectors'
session_latents = latents[session_key]

# Convert first to xarray, then to numpy for plotting
session_latents_xr = xr.Dataset.from_dataframe(session_latents.to_pandas())
# Use taste, trial, and time as coordinates
session_latents_xr = session_latents_xr.set_coords(['taste', 'trial', 'time'])

# >>> session_latents_xr
# <xarray.Dataset> Size: 914kB
# Dimensions:       (index: 14280)
# Coordinates:
#   * index         (index) int64 114kB 0 1 2 3 4 ... 14276 14277 14278 14279
#     taste         (index) int64 114kB 0 0 0 0 0 0 0 0 0 0 ... 3 3 3 3 3 3 3 3 3
#     trial         (index) int64 114kB 0 0 0 0 0 0 0 ... 118 118 118 118 118 118
#     time          (index) int64 114kB 0 1 2 3 4 5 6 7 ... 23 24 25 26 27 28 29
# Data variables:
#     latent_dim_0  (index) float32 57kB 0.9663 0.7289 0.9231 ... -0.2695 -0.05679
#     latent_dim_1  (index) float32 57kB 0.01435 0.02545 ... -0.3328 -0.3073
#     latent_dim_2  (index) float32 57kB -0.2186 -0.204 -0.5296 ... 0.9667 0.9145
#     latent_dim_3  (index) float32 57kB -0.4211 -0.683 -0.2383 ... 0.4998 -0.1343
#     latent_dim_4  (index) float32 57kB -0.2854 0.01026 ... -0.6104 -0.4006
#     latent_dim_5  (index) float32 57kB -0.5398 -0.5113 -0.3503 ... 0.2839 0.2297
#     latent_dim_6  (index) float32 57kB -0.5239 0.1935 0.8877 ... -0.9114 -0.5564
#     latent_dim_7  (index) float32 57kB 0.3084 0.175 -0.1257 ... 0.9974 0.9753
# >>> 

# Concatenate all latent dimensions into a single DataArray with a new dimension 'latent_dim' 
latent_dims = [x for x in session_latents_xr.data_vars if x.startswith('latent_dim')]
latent_dim_index = [int(x.split('_')[-1]) for x in latent_dims]
session_latents_da = xr.concat([session_latents_xr[dim] for dim in latent_dims], 
                                dim='latent_dim')
session_latents_da = session_latents_da.assign_coords(latent_dim=latent_dim_index)

# Reshape to (taste, trial, latent_dim, time) by setting multi-index and unstacking
session_latents_da = session_latents_da.set_index(index=['taste', 'trial', 'time'])
session_latents_da = session_latents_da.unstack('index')

# Transpose to get dimensions in order: (taste, trial, latent_dim, time)
session_latents_array = session_latents_da.transpose('taste', 'trial', 'latent_dim', 'time')
