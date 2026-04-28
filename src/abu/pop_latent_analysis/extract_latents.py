
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
# rel_data_path = 'output/july_25_rnn_with_fr/pred_latent'
rel_data_path = 'output/oct_rnn_thesis_rnn/pred_latent'
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

abs_data_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/oct_rnn_thesis_rnn/output_taste/AM11_4Tastes_191030_114043_repacked/artifacts'

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

# Check how many trials per taste for each session
for session_key, session_latents in latents.items():
    session_latents_df = session_latents.to_pandas()
    taste_trial_counts = session_latents_df.groupby('taste')['trial'].nunique()
    print(f"Session: {session_key}")
    print(taste_trial_counts)
    print("-" * 40)

# Load latents for one session and plot n random trials 
session_key = 'AM11_4Tastes_191030_114043_repacked_raw_latent_vectors'
session_latents = latents[session_key]

session_latents_df = session_latents.to_pandas()

# Get taste 0
taste_0_latents = session_latents_df[session_latents_df['taste'] == 0]

fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,6))
ax[0].imshow(taste_0_latents[taste_0_latents.columns[:8]].values, aspect='auto', interpolation='none')
# Also plot index on adjacent subplot to verify trial structure
im1 = ax[1].imshow(taste_0_latents['trial'].values[:,None], aspect='auto', interpolation='none')
im2 = ax[2].imshow(taste_0_latents['time'].values[:,None], aspect='auto', interpolation='none')
ax[1].set_xlabel('Trials')
ax[0].set_title('Latent Dimensions')
ax[1].set_title('Trial Index')
ax[2].set_title('Time Index')
# Add colorbars for time and trial
fig.colorbar(im1, ax=ax[1], orientation='vertical', label='Trial Index')
fig.colorbar(im2, ax=ax[2], orientation='vertical', label='Time Index')
plt.show()

# Plot all tastes to see any duplication
fig, ax = plt.subplots(1,4, sharey=True, figsize=(12,6))
ax[0].imshow(session_latents_df[session_latents_df.columns[:8]].values, aspect='auto', interpolation='none')
ax[0].set_title('Latent Dimensions')
# Also plot index on adjacent subplot to verify trial structure
im1 = ax[1].imshow(session_latents_df['taste'].values[:,None], aspect='auto', interpolation='none')
im2 = ax[2].imshow(session_latents_df['trial'].values[:,None], aspect='auto', interpolation='none')
im3 = ax[3].imshow(session_latents_df['time'].values[:,None], aspect='auto', interpolation='none')
ax[2].set_xlabel('Trials')
ax[2].set_title('Trial Index')
ax[3].set_title('Time Index')
# Add colorbars for time and trial
fig.colorbar(im1, ax=ax[1], orientation='vertical', label='Taste Index')
fig.colorbar(im2, ax=ax[2], orientation='vertical', label='Trial Index')
fig.colorbar(im3, ax=ax[3], orientation='vertical', label='Time Index')
plt.show()

# Trials are all concatenated, so we need to create a trial index that resets for each taste
# Get unique taste, trial pairs and createa a map
taste_trial_pairs = session_latents_df[['taste', 'trial']].drop_duplicates().reset_index(drop=True)
taste_trial_pairs['trial_within_taste'] = taste_trial_pairs.groupby('taste').cumcount()

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

# >>> session_latents_da
# <xarray.DataArray 'latent_dim_0' (latent_dim: 8, index: 14280)> Size: 457kB
# array([[ 0.96625423,  0.7289341 ,  0.92309684, ..., -0.6423173 ,
#         -0.2695159 , -0.05679017],
#        [ 0.01434567,  0.02545078,  0.31882048, ..., -0.7508022 ,
#         -0.3327791 , -0.30734122],
#        [-0.21861674, -0.20399822, -0.52957875, ...,  0.92214787,
#          0.9666619 ,  0.91450393],
#        ...,
#        [-0.53976333, -0.511271  , -0.35027114, ...,  0.21953066,
#          0.2839369 ,  0.22973868],
#        [-0.5239218 ,  0.1935095 ,  0.8876943 , ..., -0.36798695,
#         -0.91144234, -0.55636597],
#        [ 0.30838275,  0.17500716, -0.12573628, ...,  0.99749285,
#          0.9974137 ,  0.97525454]], shape=(8, 14280), dtype=float32)
# Coordinates:
#   * latent_dim          (latent_dim) int64 64B 0 1 2 3 4 5 6 7
#   * index               (index) object 114kB MultiIndex
#     trial               (index) int64 114kB 0 0 0 0 0 0 ... 118 118 118 118 118
#   * taste               (index) int64 114kB 0 0 0 0 0 0 0 0 ... 3 3 3 3 3 3 3 3
#   * trial_within_taste  (index) int64 114kB 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0
#   * time                (index) int64 114kB 0 1 2 3 4 5 6 ... 24 25 26 27 28 29


# Create per-taste trial index by grouping by taste and numbering within each group
taste_coords = session_latents_da.coords['taste'].values
trial_coords = session_latents_da.coords['trial'].values
time_coords = session_latents_da.coords['time'].values

# Create a per-taste trial index
import pandas as pd
df_coords = pd.DataFrame({'taste': taste_coords, 'trial': trial_coords, 'time': time_coords})
df_coords['trial_within_taste'] = df_coords.groupby('taste').cumcount() // len(session_latents_da.coords['time'])

# Add the new coordinate to the DataArray
session_latents_da = session_latents_da.assign_coords(trial_within_taste=('index', df_coords['trial_within_taste'].values))

# Reshape to (taste, trial_within_taste, latent_dim, time) by setting multi-index and unstacking
session_latents_da = session_latents_da.set_index(index=['taste', 'trial_within_taste', 'time'])
session_latents_da = session_latents_da.unstack('index')

# Transpose to get dimensions in order: (taste, trial_within_taste, latent_dim, time)
session_latents_array = session_latents_da.transpose('taste', 'trial_within_taste', 'latent_dim', 'time')

# Now each taste should have exactly 30 trials
# >>> session_latents_array.shape
# (4, 30, 8, 30)

