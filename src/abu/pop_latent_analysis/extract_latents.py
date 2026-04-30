
import os
import sys
from pprint import pprint as pp
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
# from tmux import tmux
from itertools import product

base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

from core.utils.read_parquets import read_parquet_files_into_dict
# /media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/july_25_rnn_with_fr/pred_latent
# rel_data_path = 'output/july_25_rnn_with_fr/pred_latent'
# /output/JULY_RNN_RUN
# abs_data_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/oct_rnn_thesis_rnn/output_taste/AM11_4Tastes_191030_114043_repacked/artifacts'

# abs_data_path = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/RNN_PROCESSING_PARQUETS/latent_outputs/raw_output_unwarped'
# rel_data_path = 'output/oct_rnn_thesis_rnn/pred_latent'
rel_data_path = 'output/intermediate_data/RNN_PROCESSING_PARQUETS/latent_outputs/raw_output_unwarped'
abs_data_path = f"{base_dir}/{rel_data_path}"

artifacts_dir = f"{base_dir}/output/artifacts/population_analysis"
array_artifacts_dir = f"{artifacts_dir}/latent_arrays"
if not os.path.exists(array_artifacts_dir):
    os.makedirs(array_artifacts_dir)
dfs_artifacts_dir = f"{artifacts_dir}/latent_dfs"
if not os.path.exists(dfs_artifacts_dir):
    os.makedirs(dfs_artifacts_dir)

# make sure path exists
if not os.path.exists(abs_data_path):
    raise FileNotFoundError(f"Data path does not exist: {abs_data_path}")

plot_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/plots/population_analysis/latents_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

##############################

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
# session_key = 'AM11_4Tastes_191030_114043_repacked_raw_latent_vectors'
# session_key = 'AM11_4Tastes_191030_114043_rnn_latent_raw_output_unwarped'

for session_key in latents.keys():
    session_latents = latents[session_key]

    session_latents_df = session_latents.to_pandas()
    # Convert melt all columns with latent_dim
    latent_dim_cols = [col for col in session_latents_df.columns if col.startswith('latent_dim')]
    latent_dim_index = [int(col.split('_')[-1]) for col in latent_dim_cols]
    dim_name_to_ind_map = {col: int(col.split('_')[-1]) for col in latent_dim_cols}
    session_latents_melted = session_latents_df.melt(
        id_vars=['taste', 'trial', 'time'], 
        value_vars=latent_dim_cols, 
        var_name='latent_dim', 
        value_name='latent_value'
    )
    # Map latent_dim names to indices
    session_latents_melted['latent_dim'] = session_latents_melted['latent_dim'].map(dim_name_to_ind_map)

    # Set multi-index to easily convert to xarray
    session_latents_melted.set_index(['taste', 'trial', 'latent_dim', 'time'], inplace=True)

    # Write out melted dataframe for this session for easier debugging later on
    session_latents_melted.to_csv(f"{dfs_artifacts_dir}/{session_key}_latents_melted.csv")

    # Convert to xarray
    session_latents_xr = xr.Dataset.from_dataframe(session_latents_melted)

    # Convert to numpy array
    # Shape: (4, 30, 8, 119)
    session_latents_np = np.squeeze(session_latents_xr.to_array().values)

    # Write out numpy array for this session
    np.save(f"{array_artifacts_dir}/{session_key}_latents.npy", session_latents_np)

    # # Plot all latents with taste x trial stacked
    # stacked_latents = np.concatenate(session_latents_np, axis=0)  # Shape: (4*30, 8, 119)
    # stacked_latents = np.swapaxes(stacked_latents, 0,1)  # Shape: (8, 4*30, 119) 


    # Models were fit independently to each taste, so plot latents x taste grid
    n_latent_dims = session_latents_np.shape[2]
    n_tastes = session_latents_np.shape[0]
    inds = list(product(range(n_latent_dims), range(n_tastes)))

    fig , ax = plt.subplots(n_latent_dims, n_tastes, figsize=(4*n_tastes, 3*n_latent_dims), sharex=True, sharey=True)
    for this_dim, this_taste in inds:
        ax[this_dim, this_taste].imshow(
                session_latents_np[this_taste, :, this_dim], 
                aspect='auto', origin='lower', interpolation='none', cmap='jet')
        ax[this_dim, this_taste].set_title(f'Latent Dimension {this_dim} - Taste {this_taste}')
    plt.xlabel('Time')
    plt.suptitle(f'Session: {session_key}\nAll Latent Dimensions (Taste x Trial stacked)')
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/{session_key}_all_latent_dims_stacked.png")
    plt.close(fig)

# Also write out a note to artifacts dir about how latents for each taste were fit independently, so the latent dimensions are not directly comparable across tastes. This is important to remember when analyzing the latents later on.
with open(f"{array_artifacts_dir}/README.txt", 'w') as f:
    f.write("Note: Latent dimensions for each taste were fit independently, so the latent dimensions are not directly comparable across tastes. This is important to remember when analyzing the latents later on.")

# Write same note to dfs artifacts dir
with open(f"{dfs_artifacts_dir}/README.txt", 'w') as f:
    f.write("Note: Latent dimensions for each taste were fit independently, so the latent dimensions are not directly comparable across tastes. This is important to remember when analyzing the latents later on.")

