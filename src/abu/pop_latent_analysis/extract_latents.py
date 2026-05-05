
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

############################################################
# Also extract firing rates for each session in the same format as latents for easier comparison later on. This will be used to compare the quality of the latents to the original firing rates.
############################################################

fr_pq_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/intermediate_data/pred_fr_clean'

rnn_firing_rates = read_parquet_files_into_dict(
    fr_pq_dir,
)

# Strucutre: dict
# Each entry looks like this:
# 'AM35_4Tastes_201231_105700_repacked_repacked_raw_predicted_firing': shape: (14_280, 31)
# ┌──────────┬──────────┬───────────┬──────────┬───┬───────────┬───────┬───────┬──────┐
# │ neuron_0 ┆ neuron_1 ┆ neuron_2  ┆ neuron_3 ┆ … ┆ neuron_27 ┆ taste ┆ trial ┆ time │
# │ ---      ┆ ---      ┆ ---       ┆ ---      ┆   ┆ ---       ┆ ---   ┆ ---   ┆ ---  │
# │ f64      ┆ f64      ┆ f64       ┆ f64      ┆   ┆ f64       ┆ i64   ┆ i64   ┆ i64  │
# ╞══════════╪══════════╪═══════════╪══════════╪═══╪═══════════╪═══════╪═══════╪══════╡
# │ 4.589677 ┆ 1.721299 ┆ 12.442411 ┆ 1.896385 ┆ … ┆ 0.097232  ┆ 0     ┆ 0     ┆ 0    │
# │ 5.599153 ┆ 1.538928 ┆ 12.72961  ┆ 2.255279 ┆ … ┆ -0.146568 ┆ 0     ┆ 1     ┆ 0    │
# │ 5.575372 ┆ 0.737943 ┆ 11.352234 ┆ 2.307653 ┆ … ┆ 0.310867  ┆ 0     ┆ 2     ┆ 0    │
# │ 5.947958 ┆ 0.666351 ┆ 10.949901 ┆ 2.331769 ┆ … ┆ 0.240528  ┆ 0     ┆ 3     ┆ 0    │
# │ 6.898021 ┆ 0.547621 ┆ 11.052068 ┆ 2.45869  ┆ … ┆ 0.410288  ┆ 0     ┆ 4     ┆ 0    │
# │ …        ┆ …        ┆ …         ┆ …        ┆ … ┆ …         ┆ …     ┆ …     ┆ …    │
# │ 3.56381  ┆ 2.098343 ┆ 11.858298 ┆ 1.366497 ┆ … ┆ 0.273838  ┆ 3     ┆ 114   ┆ 29   │
# │ 4.948151 ┆ 1.048931 ┆ 10.843879 ┆ 1.77379  ┆ … ┆ 0.379416  ┆ 3     ┆ 115   ┆ 29   │
# │ 4.823439 ┆ 0.660264 ┆ 9.857202  ┆ 2.011089 ┆ … ┆ 0.67263   ┆ 3     ┆ 116   ┆ 29   │
# │ 4.415578 ┆ 0.765167 ┆ 9.556287  ┆ 1.938127 ┆ … ┆ 0.779232  ┆ 3     ┆ 117   ┆ 29   │
# │ 3.485828 ┆ 2.159206 ┆ 10.837589 ┆ 1.896595 ┆ … ┆ 0.756261  ┆ 3     ┆ 118   ┆ 29   │
# └──────────┴──────────┴───────────┴──────────┴───┴───────────┴───────┴───────┴──────┘}
#

# Convert to numpy arrays with shape: (taste, trial, neuron, time) for easier comparison to latents later on
firing_rate_arrays_dir = os.path.join(artifacts_dir, 'firing_rate_arrays')
os.makedirs(firing_rate_arrays_dir, exist_ok=True)

for session_key in rnn_firing_rates.keys():
    session_fr = rnn_firing_rates[session_key]

    session_fr_df = session_fr.to_pandas()

    # Set multi-index to easily convert to xarray
    session_fr_df.set_index(['taste', 'trial', 'time'], inplace=True)

    # Convert to xarray
    session_fr_xr = xr.Dataset.from_dataframe(session_fr_df)

    # Convert to numpy array
    # Shape: (neurons, taste, time, trial) -> need to reorder to (taste, trial, neuron, time) to match latents
    session_fr_np = np.squeeze(session_fr_xr.to_array().values)
    session_fr_np = np.transpose(session_fr_np, (1, 3, 0, 2))  # Reorder to (taste, trial, neuron, time) 

    # Write out numpy array for this session
    np.save(f"{firing_rate_arrays_dir}/{session_key}_fr.npy", session_fr_np)

# write out a note about how firing rate inference for each taste was performed independently, so the firing rates are not directly comparable across tastes. This is important to remember when analyzing the firing rates later on. 
with open(f"{firing_rate_arrays_dir}/README.txt", 'w') as f:
    f.write("Note: Firing rate inference for each taste was performed independently, so the firing rates are not directly comparable across tastes. This is important to remember when analyzing the firing rates later on.")

