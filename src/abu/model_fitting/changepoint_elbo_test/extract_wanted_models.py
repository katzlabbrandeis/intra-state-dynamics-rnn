base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)
# from pytau.changepoint_io import FitHandler
import pylab as plt
# from pytau.utils import ephys_data
from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data import visualize as vz
from tqdm import tqdm
import pytau.changepoint_model as models
from pprint import pprint as pp
import pandas as pd
from ast import literal_eval
import numpy as np
import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
from cloudpickle import load, dump
from scipy import stats
import seaborn as sns
import os
import pingouin as pg

##############################
base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
output_dir = os.path.join(base_dir, 'output')
artifacts_dir = os.path.join(output_dir, 'artifacts')

artifacts_subdir = os.path.join(artifacts_dir, 'ephys_changepoint_models')
change_out_dir = os.path.join(artifacts_subdir, 'models')

plot_dir = os.path.join(output_dir, 'plots', 'changepoint_analysis')
os.makedirs(plot_dir, exist_ok=True)

##############################
# Get models list
model_files = sorted(os.listdir(change_out_dir))

# Each model contains 
# (model, approx, elbo, retry_count, train_dat, mode_tau, model_info_dict),

model_tuples = []
for mf in tqdm(model_files):
    model_fp = os.path.join(change_out_dir, mf)
    with open(model_fp, 'rb') as f:
        model_tuple = load(f)
        model_tuples.append(model_tuple)
print(f'Loaded {len(model_tuples)} models from {change_out_dir}')

model_dicts = dict(zip(model_files, model_tuples))

# Extract relevant info from each model
model_elbos = [x[2] for x in model_tuples]
model_info_dicts = [x[6] for x in model_tuples]

model_info_df = pd.DataFrame(model_info_dicts)
# Add elbo to dataframe
model_info_df['elbo'] = model_elbos

model_info_df['save_path'] = model_files

##############################
# Print out useful info
# n-unique basenames
n_unique_basenames = model_info_df['basename'].nunique()
print(f'Number of unique basenames: {n_unique_basenames}')

##############################

# For each basename, taste_name, and dat_type, calculate z-scored elbo + elbo rank
grouped = model_info_df.groupby(['basename', 'taste_name', 'dat_type'])
def zscore_elbo(group):
    group = group.copy()
    group['elbo_zscore'] = stats.zscore(group['elbo'])
    group['elbo_rank'] = group['elbo'].rank(ascending=False)
    return group
model_info_df = grouped.apply(zscore_elbo).reset_index(drop=True)

# Plot elbo
g = sns.relplot(
    data=model_info_df,
    x='n_states',
    y='elbo_zscore',
    hue='taste_name',
    col='basename',
    kind='line',
    marker='o',
)
g.fig.suptitle('Model ELBO Z-Scores by Number of States', y=1.02)
g.fig.savefig(os.path.join(plot_dir, 'model_elbo_zscores.png'))
plt.close(g.fig)

# Aggregate plot across basenames and taste names
g = sns.catplot(
    data=model_info_df,
    x='n_states',
    y='elbo_zscore',
    hue='dat_type',
    kind='box',
)
g.fig.suptitle('Model ELBO Z-Scores by Number of States (Aggregated)', y=1.02)
g.fig.savefig(os.path.join(plot_dir, 'model_elbo_zscores_aggregated.png'))
plt.close(g.fig)

# Save model info dataframe
model_info_df_fp = os.path.join(
    artifacts_subdir, 'all_model_info_df.pkl'
    )
with open(model_info_df_fp, 'wb') as f:
    dump(model_info_df, f)

##############################
# Only keep models with 3 or 4 as having lowest elbo 
def lowest_in_selected_states(group):
    group = group.copy()
    argmin_elbo = group['elbo'].idxmin()
    argmin_n_states = group.loc[argmin_elbo, 'n_states']
    if not argmin_n_states in [3, 4]:
        return pd.DataFrame()
    # return group
    # Return only the row with lowest elbo
    else:
        return group.loc[[argmin_elbo]]

best_models_df = model_info_df.groupby(
        ['basename', 'taste_name', 'dat_type']
            ).apply(lowest_in_selected_states).reset_index(drop=True)

# Save best model info dataframe
best_models_df_fp = os.path.join(
    artifacts_subdir, 'best_model_info_df.pkl'
    )
with open(best_models_df_fp, 'wb') as f:
    dump(best_models_df, f)

