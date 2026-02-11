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

base_dir = '/media/bigdata/firing_space_plot/NBT_EMB_Classifier_Analyses'
artifacts_dir = os.path.join(base_dir, 'artifacts')
artifacts_subdir = os.path.join(artifacts_dir, 'special_ephys_models')
change_out_dir = os.path.join(artifacts_subdir, 'models')

# plot_dir = os.path.join(artifacts_subdir, 'plots')
# plot_sup_dir = '/media/bigdata/firing_space_plot/NBT_EMB_Classifier_Analyses/plots/mtm_hmm_plots/
plot_sup_dir = os.path.join(base_dir, 'plots', 'mtm_hmm_plots')
plot_dir = os.path.join(plot_sup_dir, 'ephys_models_special')
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
    hue='dat_type',
    col='taste_name',
    row='basename',
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
    return group
best_models_df = model_info_df.groupby(
        ['basename', 'taste_name', 'dat_type']
            ).apply(lowest_in_selected_states).reset_index(drop=True)

# Save best model info dataframe
best_models_df_fp = os.path.join(
    artifacts_subdir, 'best_model_info_df.pkl'
    )
with open(best_models_df_fp, 'wb') as f:
    dump(best_models_df, f)

# dat_type options: ['stable_only', 'stable_responsive_dynamic_discriminative']
# Keep only stable_responsive_dynamic_discriminative 
best_models_df = best_models_df[
    best_models_df['dat_type'] == 'stable_responsive_dynamic_discriminative'
    ].reset_index(drop=True)

# Perform pairwise comparisons between n_states for elbo_zscore
# pairwise_results = pg.pairwise_tukey(
#     data=best_models_df,
#     dv='elbo_zscore',
#     between='n_states',
#     )

from itertools import combinations
unique_n_states = best_models_df['n_states'].unique()
state_comparisons = list(combinations(unique_n_states, 2))
pairwise_results_list = []
for state_pair in state_comparisons:
    # Only continue if 3.0 in state_pair
    if 3 not in [int(s) for s in state_pair]: 
        continue
    state1 = state_pair[0]
    state2 = state_pair[1]
    data1 = best_models_df[best_models_df['n_states'] == state1]['elbo_zscore']
    data2 = best_models_df[best_models_df['n_states'] == state2]['elbo_zscore']
    # ttest_res = pg.ttest(data1, data2, paired=True)
    ttest_res = stats.ttest_rel(data1, data2)
    out_dict = {
        'state1': state1,
        'state2': state2,
        't_stat': ttest_res.statistic,
        'p_value': ttest_res.pvalue,
        'mean_state1': data1.mean(),
        'mean_state2': data2.mean(),
        }
    pairwise_results_list.append(out_dict)

pairwise_results_df = pd.DataFrame(pairwise_results_list)
# Save pairwise results
pairwise_results_fp = os.path.join(
    artifacts_subdir, 'best_model_elbo_zscore_pairwise_comparisons.csv'
    )
pairwise_results_df.to_csv(pairwise_results_fp, index=False)


# test_group = list(best_models_df.groupby(
#     ['basename', 'taste_name', 'dat_type']
#         ))[0]
# lowest_in_selected_states(test_group[1])

# Make same plots for best models only
g = sns.relplot(
    data=best_models_df,
    x='n_states',
    y='elbo_zscore',
    hue='taste_name',
    col='basename',
    kind='line',
    marker='o',
)
g.fig.suptitle('Best Model ELBO Z-Scores by Number of States', y=1.02)
g.fig.savefig(os.path.join(plot_dir, 'best_model_elbo_zscores.png'))
plt.close(g.fig)
# g = sns.relplot(
#     data=best_models_df,
#     x='n_states',
#     y='elbo_zscore',
#     hue='dat_type',
#     col='taste_name',
#     row='basename',
#     kind='line',
#     marker='o',
# )
# g.fig.suptitle('Best Model ELBO Z-Scores by Number of States', y=1.02)
# g.fig.savefig(os.path.join(plot_dir, 'best_model_elbo_zscores.png'))
# plt.close(g.fig)

# Aggregate plot across basenames and taste names
# g = sns.catplot(
#     data=best_models_df,
#     x='n_states',
#     y='elbo_zscore',
#     hue='dat_type',
#     kind='box',
#     fill=False,
#     color='black'
# )
# box color = 'black'
fig,ax = plt.subplots(figsize=(3,5))
c = 'black'
g = sns.boxplot(
    data=best_models_df,
    x='n_states',
    y='elbo_zscore',
    color='white',
    linewidth=2,
    boxprops=dict(edgecolor='k'),
    medianprops=dict(color='r', lw=0),
    capprops=dict(color='k', lw=2),
    whiskerprops=dict(color='k', lw=2),
    # Don't plot median
    ax=ax
)
mean_zscored_elbo = best_models_df.groupby('n_states')['elbo_zscore'].mean().reset_index()
ax.plot(
    mean_zscored_elbo['n_states']-2, 
    mean_zscored_elbo['elbo_zscore'], 
    '-o',
    color='r', 
    lw=2, 
    label='Mean Zscored ELBO'
)
# fig.suptitle('Best Model ELBO Z-Scores by Number of States (Aggregated)', y=1.02)
fig.savefig(os.path.join(plot_dir, 'best_model_elbo_zscores_aggregated_ephys.svg'),
            bbox_inches='tight')
plt.close(fig)

##############################
# Compare mode-tau for all models and best models for n_states = 3
all_models_n3 = model_info_df[model_info_df['n_states'] == 3]
best_models_n3 = best_models_df[best_models_df['n_states'] == 3]

# Shape of each array = (n_trials, n_transitions)
all_model_n3_taus = [
        model_dicts[fp][5] for fp in all_models_n3['save_path']
        ]

all_models_n3['mode_tau'] = all_model_n3_taus
# # Add inds so mode_tau can be exploded
# all_model_n3_exploded_list = []
# for i, row in all_models_n3.iterrows():
#     mode_tau = row['mode_tau']
#     mode_tau_inds = np.array(list(np.ndindex(mode_tau.shape)))

def add_explode_inds(row):
    mode_tau = row['mode_tau']
    mode_tau_inds = np.array(list(np.ndindex(mode_tau.shape)))
    row['trial_num'] = mode_tau_inds[:, 0]
    row['transition_num'] = mode_tau_inds[:, 1]
    row['mode_tau'] = mode_tau.flatten()
    return row

all_models_n3_exploded = all_models_n3.apply(
    add_explode_inds, axis=1
    ).explode(['trial_num', 'transition_num', 'mode_tau']).reset_index(drop=True)

# Mark rows as best or not
all_models_n3_exploded['is_best_model'] = all_models_n3_exploded.apply(
    lambda row: row.save_path in best_models_n3['save_path'].values,
    axis=1
    )

# Drop dat_type stable_only
all_models_n3_exploded = all_models_n3_exploded[
    all_models_n3_exploded['dat_type'] != 'stable_only'
    ].reset_index(drop=True)

# Scale mode_tau to ms
all_models_n3_exploded['scaled_mode_tau'] = all_models_n3_exploded['mode_tau'] * 50  # 50ms bins 

# Write out dataframe
all_models_n3_exploded_fp = os.path.join(
    artifacts_subdir, 'all_models_n3_mode_tau_exploded.pkl'
    )
with open(all_models_n3_exploded_fp, 'wb') as f:
    dump(all_models_n3_exploded, f)

# Plot distribution of mode_tau for best vs all models
g = sns.catplot(
    data=all_models_n3_exploded,
    x='transition_num',
    y='scaled_mode_tau',
    hue='taste_name',
    kind='boxen',
    col='is_best_model',
)
g.fig.suptitle('Mode Tau Distribution for n_states=3 Models', y=1.02)
plt.tight_layout()
g.fig.savefig(os.path.join(plot_dir, 'mode_tau_distribution_n3_models.png'))
plt.close(g.fig)

# make a plot collapsed across taste_name
g = sns.catplot(
    data=all_models_n3_exploded,
    x='transition_num',
    y='scaled_mode_tau',
    hue='is_best_model',
    kind='boxen',
)
g.fig.suptitle('Mode Tau Distribution for n_states=3 Models (Collapsed)', y=1.02)
plt.tight_layout()
g.fig.savefig(os.path.join(plot_dir, 'mode_tau_distribution_n3_models_collapsed.png'))
plt.close(g.fig)
