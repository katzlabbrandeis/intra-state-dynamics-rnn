"""
# Non-stationarity of latents
- Check for:
    - Alignment with changepoints
    - Eigenspectrum of LDS fit to latents
    - High correlation with binned spike-counts
"""

import os
import sys
from pprint import pprint as pp
from matplotlib import pyplot as plt
import numpy as np
from itertools import product
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from glob import glob
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA

base_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn'
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

pop_analysis_src_dir = os.path.join(src_dir, 'abu', 'model_fitting','population_analysis')
sys.path.append(pop_analysis_src_dir)
import utils

rel_data_path = 'output/intermediate_data/RNN_PROCESSING_PARQUETS/latent_outputs/raw_output_unwarped'
abs_data_path = f"{base_dir}/{rel_data_path}"

artifacts_dir = f"{base_dir}/output/artifacts/population_analysis"
array_artifacts_dir = f"{artifacts_dir}/latent_arrays"
dfs_artifacts_dir = f"{artifacts_dir}/latent_dfs"
fr_artifacts_dir = f"{artifacts_dir}/firing_rate_arrays"

plot_dir = '/media/bigdata/firing_space_plot/intra-state-dynamics-rnn/output/plots'
pop_analysis_plot_dir = os.path.join(plot_dir, 'population_analysis')

############################################################
# latent_df.to_pickle(os.path.join(artifacts_dir, 'all_latent_df.pkl'))
latent_df = pd.read_pickle(os.path.join(artifacts_dir, 'all_latent_df.pkl'))
