
import os
import sys
from pprint import pprint as pp
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

