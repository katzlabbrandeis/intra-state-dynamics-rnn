#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:07:31 2024

#########################

THIS SHIT IS COMMENTED BECAUSE IT'S THE ACTIVE CONSTRUCTION ZONE
GO AWAYYYY DON'T LOOK AT THIS!


#########################


At this point, this script is just kinda where I'm building all sorts of stuff...

This needs to be orchestration, ultimately-- thinking I'm gonna split into a few files.


When finished working on this, uncomment everything




@author: vincentcalia-bogan
"""
import os
import os.path

## NON-VINCENT MODULE IMPORTS ##
# 17 datasets 365 neurons total
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from calc_fr_class_war_unwar import CalcFRStates  # now deprecated
from find_extract_info import find_copy_h5info, modify_tastes, process_info_files  # pushed and done
from FRDataProcessingWrappter import (
    FRDataProcessor,  # now deprecated; a resource-intensive conversion that doesn't work anyways
)
from freuqency_analysis_suite_rnn_class import FrequencyAnalysisPipeline  # again now its own class
from FRPipeline import FRPipeline  # new and improved firing rate and spiking processor-- does warped and unwarped
from generate_parquet_sig_all import NeuronSignificanceTester  # really only do this on rr neurons right now
from generate_parquet_sig_all import (
    consolidate_all_neuron_data,
    consolidate_all_neuron_war_data,
    sig_neurons_hz,
    sig_neurons_ttest,
    sig_neurons_war_hz,
    sig_neurons_war_ttest,
)
from load_std_changepoints_from_pkl import load_standardized_changepoints  # helper func # done
from PLTPipeline import PlottingPipeline
from read_parquets import all_nrns_to_df, read_parquet_files_into_dict  # pushed and done
from RNN_lat_spike_train_corr import latent_spike_train_correlation
from serialize_overlap import create_overlap_dataframes, serialized_neuron_df  # deprecated? ish?
from sig_testing_class import SignificanceTester_test
from spike_raster_class_plot_april import SpikeRasterPlotter  # assumes the class is in this module

from core.io.import_paths import ensure_src_on_path
from core.pre_processing.RNNLatentprocessing import (
    RNNLatentProcessor,  # big class processing the RNN stuff itself, pushed and done
)
from core.utils.extract_npz import extract_from_npz  # done
from core.utils.spike_train_to_npz import extract_to_npz, find_h5_files  # done
from core.utils.unpkl_generator import extract_valid_changepoints, unpickle_changepoints  # done


def _add_src_to_path() -> str:
    here = Path.cwd()
    for base in (here, *here.parents):
        cand = base / "src"
        if (cand / "core").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    raise RuntimeError("Could not find <repo>/src; run this from somewhere inside the repo.")


_add_src_to_path()

# line needed to make the old stuff work with the new stuff for loading all this hooey
sys.path.append('/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/underlying functions')


# rnn latent processing from underlying_functions
# from RNNLatentprocessing import RNNLatentProcessor  # big class processing the RNN stuff itself


# project_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work"
# submodule_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/underlying functions"
# # Check if paths already exist in sys.path
# sys.path.append(submodule_dir)
# sys.path.append(project_dir)
# Importing Vinmodules

# firing rate class with related calls: deprecated as of 6/24

# new firing rate calculation as of 6/24:


# from interpolation_xr import interpolate_and_average_all_tastes, interpolate_firing_rates # this is actually a deprecated func now
# from calc_firing_rate_with_states import (
#     calc_fr_states,
# )  # a new class -- I do not know that this is actually strictly needed anymore; seems like it can be deprecated


# modules that are for sure used:
# from FRpipeline import FRPipeline
# from RNNLatentprocessing import RNNLatentProcessor

# refrence name replacements
taste_replacements = {
    "nacl": "NaCl",
    "suc": "Sucrose",
    "ca": "Citric Acid",
    "qhcl": "Quinine",
}
epoch_labels = ("Identification", "Palatability", "Decision", "2000 ms Post-Stimulus")
## NECESARY PARAMETERS ##
window_length = 250
step_size = 25
alpha = 0.05  # alpha for any null hypothesis tests
threshold_hz = 2.0  # threshold firing rate freuqency for various funcs
# this is a pretty janky work-around; fix this later plz

################################################## INIT INTERMEDIATE FILES ####
# is the below even nessecary anymore? honestly, real tough to say. hmmmm
# important stuff below
file_path = "/Volumes/T7 Shield/spikesorting"
spike_trains_path = "/spike_trains"
npz_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Spike train npz data"
pkl_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/pkl files/"
info_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Spike train info data"
for data in extract_from_npz(npz_path):
    if isinstance(data, tuple):
        spike_array, dataset_num, index, key = data
        print(f"Dataset number: {dataset_num}")
        dataset_tastes = process_info_files(info_path, dataset_num)
        modified_tastes = modify_tastes(dataset_tastes, taste_replacements)
        # Call unpickle_changepoints to process the .pkl files for the current dataset_num
        extracted_pkl = unpickle_changepoints(
            pkl_path, [(spike_array, dataset_num, index, key)]
        )
npz_files_exist = any(
    file.endswith(".npz") for file in os.listdir(npz_path)
)  # save time with a check
if npz_files_exist:
    print(
        ".npz files containing spike trains already exist in npz_path. Skipping extraction from h5 files."
    )
    # Perform certain actions when npz files already exist (pass certain functions)
else:
    print("No .npz files found in npz_path. Running functions to generate npz files.")
    # Run functions to generate npz files
    h5_files = find_h5_files(file_path)  # pulling h5 file paths
    save_data = extract_to_npz(
        h5_files, file_path, spike_trains_path, npz_path
    )  # saving npz files to save location
# checking for info files
info_files_exist = any(file.endswith(".info") for file in os.listdir(info_path))
if info_files_exist:
    print(".info files already exist in info_path; skipping re-copying them")
else:
    print("No .info files found in info_path. Extracting info_files")
    info_files = find_copy_h5info(file_path, info_path)
    print(f"Found and copied {len(info_files)} .info files.")


# a quick on-off switch for some deprecated / not used all the time funcs:
run = False  # if I want to run all this hooey, then set to true
for _ in range(run):
    print("running the old stuff, bub")
    sig_ttest_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet"
    sig_war_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet_warped"
    sig_hz_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet"
    all_nrns_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet"
    all_war_nrns_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet_warped"

    sig_ttest_parquet_path_w = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet_warped"
    all_nrns_parquet_path_w = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet_warped"

    # VARIOUS INTERMEDIATE FILES THAT AT SOME POINT HAVE
    # BEEN USED FOR A GRAPH-- DOES NOT NEED TO BE INIT EVERY TIME #####################

    # checking for sig_parquet files -- unwarped
    sig_parquet_exist = any(
        file.endswith(".parquet") for file in os.listdir(sig_ttest_parquet_path)
    )
    if sig_parquet_exist:
        print(".parquet files already exist in sig_parquet_path; skipping re-copying them")
    else:
        print("No .parquet files found in sig_parquet_path. Extracting info_files")
        sig_ttest = sig_neurons_ttest(
            npz_path, pkl_path, sig_ttest_parquet_path, alpha, window_length, step_size
        )
        sig_2hz = sig_neurons_hz(
            npz_path, pkl_path, sig_hz_parquet_path, threshold_hz, window_length, step_size
        )
        # print(f"Found and copied {len(sig_ttest)} .parquet files.")
        # print(f"Found and copied {len(sig_2hz)} .parquet files.")
    # checking for parquet file with all neuron data
    consolidated_parquet_exist = any(
        file.endswith(".parquet") for file in os.listdir(all_nrns_parquet_path)
    )
    if consolidated_parquet_exist:
        print(
            ".parquet files already exist in all_nrns_parquet_path; skipping re-copying them"
        )
    else:
        print("No .info files found in all_nrns_parquet_path. Extracting info_files")
        consolidated_data = consolidate_all_neuron_data(
            npz_path, pkl_path, all_nrns_parquet_path, window_length, step_size
        )
        # print(f"Found and copied {len(consolidated_data)} .parquet files.")
    # unwarped data
    # reading data back from parquets
    sig_nrns_dict = read_parquet_files_into_dict(sig_ttest_parquet_path)
    # dataframe of all neuron data
    all_data_df = all_nrns_to_df(all_nrns_parquet_path)
    serialized_nrns_df = serialized_neuron_df(
        npz_path, pkl_path
    )  # just counts serialized nrns
    sig_overlap = create_overlap_dataframes(
        serialized_nrns_df, sig_nrns_dict
    )  # overlap of nrns
    sig_parquet_exist_w = any(
        file.endswith(".parquet") for file in os.listdir(sig_ttest_parquet_path_w)
    )
    if sig_parquet_exist_w:
        print(".parquet files already exist in sig_parquet_path; skipping re-copying them")
    else:
        print("No .parquet files found in sig_parquet_path. Extracting info_files")
        sig_ttest_w = sig_neurons_war_ttest(
            npz_path, pkl_path, sig_ttest_parquet_path_w, alpha, window_length, step_size
        )
        sig_2hz_w = sig_neurons_war_hz(
            npz_path,
            pkl_path,
            sig_ttest_parquet_path_w,
            threshold_hz,
            window_length,
            step_size,
        )
        # print(f"Found and copied {len(sig_ttest)} .parquet files.")
        # print(f"Found and copied {len(sig_2hz)} .parquet files.")
    # checking for parquet file with all neuron data
    consolidated_parquet_exist_w = any(
        file.endswith(".parquet") for file in os.listdir(all_nrns_parquet_path_w)
    )
    if consolidated_parquet_exist_w:
        print(
            ".parquet files already exist in all_nrns_parquet_path; skipping re-copying them"
        )
    else:
        print("No .info files found in all_nrns_parquet_path. Extracting info_files")
        consolidated_data_w = consolidate_all_neuron_war_data(
            npz_path, pkl_path, all_nrns_parquet_path_w, window_length, step_size
        )
    ###############################################################################

# Init Firing rate processor:
# FRDataProcessor is fundamentally a conversion layer-- which I do not want to have long-term.
# this was easier than refactoring mountains of code, so we stick with it.

# further upgrades july 2025

## FILE PATHS (ish) ##

# setting a base fof all this
BASE_DIR = os.path.expanduser('/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work')
# rnn and data used for senior thesis:
THESIS_SUBDIR = 'oct_rnn_thesis_rnn'  # this will have to be retroactively filled in
oct_rnn_path = os.path.join(BASE_DIR, THESIS_SUBDIR)  # for everything that has been done up to july 14 2025


# Firing rate stays in it's own top-level spot for now
# Instantiate pipeline

# fr goes top-level?
FR_pipeline = FRPipeline(
    npz_path=npz_path,
    info_path=info_path,
    pkl_path=pkl_path,
    taste_replacements={"NaCl": "NaCl", "Sucrose": "Sucrose", "Citric Acid": "Citric Acid", "Quinine": "Quinine"},
    output_dir='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS',
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000,
    start_time=1500,
    end_time=4500,
    save_outputs=True
)
# fr_unwar_dict, spike_unwar_dict, fr_war_dict, spike_war_dict = FR_pipeline.full_pipeline()
FR_parquets_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/rr_firing_unwarped'
FR_parquets = os.path.isdir(FR_parquets_dir) and any(
    file.endswith(".parquet") for file in os.listdir(FR_parquets_dir)
    # checking one of these dirs -- as if they're missing from one, then they're more than likely missing from all the rest.
    # now also checks if the dir even exits
)
if FR_parquets:
    print('.parquet files containing firing and spiking already exist; skipping re-calculation. Extracting...')
    fr_unwar_dict = read_parquet_files_into_dict(
        '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/rr_firing_unwarped')
    fr_war_dict = read_parquet_files_into_dict(
        '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/rr_firing_warped')
    spike_war_dict = read_parquet_files_into_dict(
        '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/spikes_warped')
    spike_unwar_dict = read_parquet_files_into_dict(
        '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/spikes_unwarped')
    # load changepoints here or with RNN? doesn't matter either way
    standardized_changepoints_dict = load_standardized_changepoints(
        '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/changepoints')
else:
    print('No .parquet files found, processing all objects...')
    fr_unwar_dict, spike_unwar_dict, fr_war_dict, spike_war_dict = FR_pipeline.full_pipeline()
    changepoints_dict = FR_pipeline.extract_changepoints_dict()  # do not use this one !!!
    standardized_changepoints_dict = {
        key.replace("dataset_", "").split("_repacked.npz")[0]: value
        for key, value in FR_pipeline.changepoints_dict.items()
    }
# Usage of the xarray objects is deprecated, but still occasionally needed. Keeping for now.
use_xarray = False  # switch to use Xarray or not
# have to eventually eliminate the usage of xarray, but for now I'm stuck. if I need, then we can use it.
for _ in range(use_xarray):
    FRprocessor = FRDataProcessor(
        npz_path=npz_path,
        info_path=info_path,
        pkl_path=pkl_path,
        taste_replacements=taste_replacements,
        # will create fr_unwarped/, fr_warped/, etc.
        output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS",
        window_length=250,
        step_size=25,
        fixed_warp_duration=1000,
    )
    print('Use xarray has been set to true. Processing entire object')
    (fr_unwar_xr,
     spikes_unwar_xr,
     fr_war_xr,
     spikes_war_xr,
     fr_unwar_df,
     spikes_unwar_df,
     fr_war_df,
     spikes_war_df, ) = FRprocessor.run_full()
########### FRPipeline_lite designed to work with RNNLatentprocessing... ##########

# from FRPipeline import FRPipeline_lite
# pipeline_lite = FRPipeline_lite(
#     npz_path=npz_path,
#     info_path=info_path,
#     taste_replacements={"NaCl":"NaCl","Sucrose":"Sucrose","Citric Acid":"Citric Acid","Quinine":"Quinine"},
#     output_dir='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS',
#     window_length=250,
#     step_size=25,
#     start_time=1500,
#     end_time=4500 + 225, # add the extra 225 to account for the beginnging 250 ms window which isn't in bin count
#     # this is a terrible work-around and I hate it.
#     save_outputs=True,
# )
# rr_fr_lite = pipeline_lite.full_pipeline()


###################### Init RNN processor for OCT rnn ############

# ==== Core Path Setup for the orinal thesis rnn ====
BASE_DIR_OCT = os.path.expanduser("/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work")
CUSTOM_SUBDIR_OCT = "RNN_PROCESSING_PARQUETS"  # <--- you can rename this as needed
RNN_OUT_DIR = os.path.join(BASE_DIR_OCT, CUSTOM_SUBDIR_OCT)


def build_path(*args):
    return os.path.join(RNN_OUT_DIR, *args)


# ==== Initialize Processor ====
RNNprocessor = RNNLatentProcessor(
    parquet_dir=os.path.join(BASE_DIR, "raw_lat_parquet"),
    npz_path=npz_path,
    info_path=info_path,
    pkl_path=pkl_path,
    taste_replacements=taste_replacements,
    save_dir=RNN_OUT_DIR,
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500,
    warp_length=1000,
    variance_threshold=95.0,
)
# additional optional bits:
# (defaults: enforce_fr_threshold=True, fr_threshold_hz=1.0)
RNNprocessor.enforce_fr_threshold = True   # set False to disable the 1 Hz gate
RNNprocessor.fr_threshold_hz = 1.0  # in hz


# ==== Check for existing parquet files ====
epoch_path = build_path("epoch_dataframes_unwarped")
RNN_processing_parquets = os.path.isdir(epoch_path) and any(
    f.endswith(".parquet") for f in os.listdir(epoch_path)
)
if RNN_processing_parquets:
    print('.parquet files processing RNN outputs already exist; skipping recalculating. Extracting...')
    epoch_dataframes_dict_uw = read_parquet_files_into_dict(build_path("epoch_dataframes_unwarped"))
    epoch_dataframes_dict_w = read_parquet_files_into_dict(build_path("epoch_dataframes_warped"))
    pca_95_uw = read_parquet_files_into_dict(build_path("robust_pca_95_unwarped"))
    pca_95_w = read_parquet_files_into_dict(build_path("robust_pca_95_warped"))
    pca_full_uw = read_parquet_files_into_dict(build_path("robust_pca_full_unwarped"))
    pca_full_w = read_parquet_files_into_dict(build_path("robust_pca_full_warped"))
    first_derivs_thresh_uw = read_parquet_files_into_dict(build_path("first_derivatives_95_unwarped"))
    second_derivs_thresh_uw = read_parquet_files_into_dict(build_path("second_derivatives_95_unwarped"))
    # consider adding the warped if needed
    standardized_changepoints_dict = load_standardized_changepoints(build_path("changepoints"))
else:
    print('No .parquet files processing RNN outputs exist; calculating and creating them.')
    (
        latents_unwarped, pca_thresh_unwarped, pca_full_unwarped,
        first_derivs_unwarped, second_derivs_unwarped,
        latents_warped, pca_thresh_warped, pca_full_warped,
        first_derivs_warped, second_derivs_warped
    ) = RNNprocessor.full_pipeline(
        compute_first_derivative=True,
        compute_second_derivative=True,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=True,  # have false for some reason? idrk
        compute_ttest=True,  # will throw errors about not having warped/unwarped data;
        # this is because it's being fed latents
    )
    changepoints_dict = RNNprocessor.extract_changepoints_dict(save_outputs=True)  # do not use this directly
    standardized_changepoints_dict = {
        key.replace("dataset_", "").split("_repacked.npz")[0]: value
        for key, value in RNNprocessor.changepoints_dict.items()
    }
    # ensuring the ttest runs:
    # !!! call to force ttest properly-- has to be done after.
    RNNprocessor.run_mwu_halfsplit(alpha=0.05, min_total_n=4)
    RNNprocessor.save_analysis_outputs(RNN_OUT_DIR)

# ATTEMPTING TO FEED THROUGH LITE RR FR DATA ####################3
BASE_DIR_RR = os.path.expanduser("/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work")
CUSTOM_SUBDIR_RR = "FR_PROCESSING_PARQUETS"
INPUT_NAME = "rr_firing_uw_lite"
CLEANED_NAME = f"{INPUT_NAME}_cleaned"
SAVE_NAME = f"{INPUT_NAME}_rnn_outputs"

RR_BASE_PATH = os.path.join(BASE_DIR_RR, CUSTOM_SUBDIR_RR)
RR_INPUT_DIR = os.path.join(RR_BASE_PATH, INPUT_NAME)
RR_CLEANED_DIR = os.path.join(RR_BASE_PATH, CLEANED_NAME)
RR_SAVE_DIR = os.path.join(RR_BASE_PATH, SAVE_NAME)

os.makedirs(RR_CLEANED_DIR, exist_ok=True)


def build_rr_path(*args):
    return os.path.join(RR_SAVE_DIR, *args)

# === Fix trial/time column ordering ===


def fix_trial_time_order(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns
    if "trial" in cols and "time" in cols:
        trial_idx = cols.index("trial")
        time_idx = cols.index("time")
        if trial_idx < time_idx:
            new_cols = cols[:]
            new_cols[trial_idx], new_cols[time_idx] = "time", "trial"
            df = df.rename(dict(zip(cols, new_cols)))
    signal_cols = [c for c in df.columns if c.startswith("neuron_")]
    final_order = signal_cols + ["taste", "trial", "time"]
    return df.select(final_order)


# === Check if output already exists ===
epoch_path = build_rr_path("epoch_dataframes_unwarped")
parquets_exist = os.path.isdir(epoch_path) and any(
    f.endswith(".parquet") for f in os.listdir(epoch_path)
)

if parquets_exist:
    print("[RR FR] Output .parquet files found — skipping reprocessing and loading from disk.")

    epoch_dataframes_dict_rr_uw = read_parquet_files_into_dict(build_rr_path("epoch_dataframes_unwarped"))
    epoch_dataframes_dict_rr_w = read_parquet_files_into_dict(build_rr_path("epoch_dataframes_warped"))
    pca_thresh_rr_uw = read_parquet_files_into_dict(build_rr_path("robust_pca_95_unwarped"))
    pca_thresh_rr_w = read_parquet_files_into_dict(build_rr_path("robust_pca_95_warped"))
    pca_full_rr_uw = read_parquet_files_into_dict(build_rr_path("robust_pca_full_unwarped"))
    pca_full_rr_w = read_parquet_files_into_dict(build_rr_path("robust_pca_full_warped"))
    first_derivs_rr_uw = read_parquet_files_into_dict(build_rr_path("first_derivatives_95_unwarped"))
    first_derivs_rr_w = read_parquet_files_into_dict(build_rr_path("first_derivatives_95_warped"))
    standardized_changepoints_dict = load_standardized_changepoints(build_rr_path("changepoints"))
    # also spike trains: pathing hardcoded for ease
    spike_trains = read_parquet_files_into_dict(
        '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS/spikes_trains')
else:
    print("[RR FR] No .parquet output found — cleaning input and running full pipeline.")
    from FRPipeline import FRPipeline_lite
    pipeline_lite = FRPipeline_lite(
        npz_path=npz_path,
        info_path=info_path,
        taste_replacements={"NaCl": "NaCl", "Sucrose": "Sucrose", "Citric Acid": "Citric Acid", "Quinine": "Quinine"},
        output_dir='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/FR_PROCESSING_PARQUETS',
        window_length=250,
        step_size=25,
        start_time=1500,
        end_time=4500 + 200,  # add the extra 225 to account for the beginnging 250 ms window which isn't in bin count
        # this is a terrible work-around and I hate it. But it will unfortunately have to do.
        save_outputs=True,
    )
    rr_fr_lite, spike_train_raw = pipeline_lite.full_pipeline()
    pred_fr_df = read_parquet_files_into_dict(RR_INPUT_DIR)
    for key, df in pred_fr_df.items():
        fixed_df = fix_trial_time_order(df)
        pred_fr_df[key] = fixed_df
        filepath = os.path.join(RR_CLEANED_DIR, f"{key}.parquet")
        fixed_df.write_parquet(filepath)

    # Run processor
    RNNprocessor = RNNLatentProcessor(
        parquet_dir=RR_CLEANED_DIR,
        npz_path=npz_path,
        info_path=info_path,
        pkl_path=pkl_path,
        taste_replacements=taste_replacements,
        save_dir=RR_SAVE_DIR,  # Distinct save path
        bin_size_ms=25,
        start_time_ms=1500,
        max_time_ms=4500,
        warp_length=1000,
        variance_threshold=95.0,
    )

    (
        fr_unwar_rr,
        pca_thresh_uw_rr,
        pca_full_uw_rr,
        first_derivs_uw_rr,
        _,
        fr_war_rr,
        pca_thresh_w_rr,
        pca_full_w_rr,
        first_derivs_w_rr,
        _,
    ) = RNNprocessor.full_pipeline(
        compute_first_derivative=True,
        compute_second_derivative=True,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=True,
    )
    # !!! call to force ttest properly-- has to be done after.
    RNNprocessor.run_mwu_halfsplit(alpha=0.05, min_total_n=4)
    RNNprocessor.save_analysis_outputs(RR_SAVE_DIR)
    changepoints_dict = RNNprocessor.extract_changepoints_dict(save_outputs=True)
    standardized_changepoints_dict = {
        key.replace("dataset_", "").split("_repacked.npz")[0]: value
        for key, value in RNNprocessor.changepoints_dict.items()
    }


############## JULY RNN PROCESS ###################
# RNN predicted firing -- something about this run is concerning me and I can't tell what...
# raw output that I need to then organize into the correct file type--
# === Core Path Setup -- JULY RNN -- WORK IN PROGRESS ANALYSIS ===
BASE_DIR = os.path.expanduser("~/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work")
JULY_SUBDIR = "july_rnn"
LATENT_INPUT_SUBDIR = os.path.join(JULY_SUBDIR, "july_25_output", "pred_latent")
JULY_OUTPUT_SUBDIR = os.path.join(JULY_SUBDIR, "RNN_PROCESSING_PARQUETS")
JULY_LATENT_INPUT_DIR = os.path.join(BASE_DIR, LATENT_INPUT_SUBDIR)
JULY_SAVE_DIR = os.path.join(BASE_DIR, JULY_OUTPUT_SUBDIR)


def build_july_path(*args):
    return os.path.join(JULY_SAVE_DIR, *args)


# === Check if .parquet outputs already exist ===
epoch_path = build_july_path("epoch_dataframes_unwarped")
parquets_exist = os.path.isdir(epoch_path) and any(
    f.endswith(".parquet") for f in os.listdir(epoch_path)
)
# === Initialize RNN Processor ===
RNNprocessor = RNNLatentProcessor(
    parquet_dir=JULY_LATENT_INPUT_DIR,
    npz_path=npz_path,
    info_path=info_path,
    pkl_path=pkl_path,
    taste_replacements=taste_replacements,
    save_dir=JULY_SAVE_DIR,
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500,
    warp_length=1000,
    variance_threshold=95.0,
)
if parquets_exist:
    print("[July RNN] .parquet files found — skipping reprocessing and loading from disk.")
    epoch_dataframes_dict_l_j_uw = read_parquet_files_into_dict(build_july_path("epoch_dataframes_unwarped"))
    epoch_dataframes_dict_l_j_w = read_parquet_files_into_dict(build_july_path("epoch_dataframes_warped"))
    pca_thresh_uw_l_j = read_parquet_files_into_dict(build_july_path("robust_pca_95_unwarped"))
    pca_thresh_w_l_j = read_parquet_files_into_dict(build_july_path("robust_pca_95_warped"))
    pca_full_uw_l_j = read_parquet_files_into_dict(build_july_path("robust_pca_full_unwarped"))
    pca_full_w_l_j = read_parquet_files_into_dict(build_july_path("robust_pca_full_warped"))
    first_derivs_uw_l_j = read_parquet_files_into_dict(build_july_path("first_derivatives_95_unwarped"))
    first_derivs_w_l_j = read_parquet_files_into_dict(build_july_path("first_derivatives_95_warped"))
    standardized_changepoints_dict = load_standardized_changepoints(build_july_path("changepoints"))
else:
    print("[July RNN] No .parquet files found — running full pipeline.")

    (
        latents_unwarped_j,
        pca_thresh_unwarped_j,
        pca_full_unwarped_j,
        first_derivs_unwarped_j,
        _,
        latents_warped_j,
        pca_thresh_warped_j,
        pca_full_warped_j,
        first_derivs_warped_j,
        _,
    ) = RNNprocessor.full_pipeline(
        compute_first_derivative=True,
        compute_second_derivative=True,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=True,
    )
    # specifically for running the MWU ttest:
    # !!! call to force ttest properly-- has to be done after.
    RNNprocessor.run_mwu_halfsplit(alpha=0.05, min_total_n=4)
    RNNprocessor.save_analysis_outputs(JULY_SAVE_DIR)

    changepoints_dict = RNNprocessor.extract_changepoints_dict(save_outputs=True)  # do not use this raw
    standardized_changepoints_dict = {
        key.replace("dataset_", "").split("_repacked.npz")[0]: value
        for key, value in RNNprocessor.changepoints_dict.items()
    }


# doing similar processing for the pred. fr; have to do some pre-processing first:
######################## REALLY DOUBLE CHECK THIS WHOLE MESS ##########################
# as I'm very concerned about accuracy...
# === Paths for the firing rate stuff ===
BASE_DIR = os.path.expanduser("~/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work")
JULY_SUBDIR = "july_rnn"
FR_INPUT_SUBDIR = os.path.join(JULY_SUBDIR, "july_25_output", "pred_fr")
FR_OUTPUT_SUBDIR = os.path.join(JULY_SUBDIR, "july_25_output", "pred_fr_clean")
FR_SAVE_SUBDIR = os.path.join(JULY_SUBDIR, "PRED_FR_RNN")
FR_INPUT_DIR = os.path.join(BASE_DIR, FR_INPUT_SUBDIR)
FR_OUTPUT_DIR = os.path.join(BASE_DIR, FR_OUTPUT_SUBDIR)
FR_SAVE_DIR = os.path.join(BASE_DIR, FR_SAVE_SUBDIR)


def build_fr_path(*args):
    return os.path.join(FR_SAVE_DIR, *args)


os.makedirs(FR_OUTPUT_DIR, exist_ok=True)
# === Fix trial/time column ordering ===


def fix_trial_time_order(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns
    # Ensure "trial" and "time" are in correct order
    if "trial" in cols and "time" in cols:
        trial_idx = cols.index("trial")
        time_idx = cols.index("time")
        if trial_idx < time_idx:
            new_cols = cols[:]
            new_cols[trial_idx], new_cols[time_idx] = "time", "trial"
            df = df.rename(dict(zip(cols, new_cols)))
    # Identify neuron columns
    signal_cols = [c for c in df.columns if c.startswith("neuron_")]
    # new: normalize neuron columns to Hz (each bin is 25 ms → divide by 0.025)
    # eliminate this when you actually fix it when running the rnn...
    df = df.with_columns([
        (pl.col(col) / 0.025).alias(col) for col in signal_cols
    ])
    final_order = signal_cols + ["taste", "trial", "time"]
    return df.select(final_order)


# think about the fact that the RNN has inferred neagtive rates, which is expected-- if not possible, strictly speaking...
# === Check if outputs already exist ===
epoch_path = build_fr_path("epoch_dataframes_unwarped")
parquets_exist = os.path.isdir(epoch_path) and any(
    f.endswith(".parquet") for f in os.listdir(epoch_path)
)
if parquets_exist:
    print("[Pred FR] Output .parquet files found — skipping reprocessing and loading from disk.")

    epoch_dataframes_dict_fr_j_uw = read_parquet_files_into_dict(build_fr_path("raw_output_unwarped"))
    epoch_dataframes_dict_fr_j_w = read_parquet_files_into_dict(build_fr_path("raw_output_warped"))
    pca_thresh_uw_fr_j = read_parquet_files_into_dict(build_fr_path("robust_pca_95_unwarped"))
    pca_thresh_w_fr_j = read_parquet_files_into_dict(build_fr_path("robust_pca_95_warped"))
    pca_full_uw_fr_j = read_parquet_files_into_dict(build_fr_path("robust_pca_full_unwarped"))
    pca_full_w_fr_j = read_parquet_files_into_dict(build_fr_path("robust_pca_full_warped"))
    first_derivs_uw_fr_j = read_parquet_files_into_dict(build_fr_path("first_derivatives_95_unwarped"))
    first_derivs_w_fr_j = read_parquet_files_into_dict(build_fr_path("first_derivatives_95_warped"))
    standardized_changepoints_dict = load_standardized_changepoints(build_fr_path("changepoints"))

else:
    print("[Pred FR] No .parquet output found — cleaning input and running full pipeline.")

    # Preprocess all predicted FR files
    pred_fr_df = read_parquet_files_into_dict(FR_INPUT_DIR)
    for key, df in pred_fr_df.items():
        fixed_df = fix_trial_time_order(df)
        pred_fr_df[key] = fixed_df
        filename = f"{key}.parquet"
        filepath = os.path.join(FR_OUTPUT_DIR, filename)
        fixed_df.write_parquet(filepath)
    # Run processor
    RNNprocessor = RNNLatentProcessor(
        parquet_dir=FR_OUTPUT_DIR,
        npz_path=npz_path,
        info_path=info_path,
        pkl_path=pkl_path,
        taste_replacements=taste_replacements,
        save_dir=FR_SAVE_DIR,
        bin_size_ms=25,
        start_time_ms=1500,
        max_time_ms=4500,
        warp_length=1000,
        variance_threshold=95.0,
    )
    (
        fr_unwarped_j,
        pca_thresh_unwarped_fr_j,
        pca_full_unwarped_fr_j,
        first_derivs_unwarped_fr_j,
        _,
        fr_warped_j,
        pca_thresh_warped_fr_j,
        pca_full_warped_fr_j,
        first_derivs_warped_fr_j,
        _,
    ) = RNNprocessor.full_pipeline(
        compute_first_derivative=True,
        compute_second_derivative=True,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=True,
    )
    # !!! call to force ttest properly-- has to be done after.
    RNNprocessor.run_mwu_halfsplit(alpha=0.05, min_total_n=4)
    RNNprocessor.save_analysis_outputs(FR_SAVE_DIR)
    changepoints_dict = RNNprocessor.extract_changepoints_dict(save_outputs=True)
    standardized_changepoints_dict = {
        key.replace("dataset_", "").split("_repacked.npz")[0]: value
        for key, value in RNNprocessor.changepoints_dict.items()
    }

# TODO: clean names, enforce the corerct data type (same as the one I use elsewhere)-- including working in time, cp, etc
# number of trials simply is not adding up if 'trial' is taste x trial-- seems as if we're missing a whole block of 30
# ugh does this mean a re-running of the script?

# the base runtime here conusmes GIGABYTES of ram, this can and should be made much cleaner prior to being handed to abu

# Run everything here to reload stuff
# %% cell dividing line
# sig-testing rr firing rate

sig_tester = NeuronSignificanceTester(
    dataset_dict=epoch_dataframes_dict_rr_uw,
    output_path='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_nrns_parquet',
    alpha=0.05,
    min_mean_fr=2.0
)
sig_tester.run()  # running on the unwarped rr window firing rate stuff only-- see if I can build support into other things
# for this type of data...

# plot experimentation -- for the oct rnn

pltpipeline = PlottingPipeline(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PLTpipeline",
    start_time_ms=1500,
    end_time_ms=4500,
    stim_time_ms=2000,
)

# 4) Run it—this will detect “_warped” in the keys and call your scatter‐metric plot:
pltpipeline.run(fr_war_dict)
# testing ttest
ttest = read_parquet_files_into_dict(
    "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/RNN_PROCESSING_PARQUETS/robust_pca_95_ttest_warped")
pltpipeline.plot_ttest_venn(ttest)

# for the newer rnn (july):

pltpipeline = PlottingPipeline(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/july_rnn/fig_data",
    start_time_ms=1500,
    end_time_ms=4500,
    stim_time_ms=2000,
)


# 4) Run it—this will detect “_warped” in the keys and call your scatter‐metric plot:
# pltpipeline.run(# your dict of dataframes here)


# you guessed it, we're back to significance testing:
# QUANTIFICATION OF SOME OF THIS MESS: shuffling data and testing against it:

# note: the new RNN pred. fr are actually already PCA'd-- so rather than running PCA on them again,
# just pass them individually.
# the first dataset we pass will be the rolling window firing rate (ground truth)
# second DS we pass will be the inferred data

binned_spk = read_parquet_files_into_dict(
    '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINN_spike_rnn_lat_comp/binned_spikes')

# finding correlation between the binned spikes and the RNN-inferred fr
tester = SignificanceTester_test(
    binned_spk,
    epoch_dataframes_dict_fr_j_uw,
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/july_rnn/comparisons_binned_spk",
    normalize=False,  # note: normalization happens after thresholding-- is not affected by thresholding
    enable_filtering=False,  # aka filtering by 2Hz threshold
    min_trial_mean=2.0
)

# trying something:
# this is a bad idea -- do not trust. Not useful metric.
tester = SignificanceTester_test(
    binned_spk,
    pca_full_uw,  # this is not a great idea because these correlations are done on the basis
    # of numerical index, so only the first eight figs are even matched.
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINN_spike_rnn_lat_comp/corr_figs",
    normalize=False,  # note: normalization happens after thresholding-- is not affected by thresholding
    enable_filtering=False,  # aka filtering by 2Hz threshold
    min_trial_mean=2.0
)

results, psth_df = tester.run_pipeline(run_ttest=False, run_spearman=True)
tester.plot_spearman_avg_corr_histograms(results)
tester.plot_spearman_rho_squared_histograms(results)
tester.plot_pearson_avg_corr_histograms(results)
tester.plot_pearson_r_sq_histograms(results)
tester.plot_sklearn_r2_histograms(results)
tester.plot_corr_histograms_all_data(results_dict=results, kind="pearson_r")
tester.plot_corr_histograms_all_data(results_dict=results, kind="pearson_r_sq")
tester.plot_corr_histograms_all_data(results_dict=results, kind="spearman_rho")
tester.plot_corr_histograms_all_data(results_dict=results, kind="spearman_rho_sq")
tester.plot_corr_histograms_all_data(results_dict=results, kind="sklearn_r2")
tester.plot_trial_averaged_psth_overlay_grid(psth_df)


corrprocessor = latent_spike_train_correlation(
    data_dict_a=pca_full_uw,                       # latents-- using the full 8 PCs for now...
    data_dict_b=spike_trains,                       # spikes (will be binned if needed)
    changepoints=standardized_changepoints_dict,
    output_dir='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINN_spike_rnn_lat_comp',
)
corr_results = corrprocessor.run_all(
    start_time=1500,
    end_time=4400,
    bin_size_ms=25,     # ignored if *_binned_spikes already exist (unless force_rebin=True)
    force_rebin=True,
)
# results:
correlation_results = read_parquet_files_into_dict(
    '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINN_spike_rnn_lat_comp/latent_spike_corr_meta')

# checking the troubled output of this script: # debug-- a check for the above since I was having issues


# plotting these correlations:
corrprocessor.plot_correlation_heatmaps(
    corr_dict=correlation_results,
    metric="spearman_rho",        # or: "spearman_rho", "spearman_rho2", "pearson_r", "pearson_r2"
    max_trials_per_fig=5        # chunk to 5 trials per plot
)
# this but paralelll
corrprocessor.plot_correlation_heatmaps_parallel(
    corr_dict=correlation_results,
    metric="pearson_r",        # or: "spearman_rho", "spearman_rho2", "pearson_r", "pearson_r2"
    max_trials_per_fig=5        # chunk to 5 trials per plot
)
corrprocessor.plot_correlation_histograms_parallel(corr_dict=correlation_results,
                                                   metric="pearson_r",
                                                   n_bins=30,
                                                   density=False,
                                                   max_trials_per_fig=5,
                                                   n_jobs=4,
                                                   share_bin_range=True,
                                                   use_abs=False,
                                                   show_zero=True,
                                                   show_mean=False,
                                                   show_median=False
                                                   )
# now the raster plot business:
# to be run on a lighter-ram process, as this is enormously memory intenseive for some reason
corrprocessor.plot_pc_raster_overlay(corr_dict=correlation_results,
                                     start_time=1500,
                                     end_time=4400,
                                     spearman_abs_thresh=0.20,  # adjust to proper thresholding at some point
                                     pearson_abs_thresh=0.20,  # have to adjust to proper thresholding
                                     sign_source="spearman",  # "spearman", "pearson", "auto" is spearman
                                     max_pcs_per_fig=4,
                                     stim_time_ms=2000,
                                     force_show_all_neurons=True)


####################################
# plotting binned spikes just to be sure:

pltpipeline = PlottingPipeline(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINN_spike_rnn_lat_comp/binned_viz",
    start_time_ms=1500,
    end_time_ms=4500,
    stim_time_ms=2000,
)
# 4) Run it—this will detect “_warped” in the keys and call your scatter‐metric plot:
pltpipeline.run(binned_spk)
###############################

# pca latent for the binned spk:
pca_binned_spk_lat = read_parquet_files_into_dict(
    '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINNED_SPIKES_PCA/PCA_BINNED_SPK_LAT')
# the firing infrence for the PCA:
pca_infer_binn_sp_fr = read_parquet_files_into_dict(
    '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINNED_SPIKES_PCA/PCA_INFR_BINN_SPK_FR')
# quick plot for idiot check-- standard plots:

pltpipeline = PlottingPipeline(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINNED_SPIKES_PCA/fig_data_pca_lat",
    start_time_ms=1500,
    end_time_ms=4500,
    stim_time_ms=2000,
)
pltpipeline.run(pca_binned_spk_lat)

# same with recostruction:
# this should give a quantification of the performance of the PCA for inferring firing rates
#
# effective R^2 value
pltpipeline = PlottingPipeline(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINNED_SPIKES_PCA/fig_data_pca_infer_fr",
    start_time_ms=1500,
    end_time_ms=4500,
    stim_time_ms=2000,
)
pltpipeline.run(pca_infer_binn_sp_fr)


# correlation between the binned spike and the RNN-inferred correlations
tester = SignificanceTester_test(
    binned_spk,
    epoch_dataframes_dict_fr_j_uw,  # inferred firing rate for the RNN
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/july_rnn/comparisons_binned_spk_vs_RNN_fr",
    normalize=False,  # note: normalization happens after thresholding-- is not affected by thresholding
    enable_filtering=False,  # aka filtering by 2Hz threshold
    min_trial_mean=2.0
)
bin_spk_rnn_corr_results, _ = tester.run_pipeline(run_ttest=False, run_spearman=True)


# testing the reconstruction:
# trying something:

tester = SignificanceTester_test(
    binned_spk,  # or binned_spk
    pca_infer_binn_sp_fr,  # the pca_inferred firing rate
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/BINNED_SPIKES_PCA/fig_data_pca_corr_bin_spk",
    normalize=False,  # note: normalization happens after thresholding-- is not affected by thresholding
    enable_filtering=False,  # aka filtering by 2Hz threshold
    min_trial_mean=2.0
)
# single var stuff
binn_spk_pca_corr_results, psth_df = tester.run_pipeline(run_ttest=False, run_spearman=True)
results = binn_spk_pca_corr_results
tester.plot_spearman_avg_corr_histograms(results)
tester.plot_spearman_rho_squared_histograms(results)
tester.plot_pearson_avg_corr_histograms(results)
tester.plot_pearson_r_sq_histograms(results)
tester.plot_sklearn_r2_histograms(results)
tester.plot_corr_histograms_all_data(results_dict=results, kind="pearson_r")
tester.plot_corr_histograms_all_data(results_dict=results, kind="pearson_r_sq")
tester.plot_corr_histograms_all_data(results_dict=results, kind="spearman_rho")
tester.plot_corr_histograms_all_data(results_dict=results, kind="spearman_rho_sq")
tester.plot_corr_histograms_all_data(results_dict=results, kind="sklearn_r2")
tester.plot_trial_averaged_psth_overlay_grid(psth_df)
# above-- testing


# testing this multivar correlation:

tester.plot_corr_histograms_diff_by_dataset(
    results_dict_A=bin_spk_rnn_corr_results, results_dict_B=binn_spk_pca_corr_results, kind="sklearn_r2", xlim=(-5, 1))

# also run a quick corr-- do this with RNN latent spk correlation, but down the line.

# insert here


####################################


# NOW FOR THE FFT-- studying the oscillatory stuff
# all the above is hooey-- this is where we're at now:


tld = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/FFT_FREQUENCY_SUITE"
pipeline = FrequencyAnalysisPipeline(
    tld=tld,
    standardized_changepoints_dict=standardized_changepoints_dict,
    modified_tastes=modified_tastes,
    peak_height_threshold=0.25,
    spectrogram_nperseg=8,
    min_freq=2,  # You can set this!
    max_freq=15,
    do_fft=True,
    do_periodogram=True,
    do_peaks=True,
    do_lombscargle=True,
    do_full_trials=True,
    do_spectrogram=True,  # And this too!
    do_individual_plots=True,
    smart_skip=True,
    min_amplitude_threshold=0.05,
)
# dataset_dict = {dataset_name: polars_dataframe}
pipeline.run(first_derivs)
# further improvments for this:
# a text file in the dataset dir that will tell me what's signficant and where
# batching plots on figs so I don't have a billion files
# RUNNING THIS ON AN EPOCH-BY-EPOCH BASIS:
# only do if the periodograms come back with some level of high-power oscillatory significance
# as otherwise this would take practically forever
# figure out quantificaiton of freuqencies as well


# 4/15: Optimized version of this code to deal in alignment and warping and stuff--
# (defining a class for this business as needed-- this is taking a lot of time for not a whole lot of return).

# class moved to its own module; func that calls the plotting below:

# deal with this moving forward too


# new one:
def process_and_plot_datasets(
    npz_path,
    pkl_path,
    base_output_dir,
    standardized_changepoints_dict=None,
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000,
):
    """
    CLI-enhanced pipeline for generating raster plots based on user options.
    Now avoids saving CalcFRStates unless debug mode is selected.
    """

    # Initial CLI prompt (just to check for debug mode)
    mode, is_warped, alignment, show_markers, sort_by_length = (
        SpikeRasterPlotter.cli_options()
    )

    filename = os.path.basename(npz_path)
    base_name = filename.split("_repacked.npz")[0]
    all_calc_objs = {}

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple) and len(data) == 4:
            spike_array, dataset_num, index, key = data

            dataset_num_str = str(dataset_num)
            dataset_num_clean = dataset_num_str.split("_repacked.npz")[0]
            core_dataset_name = dataset_num_clean

            print(f"Processing dataset number: {dataset_num_clean}")

            extracted_pkl = extract_valid_changepoints(
                pkl_path, spike_array, dataset_num_clean, index, key
            )

            if extracted_pkl is not None:
                try:
                    changepoints = extracted_pkl[:, 3]

                    calc = CalcFRStates(
                        spike_array=spike_array,
                        changepoints=changepoints,
                        window_length=window_length,
                        step_size=step_size,
                        compute_unwarped_spike_arrays=True,
                        compute_unwarped_firing_rates=False,
                        compute_warped_spike_arrays=True,
                        compute_warped_firing_rates=False,
                        fixed_warp_duration=fixed_warp_duration,
                    )

                    fr_unwarped, spike_unwarped, fr_warped, spike_warped = calc.run()

                    if mode == "debug":
                        calc.spike_arrays_unwarped = spike_unwarped
                        calc.spike_arrays_warped = spike_warped
                        all_calc_objs[core_dataset_name] = calc

                    else:
                        spike_data = spike_warped if is_warped else spike_unwarped
                        dataset_dir = os.path.join(
                            base_output_dir, f"dataset_{dataset_num_clean}"
                        )
                        os.makedirs(dataset_dir, exist_ok=True)

                        plotter = SpikeRasterPlotter(
                            spike_data_array=spike_data,
                            dataset_dir=dataset_dir,
                            dataset_name=core_dataset_name,
                            base_name=base_name,
                            is_warped=is_warped,
                            mode=mode,
                            alignment=alignment,
                            show_markers=show_markers,
                            sort_by_length=sort_by_length,
                            changepoints_dict=standardized_changepoints_dict,
                        )

                        plotter.plot_all()

                except Exception as e:
                    print(f"Error processing dataset {dataset_num_clean}: {e}")
            else:
                print(
                    f"No valid changepoints for dataset {dataset_num_clean}; skipping..."
                )

    if mode == "debug":
        if not all_calc_objs:
            print("No datasets were successfully loaded. Cannot enter debug mode.")
            return
        SpikeRasterPlotter.debug_extract_from_calc(all_calc_objs)

    print("All datasets processed.")


### all manner of spike train hooey above from a class that is called ###


# also for warping etc.


# OLD

# so... I should probably eliminate this at some point. There's so much old tech-debt and other backlog stuff that I need to go through.

"""

###### this is very old code that I'll have to clean up at a later date #### note as of 4/18

## CHECKING VIABILITY FOR CHANGING THE NUMBER OF LATANTS IN EACH PLOT -- old buiz (gotta do major housecleaning)
# using PCA to optimize for the number of latants I'm inferring


# loop for going through the thresholded neurons; doing the whole PCA analysis pipeline on them too
# all these plots are now in a separate file
base_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/pca_thresholded_w_poststim"
os.makedirs(base_dir, exist_ok=True)
# Iterate over each DataFrame in sig_nrns_dict
for key, df in sig_nrns_dict.items():
    # Create a subdirectory for the current DataFrame
    sub_dir = os.path.join(base_dir, key)
    os.makedirs(sub_dir, exist_ok=True)
    # Apply the PCA function to the current DataFrame
    pca_results_df = whitened_pca_all_nrns_sep_taste(df, sub_dir)
    plot_averaged_pca_trajectories_thresholded_SEM(
        pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size
    )
# plot_averaged_pca_heatmaps_all_taste(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
# plot_pca_scatter(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
# plot_pca_scatter_3d(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
# distance_df = calculate_pca_distances(pca_results_df)
# plot_euclidean_distances_bar(distance_df, sub_dir, modified_tastes, epoch_labels, step_size)
# plot_explained_variance_combined_by_taste(pca_results_df, sub_dir, modified_tastes, epoch_labels)

# actual generator that runs through ALL .npz files and datasets; just kinda doing everything
# creating a dataframe for ALL significant neurons

# all under generator that I'm increasingly not using anymore in favor of newer, faster code
for data in extract_from_npz(npz_path):
    if isinstance(data, tuple):
        spike_array, dataset_num, index, key = data
        print(f"Dataset number: {dataset_num}")
        dataset_tastes = process_info_files(info_path, dataset_num)
        modified_tastes = modify_tastes(dataset_tastes, taste_replacements)
        # Call unpickle_changepoints to process the .pkl files for the current dataset_num
        extracted_pkl = unpickle_changepoints(
            pkl_path, [(spike_array, dataset_num, index, key)]
        )
        if extracted_pkl is not None:
            # Process the extracted_pkl data as needed
            print("Processed .pkl data:")
        else:
            print("Failed to process .pkl data")
        try:
            changepoints = extracted_pkl[
                :, 3
            ]  # Giving 4 changepoint arrays, 1 for each taste-- for each bit of data
            window_length = 250  # make 25 for just straight binning, no sliding window lol (might work); for sliding make 250
            step_size = 25  # for sliding make 25
            alpha = 0.05  # for the t-test being run

            # start_bin = 000/25 # not needed as specified inside function now
            # end_bin = 7000/25 # not needed as specified inside function now
            # Calculate firing rates with changepoints
            state_firing_rates_all_trials, state_spike_arrays_all_trials = (
                calc_fr_states(spike_array, changepoints, window_length, step_size)
            )
            # Process firing rate data as needed
            print(
                f"Firing rate with changepoints calculated for dataset {dataset_num}:"
            )
            all_tastes_averaged, all_interpolated_states = (
                interpolate_and_average_all_tastes(state_firing_rates_all_trials)
            )

            ### plotting functions ###

            ## Warped Plots ##
            # plt_inter_cpfr_avg_nrn_line(all_interpolated_states, modified_tastes, dataset_num)
            # plt_inter_cpfr_avg_tr_line(all_interpolated_states, modified_tastes, dataset_num)
            # plt_single_neuron_firing_warped_line(all_interpolated_states, dataset_num, modified_tastes, output_dir_warped)

            ## Unwarped Plots ##
            # plt_single_neuron_firing_unwarped_line(state_firing_rates_all_trials, dataset_num, output_dir_unwarped, modified_tastes, epoch_labels, step_size)
        # plt_single_neuron_firing_unwarped_heatmap(state_firing_rates_all_trials, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels)

        ## Changepoint plots ##
        # plt_changepoints_scat(changepoints, modified_tastes, dataset_num)
        # plt_changepoint_hist(changepoints, modified_tastes, dataset_num)

        ## Unwarped plots with significance dictated via t-test or 1 hz firing threshold## -- old
        # thresh_firing_rates_unwarped = thresh_ttest_unwarped_hz(state_firing_rates_all_trials, alpha) # significance via ttest
        # thresh_firing_rates_unwarped = thresh_min_fr_unwarped_hz(state_firing_rates_all_trials, threshold_hz) # signifigance via 1 hz firing rate
        # plt_single_neuron_firing_unwarped_heatmap_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels)
        # plt_single_neuron_firing_unwarped_line_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, modified_tastes, epoch_labels, step_size)

        # appending individual significant neurons to one big happy dataset
        except TypeError as e:
            if str(e) == "'float' object is not iterable":
                print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
            else:
                raise
                # error handling for one dataset whose data type seems to be irritating
    else:
        spike_array, index, key = data
        dataset_num = "Unknown Dataset"  # as in for some reason we can't get a number
        print(f"Dataset number: {dataset_num}")


"""
