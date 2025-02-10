from math import gamma
from tkinter import Y
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
import os, sys, gc, time, warnings, pickle, psutil, random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .config import *


# Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2.0 ** 30, 2)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


# Memory Reducer
def reduce_mem_usage(df, float16_flag=True, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if float16_flag and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how="left")
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def acquire_data(store, dept, fill_flag=False):
    TARGET = "sales"
    suffix = f"_fill" if fill_flag else ""
    train = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, f"train_{store}_{dept}{suffix}.pkl"))
    val = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, f"val_{store}_{dept}{suffix}.pkl"))

    train_y = train[TARGET]
    train_x = train.drop(labels=TARGET, axis=1)
    val_y = val[TARGET]
    val_x = val.drop(labels=TARGET, axis=1)

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    val_x = val_x.to_numpy()
    val_y = val_y.to_numpy()

    return train_x, train_y, val_x, val_y