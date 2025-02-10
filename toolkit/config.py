import os
import logging

ROOT_PATH = os.path.abspath(os.path.join(__file__, ".."))
STORAGE_DIR = os.path.join(ROOT_PATH, "storage")

LEARNWARE_DIR = os.path.join(STORAGE_DIR, "learnware_pool")
LEARNWARE_ZIP_DIR = os.path.join(LEARNWARE_DIR, "zips")
LEARNWARE_FOLDER_DIR = os.path.join(LEARNWARE_DIR, "learnwares")

ERR_MAT_DIR = os.path.join(STORAGE_DIR, "err_mat")
RKME_DIST_MAT_DIR = os.path.join(STORAGE_DIR, "rkme_dist_mat")

RES_DIR = os.path.join(STORAGE_DIR, "results")
EVAL_DIR = os.path.join(RES_DIR, "evaluation")
FIG_DIR = os.path.join(RES_DIR, "figure")

REGRESSION_DATASETS = ["ppg", "air_quality", "m5"]
CLASSIFICATION_DATASETS = ["covertype", "har70", "diabetes"]

# For methods
MAX_LEVEL = 1e8
MAX_RADIUS = 1e8

LOGGER_CONFIG = {
    "logging_level": logging.INFO, # logger under this level will be ignored
    "logging_outfile": None,
}

PARAM_DICT = {
    "ppg": {
        "nearest_neighbor_search_K": 5,
        "rkme_cover_tree_params": {
            "loss_threshold": 10.0,
            "dist_weight": 250,
        },
        "rkme_thres": 1e-4,
    },
    "m5": {
        "nearest_neighbor_search_K": 5,
        "rkme_cover_tree_params": {
            "loss_threshold": 2.75,
            "dist_weight": 23,
        },
        "rkme_thres": 1e-4,
    },
    "har70": {
        "nearest_neighbor_search_K": 5,
        "rkme_cover_tree_params": {
            "loss_threshold": 0.03,
            "dist_weight": 4,
        },
        "rkme_thres": 1e-4,
    },
    "diabetes": {
        "nearest_neighbor_search_K": 5,
        "rkme_cover_tree_params": {
            "loss_threshold": 0.03,
            "dist_weight": 5,
        },
        "rkme_thres": 1e-4,
    },
    "covertype": {
        "nearest_neighbor_search_K": 20,
        "rkme_cover_tree_params": {
            "loss_threshold": 0.12,
            "dist_weight": 5,
        },
        "rkme_thres": 1e-4,
    },
    "air_quality": {
        "nearest_neighbor_search_K": 5,
        "rkme_cover_tree_params": {
            "loss_threshold": 8.0,
            "dist_weight": 100,
        },
        "rkme_thres": 1e-5,
    },
}