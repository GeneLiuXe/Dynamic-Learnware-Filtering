import os
import math

ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "data"))
RAW_DATA_DIR = os.path.join(ROOT_PATH, "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_PATH, "processed")
MODEL_DIR = os.path.join(ROOT_PATH, "models")
USER_RKME_DIR = os.path.join(ROOT_PATH, "rkme")
PARAM_DIR = os.path.join(ROOT_PATH, "params")

SHOP_NUM = 57
TASK_TYPE = "classification"

SEP = 1000000
LEARNWARE_NUM = 10000
USER_TASK_NUM = 1000

Elevation_Range = [0] + [2300 + 20 * i for i in range(56)] + [4000]