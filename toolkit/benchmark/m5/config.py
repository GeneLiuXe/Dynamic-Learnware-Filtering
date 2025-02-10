import os
import math

ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "data"))
RAW_DATA_DIR = os.path.join(ROOT_PATH, "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_PATH, "processed")
MODEL_DIR = os.path.join(ROOT_PATH, "models")
USER_RKME_DIR = os.path.join(ROOT_PATH, "rkme")
PARAM_DIR = os.path.join(ROOT_PATH, "params")

SHOP_NUM = 70
TASK_TYPE = "regression"

SEP = 1000000
LEARNWARE_NUM = 10000
USER_TASK_NUM = 1000

TARGET = "sales"
START_TRAIN = 1
END_TRAIN = 1941 - 28

category_list = ["item_id", "dept_id", "cat_id", "event_name_1", "event_name_2", "event_type_1", "event_type_2"]
features_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "release",
    "sell_price",
    "price_max",
    "price_min",
    "price_std",
    "price_mean",
    "price_norm",
    "price_nunique",
    "item_nunique",
    "price_momentum",
    "price_momentum_m",
    "price_momentum_y",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "tm_d",
    "tm_w",
    "tm_m",
    "tm_y",
    "tm_wm",
    "tm_dw",
    "tm_w_end",
    "sales_lag_28",
    "sales_lag_29",
    "sales_lag_30",
    "sales_lag_31",
    "sales_lag_32",
    "sales_lag_33",
    "sales_lag_34",
    "sales_lag_35",
    "sales_lag_36",
    "sales_lag_37",
    "sales_lag_38",
    "sales_lag_39",
    "sales_lag_40",
    "sales_lag_41",
    "sales_lag_42",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_30",
    "rolling_std_30",
    "rolling_mean_60",
    "rolling_std_60",
    "rolling_mean_180",
    "rolling_std_180",
    "rolling_mean_tmp_1_7",
    "rolling_mean_tmp_1_14",
    "rolling_mean_tmp_1_30",
    "rolling_mean_tmp_1_60",
    "rolling_mean_tmp_7_7",
    "rolling_mean_tmp_7_14",
    "rolling_mean_tmp_7_30",
    "rolling_mean_tmp_7_60",
    "rolling_mean_tmp_14_7",
    "rolling_mean_tmp_14_14",
    "rolling_mean_tmp_14_30",
    "rolling_mean_tmp_14_60",
]
label_column = ["sales"]


store_list = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
dept_list = [0, 1, 2, 3, 4, 5, 6]
dataset_info = {
    "name": "M5",
    "range of date": "2011.01.29-2016.06.19",
    "description": "Walmart store, involves the unit sales of various products sold in the USA, organized in the form of grouped time series. More specifically, the dataset involves the unit sales of 3049 products, classified in 3 product categories (Hobbies, Foods, and Household).",
    "location": [
        "California, United States",
        "California, United States",
        "California, United States",
        "California, United States",
        "Texas, United States",
        "Texas, United States",
        "Texas, United States",
        "Wisconsin, United States",
        "Wisconsin, United States",
        "Wisconsin, United States",
    ],
}
