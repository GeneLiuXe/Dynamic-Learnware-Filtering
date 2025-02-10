import os
import json
import torch
import random
import pickle
import numpy as np

from learnware.learnware import Learnware, get_stat_spec_from_config
from learnware.specification import Specification, RKMETableSpecification
from learnware.config import C
from learnware.utils import read_yaml_to_dict


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_uniform_distribution(N: int):
    return random.randint(1, N // 2)


def get_sampled_learnware_params(total_shop_num: int, sample_size: int, total_model_type: int = 3, total_model_param_num: int = 5):
    learnware_params = []
    model_type = ["rf", "xgboost", "lightgbm"]
    for _ in range(sample_size):
        selected_shop_num = sample_uniform_distribution(total_shop_num)
        param = {
            "shop_ids": list(random.sample(range(total_shop_num), selected_shop_num)),
            "model_type": model_type[random.randint(0, total_model_type - 1)],
            "param_idx": random.randint(0, total_model_param_num - 1),
        }
        learnware_params.append(param)
    return learnware_params


def get_sampled_user_task_params(total_shop_num: int, sample_size: int):
    user_task_params = []
    for _ in range(sample_size):
        selected_shop_num = sample_uniform_distribution(total_shop_num)
        param = {
            "shop_ids": list(random.sample(range(total_shop_num), selected_shop_num)),
        }
        user_task_params.append(param)
    return user_task_params


def save_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def save_pickle_file(data, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def load_pickle_file(file_path):
    with open(file_path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def get_learnware_from_dirpath(learnware_dirpath: str):
    learnware_yaml_path = os.path.join(learnware_dirpath, C.learnware_folder_config["yaml_file"])
    learnware_config = read_yaml_to_dict(learnware_yaml_path)

    if "module_path" not in learnware_config["model"]:
        learnware_config["model"]["module_path"] = C.learnware_folder_config["module_file"]

    learnware_spec = Specification()
    for _stat_spec in learnware_config["stat_specifications"]:
        stat_spec = _stat_spec.copy()
        stat_spec_path = os.path.join(learnware_dirpath, stat_spec["file_name"])

        stat_spec["file_name"] = stat_spec_path
        if stat_spec["class_name"] == "RKMETableSpecification":
            stat_spec_inst = RKMETableSpecification()
            stat_spec_inst.load(stat_spec_path)
        else:
            stat_spec_inst = get_stat_spec_from_config(stat_spec)
        learnware_spec.update_stat_spec(**{stat_spec_inst.type: stat_spec_inst})

    return Learnware(
        id=os.path.basename(learnware_dirpath), model=learnware_config["model"], specification=learnware_spec, learnware_dirpath=learnware_dirpath,
    )