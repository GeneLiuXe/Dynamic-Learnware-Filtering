import os
import torch
import shutil
import zipfile
import tempfile
import importlib
import numpy as np
from typing import Optional, List, Tuple
from learnware.tests.templates import StatSpecTemplate
from learnware.specification import generate_stat_spec

from ..templates import OurModelTemplate, OurLearnwareTemplate
from ..config import REGRESSION_DATASETS, CLASSIFICATION_DATASETS, LEARNWARE_ZIP_DIR, LEARNWARE_FOLDER_DIR, ERR_MAT_DIR, RKME_DIST_MAT_DIR
from .utils import get_learnware_from_dirpath, set_seed, save_pickle_file, load_pickle_file
from ..logger import get_module_logger

logger = get_module_logger("Benchmark")


class Benchmark:
    def __init__(self, dataset: str, **kwargs):
        self.dataset = dataset
        self.set_dataset(dataset, **kwargs)

    def get_dataset_list(self):
        return REGRESSION_DATASETS + CLASSIFICATION_DATASETS
    
    def set_dataset(self, dataset: str, **kwargs):
        self.loader = importlib.import_module(f".{dataset}", package="toolkit.benchmark").DataLoader(**kwargs)
        self.dataset = dataset

        self.err_mat = {}
        self.err_mat_path = os.path.join(ERR_MAT_DIR, f"err_mat_{self.dataset}.pkl")
        if os.path.exists(self.err_mat_path):
            self.err_mat = load_pickle_file(self.err_mat_path)
        else:
            os.makedirs(ERR_MAT_DIR, exist_ok=True)

        self.rkme_dist_mat = {}
        self.rkme_dist_mat_path = os.path.join(RKME_DIST_MAT_DIR, f"rkme_dist_mat_{self.dataset}.pkl")
        if os.path.exists(self.rkme_dist_mat_path):
            self.rkme_dist_mat = load_pickle_file(self.rkme_dist_mat_path)
        else:
            os.makedirs(RKME_DIST_MAT_DIR, exist_ok=True)

    def get_learnware_ids(self, first_num: Optional[int] = None):
        learnware_ids = self.loader.get_learnware_ids()
        if first_num is not None:
            learnware_ids = learnware_ids[:first_num]
        return learnware_ids

    def get_user_ids(self, first_num: Optional[int] = None):
        user_ids = self.loader.get_user_ids()
        if first_num is not None:
            user_ids = user_ids[:first_num]
        return user_ids

    def get_learnware_performance_array(self, learnware_idx: int, user_ids: List[int]):
        item_count = len(self.err_mat)
        performance_array = np.zeros(len(user_ids))

        keys = [f"{learnware_idx}-{user_idx}" for user_idx in user_ids]
        if all(key in self.err_mat for key in keys):
            for i, key in enumerate(keys):
                performance_array[i] = self.err_mat[key]
        else:
            learnware = self.get_idx_learnware(learnware_idx)

            def _get_shop_err(shop_idx) -> Tuple[float, int]:
                key = f"{learnware_idx}-shop{shop_idx}"
                len_key = f"len-shop{shop_idx}"
                if key not in self.err_mat:
                    _, _, val_x, val_y = self.loader.get_shop_data(shop_idx)
                    pred_y = learnware.predict(val_x)
                    self.err_mat[key] = self.score(val_y, pred_y)
                    self.err_mat[len_key] = len(val_y)
                return self.err_mat[key], self.err_mat[len_key]

            for i, user_idx in enumerate(user_ids):
                key = f"{learnware_idx}-{user_idx}"
                if key not in self.err_mat:
                    user_task_shop_ids = self.get_user_task_params(user_idx)["shop_ids"]
                    score, length = 0, 0
                    for shop_id in user_task_shop_ids:
                        shop_score, shop_length = _get_shop_err(shop_id)
                        score += shop_score * shop_length
                        length += shop_length
                    self.err_mat[key] = score / length
                performance_array[i] = self.err_mat[key]

        if item_count != len(self.err_mat):
            save_pickle_file(self.err_mat, self.err_mat_path)
        return performance_array

    def get_learnware_rkme_dist_array(self, learnware_idx: int, user_ids: List[int]):
        item_count = len(self.rkme_dist_mat)
        rkme_dist_array = np.zeros(len(user_ids))

        keys = [f"{learnware_idx}-{user_idx}" for user_idx in user_ids]
        if all(key in self.rkme_dist_mat for key in keys):
            for i, key in enumerate(keys):
                rkme_dist_array[i] = self.rkme_dist_mat[key]
        else:
            learnware = self.get_idx_learnware(learnware_idx)
            RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")

            for i, user_idx in enumerate(user_ids):
                key = f"{learnware_idx}-{user_idx}"
                if key not in self.rkme_dist_mat:
                    user_rkme = self.get_user_rkme(user_idx)
                    self.rkme_dist_mat[key] = RKME.dist(user_rkme)
                rkme_dist_array[i] = self.rkme_dist_mat[key]

        if item_count != len(self.rkme_dist_mat):
            save_pickle_file(self.rkme_dist_mat, self.rkme_dist_mat_path)
        return rkme_dist_array

    def get_learnware_params(self, idx):
        return self.loader.get_learnware_params(idx)
        
    def get_user_task_params(self, idx):
        return self.loader.get_user_task_params(idx)

    def get_idx_data(self, idx: int):
        """Get the dataset by learnware_id or user_id

        :param idx: learnware_id or user_id
        :return: train_x, train_y, val_x, val_y
        """
        return self.loader.get_idx_data(idx)
    
    def get_user_rkme(self, user_idx: int):
        return self.loader.get_user_rkme(user_idx)

    def get_idx_learnware(self, idx: int, regenerate_flag: bool = False):
        """Get the learnware by learnware_id

        Parameters
        ----------
        idx : int
            learnware_id
        """
        learnware_folderpath = os.path.join(LEARNWARE_FOLDER_DIR, self.dataset, f"{idx}")

        if regenerate_flag or not os.path.exists(learnware_folderpath):
            learnware_zippath = os.path.join(LEARNWARE_ZIP_DIR, self.dataset, f"{idx}.zip")
            
            if os.path.exists(learnware_folderpath):
                shutil.rmtree(learnware_folderpath)
            if os.path.exists(learnware_zippath):
                os.remove(learnware_zippath)
            
            os.makedirs(os.path.dirname(learnware_zippath), exist_ok=True)
            model_path = self.loader.get_model_path(idx)

            train_x, train_y, val_x, val_y = self.get_idx_data(idx)
            input_shape = train_x.shape[1]
            predict_method = "predict"
            model_type = self.loader.get_learnware_params(idx)["model_type"]
            if model_type in ["rf", "xgboost", "lightgbm"]:
                model_class_name = "JoblibLoadedModel"

            if self.dataset in REGRESSION_DATASETS:
                output_shape = 1
            elif self.dataset in CLASSIFICATION_DATASETS:
                output_shape = len(np.unique(train_y))
            else:
                raise ValueError(f"Invalid dataset: {self.dataset}")
            
            with tempfile.TemporaryDirectory(suffix=f"{self.dataset}_{idx}_spec") as tempdir:
                set_seed(0)
                basic_spec_file_path = os.path.join(tempdir, "basic_rkme.json")
                memory_ratio = 1
                max_length = int(1e7)
                while True:
                    try:
                        if train_x.shape[0] <= max_length:
                            basic_rkme = generate_stat_spec(type="table", X=train_x)
                        else:
                            max_length = int(max_length * memory_ratio)
                            basic_rkme = generate_stat_spec(type="table", X=train_x[np.random.choice(train_x.shape[0], size=max_length, replace=False)])
                        torch.cuda.empty_cache()
                        break
                    except Exception as e:
                        memory_ratio *= 0.95
                        if max_length <= int(1e6): memory_ratio = 1
                        logger.error(f"max_length: {max_length}, memory_ratio: {memory_ratio}\nError: {e}")
                
                basic_rkme.save(basic_spec_file_path)
                stat_spec_templates = [
                    StatSpecTemplate(filepath=basic_spec_file_path, type=basic_rkme.type),
                ]

                OurLearnwareTemplate.generate_learnware_zipfile(
                    learnware_zippath=learnware_zippath,
                    model_template=OurModelTemplate(
                        class_name=model_class_name,
                        model_filepath=model_path,
                        model_kwargs={
                            "input_shape": (input_shape,),
                            "output_shape": (output_shape,),
                            "predict_method": predict_method,
                            "model_filename": os.path.basename(model_path),
                        },
                    ),
                    stat_spec_templates=stat_spec_templates,
                    requirements=["lightgbm==3.3.2"],
                )

            os.makedirs(learnware_folderpath, exist_ok=True)
            with zipfile.ZipFile(learnware_zippath, "r") as zip_ref:
                zip_ref.extractall(learnware_folderpath)

        learnware = get_learnware_from_dirpath(learnware_folderpath)
        learnware.id = idx
        model_type = self.loader.get_learnware_params(idx)["model_type"]
        if model_type == "rf":
            learnware.instantiate_model()
            learnware.model.model.set_params(n_jobs=1)

        return learnware

    def score(self, real_y, pred_y, sample_weight=None, multioutput="raw_values"):
        return self.loader.score(real_y, pred_y, sample_weight=sample_weight, multioutput=multioutput)

    def regenerate_data(self):
        self.loader.regenerate_data()

    def retrain_models(self):
        self.loader.retrain_models()
        
    def regenerate_learnwares(self):
        for idx in self.loader.get_learnware_ids():
            self.get_idx_learnware(idx, regenerate_flag=True)