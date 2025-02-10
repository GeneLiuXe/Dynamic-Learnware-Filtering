import torch
import numpy as np
from typing import Union, List
from sklearn.metrics import accuracy_score
from learnware.reuse import JobSelectorReuser
from learnware.specification import RKMETextSpecification
from learnware.market.utils import parse_specification_type


class OurJobSelectorReuser(JobSelectorReuser):
    def _selector_grid_search(
        self,
        org_train_x: np.ndarray,
        org_train_y: np.ndarray,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        num_class: int,
    ):
        """Train a LGBMClassifier as job selector using the herding data as training instances.

        Parameters
        ----------
        org_train_x : np.ndarray
            The original herding features.
        org_train_y : np.ndarray
            The original hearding labels(which are learnware indexes).
        train_x : np.ndarray
            Herding features used for training.
        train_y : np.ndarray
            Herding labels used for training.
        val_x : np.ndarray
            Herding features used for validation.
        val_y : np.ndarray
            Herding labels used for validation.
        num_class : int
            Total number of classes for the job selector(which is exactly the total number of learnwares to be reused).

        Returns
        -------
        LGBMClassifier
            The job selector model.
        """
        try:
            from lightgbm import LGBMClassifier, early_stopping
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "JobSelectorReuser is not available because 'lightgbm' is not installed! Please install it manually."
            )

        score_best = -1
        learning_rate = [0.01]
        max_depth = [66]
        params = (0, 0)

        lgb_params = {"boosting_type": "gbdt", "n_estimators": 100, "boost_from_average": False, "verbose": -1}

        if num_class == 2:
            lgb_params["objective"] = "binary"
            lgb_params["metric"] = "binary_logloss"
        else:
            lgb_params["objective"] = "multiclass"
            lgb_params["metric"] = "multi_logloss"

        for lr in learning_rate:
            for md in max_depth:
                lgb_params["learning_rate"] = lr
                lgb_params["max_depth"] = md
                model = LGBMClassifier(**lgb_params)
                train_y = train_y.astype(int)
                model.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[early_stopping(300, verbose=False)])
                pred_y = model.predict(org_train_x)
                score = accuracy_score(pred_y, org_train_y)

                if score > score_best:
                    score_best = score
                    params = (lr, md)

        lgb_params["learning_rate"] = params[0]
        lgb_params["max_depth"] = params[1]
        model = LGBMClassifier(**lgb_params)
        model.fit(org_train_x, org_train_y)

        return model

    def predict(self, user_data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Give prediction for user data using baseline job-selector method

        Parameters
        ----------
        user_data : Union[np.ndarray, List[str]]
            User's unlabeled raw data.

        Returns
        ------
        np.ndarray
            Prediction given by job-selector method
        """
        raw_user_data = user_data
        if isinstance(user_data[0], str):
            stat_spec_type = parse_specification_type(self.learnware_list[0].get_specification().stat_spec)
            assert (
                stat_spec_type == "RKMETextSpecification"
            ), "stat_spec_type must be 'RKMETextSpecification' when user data is the List of string."
            user_data = RKMETextSpecification.get_sentence_embedding(user_data)

        select_result = self.job_selector(user_data)
        pred_y_list = []
        data_idxs_list = []

        for idx in range(len(self.learnware_list)):
            data_idx_list = np.where(select_result == idx)[0]
            if len(data_idx_list) > 0:
                if isinstance(raw_user_data, list):
                    pred_y = self.learnware_list[idx].predict([raw_user_data[i] for i in data_idx_list])
                else:
                    pred_y = self.learnware_list[idx].predict(raw_user_data[data_idx_list])

                if isinstance(pred_y, torch.Tensor):
                    pred_y = pred_y.detach().cpu().numpy()
                # elif isinstance(pred_y, tf.Tensor):
                #     pred_y = pred_y.numpy()

                if not isinstance(pred_y, np.ndarray):
                    raise TypeError("Model output must be np.ndarray or torch.Tensor")

                pred_y_list.append(pred_y)
                data_idxs_list.append(data_idx_list)

        if pred_y_list[0].ndim == 1:
            selector_pred_y = np.zeros(user_data.shape[0])
            for i in range(len(pred_y_list)):
                pred_y_list[i] = pred_y_list[i].reshape(-1)
        else:
            selector_pred_y = np.zeros((user_data.shape[0], pred_y_list[0].shape[1]))
            for i in range(len(pred_y_list)):
                pred_y_list[i] = pred_y_list[i].reshape(-1, pred_y_list[0].shape[1])

        for pred_y, data_idx_list in zip(pred_y_list, data_idxs_list):
            selector_pred_y[data_idx_list] = pred_y

        return selector_pred_y