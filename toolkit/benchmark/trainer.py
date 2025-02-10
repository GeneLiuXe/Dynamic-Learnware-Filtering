import joblib
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class OurXGBClassifier(XGBClassifier):
    def __init__(self, **kwargs):
        self._le_enc = LabelEncoder()
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        y_encoded = self._le_enc.fit_transform(y)
        super().fit(X, y_encoded, **kwargs)
        return self

    def predict(self, X, **kwargs):
        y_pred_encoded = super().predict(X, **kwargs)
        return self._le_enc.inverse_transform(y_pred_encoded)
    

class ModelTrainer:
    def __init__(self, task_type: str, model_type: str, param_idx: int):
        """Train a model for the specified task type

        Parameters
        ----------
        task_type : str
            Classification or regression
        model_type : str
            Ridge / LogisticRegression, Neural Network, or LightGBM
        param_idx : int
            The index of the hyperparameter set
        train_X : np.ndarray
            Training data
        train_y : np.ndarray
            Training labels
        save_path : _type_, optional
            Path to save the trained model, by default None
        """
        assert task_type in ["classification", "regression"]
        assert model_type in ["rf", "xgboost", "lightgbm"]
        assert param_idx in list(range(0, 5))

        self.task_type = task_type
        self.model_type = model_type
        self.param_idx = param_idx
        self.joblib_compress = 3

    def train_model(self, train_X: np.ndarray, train_y: np.ndarray, save_path: Optional[str] = None):
        if self.model_type == "rf":
            return self.train_sklearn_rf_model(train_X, train_y, save_path)
        elif self.model_type == "xgboost":
            return self.train_xgb_model(train_X, train_y, save_path)
        elif self.model_type == "lightgbm":
            return self.train_lgb_model(train_X, train_y, save_path)

    def train_sklearn_rf_model(self, X, y, save_path: Optional[str] = None):
        rf_params_list = [
            # [max_depth, min_samples_split, min_samples_leaf]
            [15, 2, 1],
            [8, 5, 2],
            [7, 4, 2],
            [10, 10, 5],
            [12, 8, 4],
        ]

        current_params = rf_params_list[self.param_idx]
        max_depth, min_samples_split, min_samples_leaf = current_params

        rf_params = {
            "n_estimators": 100,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
        }

        if self.task_type == "classification":
            rf_params["n_estimators"] = 200
            model = RandomForestClassifier(**rf_params)
        else:
            model = RandomForestRegressor(**rf_params)

        model.fit(X, y)
        if save_path is not None:
            joblib.dump(model, save_path, compress=self.joblib_compress)
        return model

    def train_xgb_model(self, X, y, save_path: Optional[str] = None):
        xgb_params_list = [
            [0.01, 6, 0.8, 0.8],
            [0.05, 8, 0.9, 0.7],
            [0.1, 10, 0.7, 0.9],
            [0.02, 12, 0.85, 0.75],
            [0.03, 15, 0.6, 0.6],
        ]

        objective, eval_metric = "reg:squarederror", "rmse"
        if self.task_type == "classification":
            if len(np.unique(y)) == 2:
                objective, eval_metric = "binary:logistic", "logloss"
            else:
                objective, eval_metric = "multi:softprob", "mlogloss"

        current_params = xgb_params_list[self.param_idx]
        learning_rate, max_depth, subsample, colsample_bytree = current_params

        xgb_params = {
            "tree_method": "hist",
            "device": "cuda",
            "objective": objective,
            "eval_metric": eval_metric,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": 1000,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "verbosity": 0,
            "use_label_encoder": False,
        }

        if self.task_type == "classification":
            xgb_params["n_estimators"] = 500
            if len(np.unique(y)) == 2:
                model = OurXGBClassifier(**xgb_params)
            else:
                model = OurXGBClassifier(**xgb_params, num_class=len(np.unique(y)))
        else:
            model = XGBRegressor(**xgb_params)

        model.fit(X, y)
        if save_path is not None:
            joblib.dump(model, save_path, compress=self.joblib_compress)
        return model

    def train_lgb_model(self, X, y, save_path: Optional[str] = None):
        lgb_params_list = [
            [0.01, 31, 5],
            [0.05, 50, 10],
            [0.1, 64, 15],
            [0.02, 40, 12],
            [0.03, 80, 20],
        ]
        objective, metric = "regression", "rmse"
        if self.task_type == "classification":
            if len(np.unique(y)) == 2:
                objective, metric = "binary", "binary_logloss"
            else:
                objective, metric = "multiclass", "multi_logloss"
        lgb_params = {
            "boosting_type": "gbdt",
            "objective": objective,
            "metric": metric,
            "learning_rate": lgb_params_list[self.param_idx][0],
            "num_leaves": lgb_params_list[self.param_idx][1],
            "max_depth": lgb_params_list[self.param_idx][2],
            "n_estimators": 1000,
            "boost_from_average": False,
            "verbose": -1,
        }
        if self.task_type == "classification":
            lgb_params["n_estimators"] = 500
            model = lgb.LGBMClassifier(**lgb_params)
        else:
            model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y)
        if save_path is not None:
            joblib.dump(model, save_path, compress=self.joblib_compress)
        return model