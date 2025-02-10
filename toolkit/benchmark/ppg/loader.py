import os
from sklearn.metrics import root_mean_squared_error
from learnware.specification import generate_stat_spec, RKMETableSpecification

from .config import *
from .generate_data import *
from ..trainer import ModelTrainer
from ..utils import get_sampled_learnware_params, get_sampled_user_task_params, save_json_file, load_json_file
from ...logger import get_module_logger

logger = get_module_logger("PPGBenchmark")


class DataLoader:
    def __init__(self):
        os.makedirs(ROOT_PATH, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(USER_RKME_DIR, exist_ok=True)
        os.makedirs(PARAM_DIR, exist_ok=True)

        learnware_params_path = os.path.join(PARAM_DIR, "learnware_params.json")
        if not os.path.exists(learnware_params_path):
            self.learnware_params = get_sampled_learnware_params(SHOP_NUM, LEARNWARE_NUM)
            save_json_file(self.learnware_params, learnware_params_path)
        else:
            self.learnware_params = load_json_file(learnware_params_path)

        user_task_params_path = os.path.join(PARAM_DIR, "user_task_params.json")
        if not os.path.exists(user_task_params_path):
            self.user_task_params = get_sampled_user_task_params(SHOP_NUM, USER_TASK_NUM)
            save_json_file(self.user_task_params, user_task_params_path)
        else:
            self.user_task_params = load_json_file(user_task_params_path)

    def get_learnware_ids(self):
        return list(range(LEARNWARE_NUM))

    def get_user_ids(self):
        return (np.array(range(USER_TASK_NUM)) + SEP + LEARNWARE_NUM).tolist()

    def get_learnware_params(self, idx):
        assert idx < LEARNWARE_NUM
        return self.learnware_params[idx]
    
    def get_user_task_params(self, idx):
        assert idx >= LEARNWARE_NUM + SEP
        idx -= LEARNWARE_NUM + SEP
        return self.user_task_params[idx]

    def _convert_idx2param(self, idx):
        if idx < LEARNWARE_NUM + SEP:
            param = self.learnware_params[idx]
        else:
            idx -= LEARNWARE_NUM + SEP
            param = self.user_task_params[idx]
        return param

    def get_shop_data(self, idx):
        train_data_path = os.path.join(PROCESSED_DATA_DIR, f"train_{idx}.npy")
        test_data_path = os.path.join(PROCESSED_DATA_DIR, f"test_{idx}.npy")

        train_data = np.load(train_data_path)
        test_data = np.load(test_data_path)

        train_X = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_X = test_data[:, :-1]
        test_y = test_data[:, -1]

        return train_X, train_y, test_X, test_y

    def get_idx_data(self, idx):
        param = self._convert_idx2param(idx)
        comb = param["shop_ids"]
        train_xs, train_ys, test_xs, test_ys = [], [], [], []
        for cls in comb:
            train_x, train_y, test_x, test_y = self.get_shop_data(cls)
            train_xs.append(train_x)
            train_ys.append(train_y)
            test_xs.append(test_x)
            test_ys.append(test_y)

        train_xs = np.concatenate(train_xs, axis=0)
        train_ys = np.concatenate(train_ys, axis=0)
        test_xs = np.concatenate(test_xs, axis=0)
        test_ys = np.concatenate(test_ys, axis=0)

        return train_xs, train_ys, test_xs, test_ys

    def get_user_rkme(self, user_idx):
        if user_idx < LEARNWARE_NUM + SEP:
            raise ValueError("User index should be greater than or equal to LEARNWARE_NUM")
        
        rkme_path = os.path.join(USER_RKME_DIR, f"{user_idx}.json")
        if not os.path.exists(rkme_path):
            train_x, train_y, val_x, val_y = self.get_idx_data(user_idx)
            rkme = generate_stat_spec("table", val_x)
            rkme.save(rkme_path)
        else:
            rkme = RKMETableSpecification()
            rkme.load(rkme_path)
        return rkme

    def score(self, real_y, pred_y, sample_weight=None, multioutput="raw_values"):
        return root_mean_squared_error(real_y, pred_y, sample_weight=sample_weight, multioutput=multioutput)

    def _train_model(self, idx):
        model_path = os.path.join(MODEL_DIR, f"{idx}.out")
        param = self._convert_idx2param(idx)
        train_X, train_y, _, _ = self.get_idx_data(idx)
        model_type = param["model_type"]
        param_idx = param["param_idx"]

        model_trainer = ModelTrainer(TASK_TYPE, model_type, param_idx)
        trained_model = model_trainer.train_model(train_X, train_y, model_path)

        if model_type == "mlp":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
            with torch.no_grad():
                predy = trained_model(train_X)
                predy = predy.detach().cpu().numpy()
        else:
            predy = trained_model.predict(train_X)

        score = self.score(train_y, predy)
        logger.info(f"Model {idx} is trained, model_type = {model_type}, param_idx = {param_idx}, training loss = {score}")

    def get_model_path(self, idx):
        model_path = os.path.join(MODEL_DIR, f"{idx}.out")
        if os.path.exists(model_path):
            return model_path
        else:
            self._train_model(idx)

            if os.path.exists(model_path):
                return model_path
            else:
                raise ValueError(f"Model path {model_path} is not found")

    def regenerate_data(self):
        raw_path = os.path.join(RAW_DATA_DIR, "data", "PPG_FieldStudy")
            
        # Data Preprocessing
        idx = 0
        subjects = list(range(1, 16)) # [1, 15]
        for subject in subjects:
            raw_pkl_path = os.path.join(raw_path, f"S{subject}", f"S{subject}.pkl")
            raw_df = load_data(raw_pkl_path)
            
            # Remove columns
            remove = ["Subject", "WEIGHT", "Gender", "AGE", "HEIGHT", "SKIN", "SPORT"]
            features = [column for column in list(raw_df.columns) if column not in remove]
            raw_df = raw_df[features]
            
            # Split based on activity
            act_list = [-1, 0, 4, 8]
            for i in range(len(act_list) - 1): # [0, 8]
                act_l = act_list[i]
                act_r = act_list[i + 1]
                df = raw_df[(raw_df["Activity"] > act_l) & (raw_df["Activity"] <= act_r)]
                if len(df) > 0:
                    df = df.drop(columns=["Activity"])
                    split_save_data(idx, df)
                    idx += 1

    def retrain_models(self):
        idx_list = self.get_learnware_ids()
        for idx in idx_list:
            self._train_model(idx)