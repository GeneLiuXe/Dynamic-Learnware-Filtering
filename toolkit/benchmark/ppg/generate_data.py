import torch
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from .config import *


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin-1")
    
    signal = pd.DataFrame(data["signal"])
    ACC = pd.DataFrame(signal["chest"].ACC)
    ACC = ACC.iloc[::175, :]
    ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
    ACC.reset_index(drop = True, inplace=True)
    
    ECG = pd.DataFrame(signal["chest"].ECG)
    ECG = ECG.iloc[::175, :]
    ECG.reset_index(drop = True, inplace=True)
    
    Resp = pd.DataFrame(signal["chest"].Resp)
    Resp = Resp.iloc[::175, :]
    Resp.columns = ["Resp"]
    Resp.reset_index(drop = True, inplace=True)
    
    chest = pd.concat([ACC], sort=False)
    chest["Resp"] = Resp
    chest["ECG"] = ECG
    chest.reset_index(drop=True, inplace=True)
    chest = chest.add_prefix('chest_')
    
    ACC = pd.DataFrame(signal["wrist"].ACC)
    ACC = ACC.iloc[::8, :]
    ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
    ACC.reset_index(drop = True, inplace=True)
    
    EDA = pd.DataFrame(signal["wrist"].EDA)
    EDA.columns = ["EDA"]
    
    BVP = pd.DataFrame(signal["wrist"].BVP)
    BVP = BVP.iloc[::16, :]
    BVP.columns = ["BVP"]
    BVP.reset_index(drop = True, inplace=True)
    
    TEMP = pd.DataFrame(signal["wrist"].TEMP)
    TEMP.columns = ["TEMP"]
    
    wrist = pd.concat([ACC], sort=False)
    wrist["BVP"] = BVP
    wrist["TEMP"] = TEMP
    wrist.reset_index(drop = True, inplace=True)
    wrist = wrist.add_prefix('wrist_')
    
    signals = chest.join(wrist)
    for k,v in data["questionnaire"].items() :
        signals[k] = v
    
    rpeaks = data['rpeaks']
    counted_rpeaks = []
    index = 0 # index of rpeak element
    time = 175 # time portion
    count = 0 # number of rpeaks

    while(index < len(rpeaks)):
        rpeak = rpeaks[index]

        if(rpeak > time): # Rpeak appears after the time portion
            counted_rpeaks.append(count)
            count = 0
            time += 175

        else:
            count += 1
            index += 1
    # The rpeaks will probably end before the time portion so we need to fill the last portions with 0
    if(len(counted_rpeaks) < np.size(signals, axis = 0)):
        while(len(counted_rpeaks) < np.size(signals, axis = 0)):
            counted_rpeaks.append(0)
    peaks = pd.DataFrame(counted_rpeaks)
    peaks.columns = ["Rpeaks"]
    signals = signals.join(peaks)
    
    activity = pd.DataFrame(data["activity"]).astype(int)
    activity.columns = ["Activity"]
    signals = signals.join(activity)
    
    label = pd.DataFrame(data["label"])
    
    label = pd.DataFrame(np.repeat(label.values,8,axis=0))
    label.columns = ["Label"]
    if(np.size(label, axis = 0) < np.size(activity, axis = 0)):
        mean = label.mean()
        while(np.size(label, axis = 0) < np.size(activity, axis = 0)):
            label = label.append(mean, ignore_index=True)
    
    signals = signals.join(label)
    signals['Subject'] = data["subject"]
    
    return signals


def fill_nan(data):
    # fill np.nan
    X_nan = np.isnan(data)
    if X_nan.max() == 1:
        for col in range(data.shape[1]):
            col_mean = np.nanmean(data[:, col])
            data[:, col] = np.where(X_nan[:, col], col_mean, data[:, col])
    return data


def split_save_data(idx, raw_df):
    train_data_path = os.path.join(PROCESSED_DATA_DIR, f"train_{idx}.npy")
    test_data_path = os.path.join(PROCESSED_DATA_DIR, f"test_{idx}.npy")
    
    train_X = raw_df.drop(columns=["Label"]).to_numpy()
    train_y = raw_df["Label"].to_numpy().reshape(-1, 1)
    
    train_X = fill_nan(train_X)
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    data = np.concatenate((train_X, train_y), axis=1)
    set_seed(0)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=0, shuffle=True)
    
    np.save(train_data_path, train_data)
    np.save(test_data_path, test_data)