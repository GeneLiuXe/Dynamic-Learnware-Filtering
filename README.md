# Dynamic Learnware Filtering for Efficient Learnware Identification and System Slimming

## Introduction

Welcome to the source code for the paper "Dynamic Learnware Filtering for Efficient Learnware Identification and System Slimming".

In this work, we have developed a learnware dock system encompassing over ten thousand models with different types, spanning various real-world scenarios. We conduct comparisons against contenders, validating the effectiveness and efficiency of our proposed approach.

## Reproduction

### Setup

To reproduce the experiments, you need to first install the necessary dependencies with the following command:
```bash
python -m pip install -r requirements.txt
```

### Data Preparation

The datasets below should be manually downloaded and save the files in the following directory: `toolkit/benchmark/{dataset_name}/data/raw`.

We provide the download link for each dataset as follows:
- `Diabetes`: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- `HAR70+`: https://archive.ics.uci.edu/dataset/780/har70
- `Air-Quality`: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- `PPG-DaLiA`: https://archive.ics.uci.edu/dataset/495/ppg+dalia
- `M5`: https://www.kaggle.com/competitions/m5-forecasting-accuracy

### Code Execution

To conduct the experiments, please execute the command below:
```bash
python main.py --dataset <dataset_name>
```

The `dataset_name` corresponds to the dataset list:
```
["diabetes", "har70", "covertype", "air_quality", "ppg", "m5"]
```

Examples:
- For `Diabetes`, execute: `python main.py --dataset diabetes`
- For `PPG-DaLiA`, execute: `python main.py --dataset ppg`