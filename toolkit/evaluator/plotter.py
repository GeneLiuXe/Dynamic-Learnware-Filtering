import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dco

from ..benchmark.utils import load_json_file
from ..config import RES_DIR, FIG_DIR, EVAL_DIR
from ..logger import get_module_logger

logger = get_module_logger("Plotter")


class Plotter:
    def __init__(self, param_dict: dict, start_point: int = 200, plot_delta: int = 300, total_len: int = 2000):
        dataset = param_dict["dataset"]
        learnware_num = param_dict["learnware_num"]
        user_task_num = param_dict["user_task_num"]
        knn_num = param_dict["nearest_neighbor_search_K"]
        loss_thres = param_dict["rkme_cover_tree_params"]["loss_threshold"]
        weight = param_dict["rkme_cover_tree_params"]["dist_weight"]
        rkme_thres = param_dict["rkme_thres"]

        method_filename = f"{dataset}_online_eval_CheckModelCapOrganizer_{learnware_num}_{user_task_num}_knn{knn_num}_thres{loss_thres}_weight{weight}.json"
        baseline_filename = f"{dataset}_online_eval_CheckRKMEOrganizer_{learnware_num}_{user_task_num}_{rkme_thres}.json"
        raw_filename = f"{dataset}_online_eval_NoCheckOrganizer_{learnware_num}_{user_task_num}.json"

        raw_data = {
            "ours": load_json_file(os.path.join(EVAL_DIR, dataset, method_filename)),
            "baseline": load_json_file(os.path.join(EVAL_DIR, dataset, baseline_filename)),
            "raw": load_json_file(os.path.join(EVAL_DIR, dataset, raw_filename)),
        }

        self.data_dict = {
            "rkme_task": {},
            "random_search": {},
            "best_search": {},
            "learnware_count": {},
        }
        self.raw_data_dict = dco(self.data_dict)
        min_len = min(len(raw_data["ours"]), len(raw_data["baseline"]), len(raw_data["raw"]))
        min_len = min(min_len, total_len)
        for key in self.data_dict.keys():
            for name in raw_data.keys():
                self.raw_data_dict[key][name] = [raw_data[name][i][key] for i in range(min_len)]
                self.data_dict[key][name] = [raw_data[name][i][key] for i in range(start_point-1, min_len, plot_delta)]
        self.x_labels = [i for i in range(start_point, min_len + 1, plot_delta)]

        self.dataset = dataset
        self.param = (knn_num, loss_thres, weight)
        self.figure_name = method_filename.split(".json")[0]
        os.makedirs(FIG_DIR, exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online_ratio"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online", dataset), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online", dataset, "PDF"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online", dataset, "JPG"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online_ratio", dataset), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online_ratio", dataset, "PDF"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "online_ratio", dataset, "JPG"), exist_ok=True)
        # self.analyze_method()

    def analyze_method(self):
        rkme_task_ratio = np.array(self.data_dict["rkme_task"]["ours"]) / np.array(self.data_dict["rkme_task"]["raw"])
        random_search_ratio = np.array(self.data_dict["random_search"]["ours"]) / np.array(self.data_dict["random_search"]["raw"])
        best_search_ratio = np.array(self.data_dict["best_search"]["ours"]) / np.array(self.data_dict["best_search"]["raw"])
        learnware_count_ratio = np.array(self.data_dict["learnware_count"]["ours"]) / np.array(self.data_dict["learnware_count"]["raw"])

        rkme_win = np.sum(rkme_task_ratio <= 1) / len(rkme_task_ratio)
        random_win = np.sum(random_search_ratio <= 1) / len(random_search_ratio)
        rkme_mean = np.mean(rkme_task_ratio)
        random_mean = np.mean(random_search_ratio)
        best_mean = np.mean(best_search_ratio)
        count_mean = np.mean(learnware_count_ratio)

        logger.info(f"dataset: {self.dataset}, param: {self.param}, total length: {len(rkme_task_ratio)}")
        logger.info(f"RKME Win Ratio: {round(rkme_win, 4)}, Random Search Win Ratio: {round(random_win, 4)}")
        logger.info(f"RKME Task Ratio Mean: {round(rkme_mean, 4)}, Random Search Ratio Mean: {round(random_mean, 4)}, Best Search Ratio Mean: {round(best_mean, 4)}, Learnware Count Ratio Mean: {round(count_mean, 4)}")

        return rkme_task_ratio, random_search_ratio, best_search_ratio, learnware_count_ratio

    def plot_online_ratio_evaluation(self):
        fig, axes = plt.subplots(1, figsize=(8, 5))
        styles = [
            {"color": "#2992ed", "linestyle": "-", "marker": "o", "markersize": 5},
            {"color": "#1eae63", "linestyle": "-", "marker": "s", "markersize": 5},
            {"color": "#f26462", "linestyle": "--", "marker": "^", "markersize": 5},
            {"color": "#c82423", "linestyle": "-.", "marker": "d", "markersize": 5},
        ]
        methods = ["rkme_task", "random_search", "best_search", "learnware_count"]
        labels = ["rkme_task", "random_search", "best_search", "learnware_count"]

        ratio_dict = {}
        ratio_dict["rkme_task"] = np.array(self.data_dict["rkme_task"]["ours"]) / np.array(self.data_dict["rkme_task"]["raw"])
        ratio_dict["random_search"] = np.array(self.data_dict["random_search"]["ours"]) / np.array(self.data_dict["random_search"]["raw"])
        ratio_dict["best_search"] = np.array(self.data_dict["best_search"]["ours"]) / np.array(self.data_dict["best_search"]["raw"])
        ratio_dict["learnware_count"] = np.array(self.data_dict["learnware_count"]["ours"]) / np.array(self.data_dict["learnware_count"]["raw"])

        plt.sca(axes)
        for method, label, style in zip(methods, labels, styles):
            plt.plot(self.x_labels, np.array(ratio_dict[method]) * 100, **style, label=f"{label}: {ratio_dict[label].mean()*100:.2f}%")
        plt.title(f"{self.dataset}", fontsize=21)
        plt.ylabel("Ratio (%)", fontsize=20)
        plt.xticks([200, 500, 800, 1100, 1400, 1700, 2000], [200, 500, 800, "1.1k", "1.4k", "1.7k", "2k"])
        plt.xlim(100, 2100)
        plt.xlabel("Number of uploaded learnwares", fontsize=20)
        
        axes.spines["top"].set_linewidth(0.5)
        axes.spines["right"].set_linewidth(0.5)
        axes.spines["left"].set_linewidth(0.5)
        axes.spines["bottom"].set_linewidth(0.5)

        plt.legend(fontsize=18, loc="best")
        plt.tick_params(axis='both', labelsize=13)
        plt.grid(linestyle = '--', linewidth=0.5, color='gray', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "online_ratio", self.dataset, "JPG", f"online_eval_ratio_{self.figure_name}.jpg"), bbox_inches="tight", dpi=700)
        plt.savefig(os.path.join(FIG_DIR, "online_ratio", self.dataset, "PDF", f"online_eval_ratio_{self.figure_name}.pdf"), bbox_inches="tight", dpi=700)
        plt.close()


class PlotterMulti:
    def __init__(self, param_dict_list):
        start_point, plot_delta, total_len = 200, 300, 2000
        self.figure_name = f"{start_point}_{plot_delta}_{total_len}"
        self.param_dict_list = param_dict_list
        self.x_labels = [i for i in range(start_point, total_len + 1, plot_delta)]

        for i in range(len(param_dict_list)):
            plotter = Plotter(param_dict_list[i], start_point, plot_delta, total_len)
            rkme_task_ratio, random_search_ratio, best_search_ratio, learnware_count_ratio = plotter.analyze_method()
            param_dict_list[i]["rkme_task_ratio"] = rkme_task_ratio
            param_dict_list[i]["random_search_ratio"] = random_search_ratio
            param_dict_list[i]["best_search_ratio"] = best_search_ratio
            param_dict_list[i]["learnware_count_ratio"] = learnware_count_ratio
            self.figure_name += f"_{(param_dict_list[i]['dataset'], param_dict_list[i]['nearest_neighbor_search_K'], param_dict_list[i]['rkme_cover_tree_params']['loss_threshold'], param_dict_list[i]['rkme_cover_tree_params']['dist_weight'])}"

        self.dataset2name = {
            "ppg": "PPG-DaLiA",
            "m5_large": "M5",
            "har70": "HAR70+",
            "diabetes": "Diabetes",
            "covtype": "Covertype",
            "air_quality": "Air-Quality",
        }

        os.makedirs(FIG_DIR, exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "final"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "final", "JPG"), exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "final", "PDF"), exist_ok=True)

    def plot_multi_ratio_evaluation(self):
        plt.rc('axes', axisbelow=True)
        fig, axes = plt.subplots(1, len(self.param_dict_list), figsize=(29, 4.5))
        styles = [
            {"color": "#2992ed", "linestyle": "--", "marker": "^", "markersize": 10}, # navy, #2992ed, #2878B5
            {"color": "#f26462", "linestyle": "-.", "marker": "d", "markersize": 10}, # magenta, #f26462, #c82423
        ]
        methods = ["best_search", "learnware_count"]
        labels = ["best_search", "learnware_count"]

        for i, param_dict in enumerate(self.param_dict_list):
            ratio_dict = {}
            ratio_dict["best_search"] = np.array(param_dict["best_search_ratio"])
            ratio_dict["learnware_count"] = np.array(param_dict["learnware_count_ratio"])

            plt.sca(axes[i])
            for method, label, style in zip(methods, labels, styles):
                plt.plot(self.x_labels, np.array(ratio_dict[method]) * 100, **style, label=label)
            plt.title(f"{self.dataset2name[param_dict['dataset']]}", fontsize=26)
            if i == 0:
                plt.ylabel("Ratio (%)", fontsize=26)
            plt.xticks([200, 500, 800, 1100, 1400, 1700, 2000], [200, 500, 800, "1.1k", "1.4k", "1.7k", "2k"])
            plt.xlim(100, 2100)
            plt.xlabel("Learnware Count", fontsize=26)
            
            axes[i].spines["top"].set_linewidth(0.5)
            axes[i].spines["right"].set_linewidth(0.5)
            axes[i].spines["left"].set_linewidth(0.5)
            axes[i].spines["bottom"].set_linewidth(0.5)
            
            plt.tick_params(axis='both', labelsize=17.5)
            plt.grid(linestyle = '--', linewidth=0.5, color='gray', alpha=0.3)
            plt.tight_layout()

        fig.legend(["Ratio of Best Identification Performance (Ours / NoFilter)", "Ratio of System Size (Ours / NoFilter)"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.18), fontsize=26)
        plt.savefig(os.path.join(FIG_DIR, "final", "JPG", f"online_eval_ratio_{self.figure_name}.jpg"), bbox_inches="tight", dpi=700)
        plt.savefig(os.path.join(FIG_DIR, "final", "PDF", f"online_eval_ratio_{self.figure_name}.pdf"), bbox_inches="tight", dpi=700)
        plt.close()