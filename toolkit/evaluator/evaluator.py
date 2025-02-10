import os
import time
import numpy as np
from tqdm import tqdm
from typing import List

from learnware.learnware import Learnware
from learnware.market.easy import EasyStatSearcher

from .reuse import OurJobSelectorReuser
from ..benchmark import Benchmark
from ..organizer import BaseOrganizer
from ..logger import get_module_logger
from ..benchmark.utils import save_json_file, load_json_file
from ..config import RES_DIR, EVAL_DIR

logger = get_module_logger("Evaluator")


class Evaluator:
    def __init__(self, dataset: str, organizer: BaseOrganizer, learnware_list: List[Learnware], user_ids: List[int] = None, path_suffix: str = ""):
        self.dataset = dataset
        self.benchmark = Benchmark(dataset)
        self.learnware_list = learnware_list
        self.organizer = organizer
        self.path_suffix = path_suffix
        os.makedirs(EVAL_DIR, exist_ok=True)
        os.makedirs(os.path.join(EVAL_DIR, dataset), exist_ok=True)

        if user_ids is None:
            self.user_ids = self.benchmark.get_user_ids()
        else:
            self.user_ids = user_ids
        self.user_RKMEs = {}
        for user_id in self.user_ids:
            self.user_RKMEs[user_id] = self.benchmark.get_user_rkme(user_id)

        max_idx = max([learnware.id for learnware in learnware_list])
        self.err_mat = np.zeros((max_idx + 1, len(self.user_ids)))
        for learnware in learnware_list:
            idx = learnware.id
            self.err_mat[idx] = self.benchmark.get_learnware_performance_array(idx, self.user_ids)

        self.rkme_dist_mat = np.zeros((max_idx + 1, len(self.user_ids)))
        for learnware in learnware_list:
            idx = learnware.id
            self.rkme_dist_mat[idx] = self.benchmark.get_learnware_rkme_dist_array(idx, self.user_ids)

    def _rkme_task(self, learnware_ids: List[int], **kwargs):
        model_ids = np.array(learnware_ids)[np.argmin(self.rkme_dist_mat[learnware_ids], axis=0)]
        return np.mean(self.err_mat[model_ids, np.arange(len(self.user_ids))])

    def _rkme_task_raw(self, learnware_list: List[Learnware], **kwargs):
        final_loss = 0
        for i, user_id in enumerate(self.user_ids):
            pos, dist = None, None
            user_rkme = self.user_RKMEs[user_id]
            for j, learnware in enumerate(learnware_list):
                RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")
                dist = RKME.dist(user_rkme)
                if pos is None or dist < min_dist:
                    pos, min_dist = j, dist
            # learnware = learnware_list[pos]
            # _, _, val_x, val_y = self.benchmark.get_idx_data(user_id)
            # pred_y = learnware.predict(val_x)
            final_loss += self.err_mat[learnware_list[pos].id, i]
        return final_loss / len(self.user_ids)

    def _rkme_instance(self, learnware_list: List[Learnware], search_method: str = "auto", **kwargs):
        def _search_auto(user_rkme) -> List[Learnware]:
            weight_cutoff = 0.99
            max_learnware_num = 5
            searcher = EasyStatSearcher(organizer=None)
            searcher.stat_spec_type = user_rkme.type

            learnware_num = len(learnware_list)
            max_search_num = min(learnware_num, max_learnware_num)
            weight, _ = searcher._calculate_rkme_spec_mixture_weight(learnware_list, user_rkme)
            sorted_idx_list = sorted(range(learnware_num), key=lambda k: weight[k], reverse=True)

            weight_sum = 0
            mathch_idx_list = []
            for i in range(max_search_num):
                weight_sum += weight[sorted_idx_list[i]]
                if len(mathch_idx_list) == 0 or weight_sum <= weight_cutoff:
                    mathch_idx_list.append(sorted_idx_list[i])
                else:
                    break
            return [learnware_list[i] for i in mathch_idx_list]
        
        def _search_greedy(user_rkme) -> List[Learnware]:
            decay_rate = 0.90
            max_learnware_num = 3
            searcher = EasyStatSearcher(organizer=None)
            searcher.stat_spec_type = user_rkme.type

            learnware_num = len(learnware_list)
            max_search_num = min(learnware_num, max_learnware_num)

            _, _, mixture_learnware_list = searcher._search_by_rkme_spec_mixture_greedy(learnware_list, user_rkme, max_search_num, decay_rate)
            return mixture_learnware_list

        final_loss = 0
        for user_id in tqdm(self.user_ids):
            rkme = self.user_RKMEs[user_id]
            if search_method == "auto":
                matched_learnwares = _search_auto(rkme)
            elif search_method == "greedy":
                matched_learnwares = _search_greedy(rkme)
            else:
                raise ValueError(f"Invalid search method: {search_method}")
            reuser = OurJobSelectorReuser(matched_learnwares, use_herding=False)

            _, _, val_x, val_y = self.benchmark.get_idx_data(user_id)
            pred_y = reuser.predict(val_x)
            final_loss += float(self.benchmark.score(val_y, pred_y))

        return final_loss / len(self.user_ids)

    def _random_search(self, learnware_ids: List[int], **kwargs):
        return np.sum(self.err_mat[list(learnware_ids)]) / (len(learnware_ids) * len(self.user_ids))

    def _best_search(self, learnware_ids: List[int], **kwargs):
        tmp_err_mat = self.err_mat[list(learnware_ids)].T
        return float(np.mean(np.min(tmp_err_mat, axis=1)))
    
    def get_organizer(self) -> BaseOrganizer:
        return self.organizer

    def online_evaluation(self, evaluate_methods: List[str] = ["rkme_task", "random_search", "best_search"], save_path: str = None):
        """Online evaluation for all learnwares

        Parameters
        ----------
        evaluate_methods : List[str], optional
            The whole method list is ["rkme_instance", "rkme_task", "random_search", "best_search"]
        save_path : str, optional
            The path to save the evaluation results
        """
        if save_path is None:
            res_name = f"{self.dataset}_online_eval_{self.organizer.type}_{len(self.learnware_list)}_{len(self.user_ids)}{self.path_suffix}.json"
            save_path = os.path.join(EVAL_DIR, self.dataset, res_name)

        if os.path.exists(save_path):
            data_list = load_json_file(save_path)
            if len(data_list) >= len(self.learnware_list):
                logger.info(f"Skip online evaluation for {self.organizer.type} since {save_path} exists.")
                return

        online_score_list = []
        for iter, learnware in enumerate(self.learnware_list):
            self.organizer.add_learnware(learnware)

            learnware_list = self.organizer.get_learnware_list()
            learnware_ids = self.organizer.get_learnware_ids()
            learnware_count = len(learnware_list)

            evaluate_score = {}
            for method in evaluate_methods:
                method_name = f"_{method}"
                if hasattr(self, method_name):
                    evaluate_score[method] = getattr(self, method_name)(learnware_list=learnware_list, learnware_ids=learnware_ids)
            evaluate_score["learnware_count"] = learnware_count
            result_str = f"Learnware count: {learnware_count}, " + ", ".join([f"{method}: {round(score, 2)}" for method, score in evaluate_score.items()])
            logger.info(f"{self.organizer.type} online evaluation (Iteration{iter}):\n {result_str}")
            
            online_score_list.append(evaluate_score)
            save_json_file(online_score_list, save_path)

    def online_record_learnware_ids(self, save_path: str = None):
        if save_path is None:
            res_name = f"{self.dataset}_online_record_learnware_ids_{self.organizer.type}_{len(self.learnware_list)}_{len(self.user_ids)}{self.path_suffix}.json"
            save_path = os.path.join(EVAL_DIR, self.dataset, res_name)

        if os.path.exists(save_path):
            data_list = load_json_file(save_path)
            if len(data_list) >= len(self.learnware_list):
                logger.info(f"Skip online evaluation for {self.organizer.type} since {save_path} exists.")
                return

        online_learnware_list = []
        for iter, learnware in enumerate(self.learnware_list):
            self.organizer.add_learnware(learnware)

            learnware_ids = self.organizer.get_learnware_ids()
            online_learnware_list.append(learnware_ids)
            save_json_file(online_learnware_list, save_path)
    
    def online_record_time(self, save_path: str = None):
        if save_path is None:
            res_name = f"{self.dataset}_online_record_time_{self.organizer.type}_{len(self.learnware_list)}_{len(self.user_ids)}{self.path_suffix}.json"
            save_path = os.path.join(EVAL_DIR, self.dataset, res_name)

        if os.path.exists(save_path):
            data_list = load_json_file(save_path)
            if len(data_list) >= len(self.learnware_list):
                logger.info(f"Skip online evaluation for {self.organizer.type} since {save_path} exists.")
                return

        online_score_list = []
        for iter, learnware in enumerate(self.learnware_list):
            start_time = time.time()
            self.organizer.add_learnware(learnware)
            end_time = time.time()

            data_dict = {}
            learnware_ids = self.organizer.get_learnware_ids()
            data_dict["best_search"] = self._best_search(learnware_ids)
            data_dict["learnware_ids"] = learnware_ids
            data_dict["time"] = end_time - start_time
            online_score_list.append(data_dict)
            save_json_file(online_score_list, save_path)

    def batch_evaluation(self, evaluate_methods: List[str] = ["rkme_task", "random_search", "best_search"]):
        """Batch evaluation for all learnwares

        Parameters
        ----------
        evaluate_methods : List[str], optional
            The whole method list is ["rkme_instance", "rkme_task", "random_search", "best_search"]
        """
        for learnware in self.learnware_list:
            self.organizer.add_learnware(learnware)

        learnware_list = self.organizer.get_learnware_list()
        learnware_ids = self.organizer.get_learnware_ids()
        learnware_count = len(learnware_list)

        evaluate_score = {}
        for method in evaluate_methods:
            method_name = f"_{method}"
            if hasattr(self, method_name):
                evaluate_score[method] = getattr(self, method_name)(learnware_list=learnware_list, learnware_ids=learnware_ids)
        evaluate_score["learnware_count"] = learnware_count

        result_str = f"Learnware count: {learnware_count}, " + ", ".join([f"{method}: {round(score, 2)}" for method, score in evaluate_score.items()])
        logger.info(f"{self.organizer.type} batch evaluation:\n {result_str}\n")