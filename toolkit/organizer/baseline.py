import numpy as np
from typing import List, Dict
from sortedcontainers import SortedSet
from learnware.learnware import Learnware
from learnware.specification import RKMETableSpecification

from .base import BaseOrganizer
from .. import Benchmark
from ..logger import get_module_logger

logger = get_module_logger("BaselineOrganizer")


class NoCheckOrganizer(BaseOrganizer):
    def __init__(self):
        self.learnware_list = []
        self.learnware_ids = []
        super(NoCheckOrganizer, self).__init__(type=self.__class__.__name__)

    def add_learnware(self, learnware: Learnware) -> bool:
        self.learnware_list.append(learnware)
        self.learnware_ids.append(learnware.id)
        return True

    def get_learnware_list(self) -> List[Learnware]:
        return self.learnware_list

    def get_learnware_count(self) -> int:
        return len(self.learnware_list)
    
    def get_learnware_ids(self) -> List[int]:
        return self.learnware_ids


class CheckRKMEOrganizer(BaseOrganizer):
    def __init__(self, rkme_thres: float = 1e-5):
        self.rkme_thres = rkme_thres
        self.RKME_list = []
        self.learnware_list = []
        self.learnware_ids = []
        super(CheckRKMEOrganizer, self).__init__(type=self.__class__.__name__)

    def add_learnware(self, learnware: Learnware) -> bool:
        new_RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")
        for RKME in self.RKME_list:
            if new_RKME.dist(RKME) <= self.rkme_thres:
                return False
        self.learnware_list.append(learnware)
        self.learnware_ids.append(learnware.id)
        self.RKME_list.append(new_RKME)
        return True

    def get_learnware_list(self) -> List[Learnware]:
        return self.learnware_list

    def get_learnware_count(self) -> int:
        assert len(self.learnware_list) == len(self.RKME_list)
        return len(self.learnware_list)
    
    def get_learnware_ids(self) -> List[int]:
        return self.learnware_ids


class NaiveCheckModelCapOrganizer(BaseOrganizer):
    def __init__(self, dataset: str, loss_threshold: float):
        self.learnware_list: Dict[int, Learnware] = {}
        self.top_RKME_ids_map: Dict[int, SortedSet] = {}
        self.rkme_list: Dict[int, (RKMETableSpecification, np.ndarray)] = {}
        self.loss_func = Benchmark(dataset).score
        self.loss_threshold = loss_threshold
        super(NaiveCheckModelCapOrganizer, self).__init__(type=self.__class__.__name__)

    def _does_perform_well(self, learnware: Learnware, rkme: RKMETableSpecification, rkme_label: np.ndarray) -> float:
        test_x = rkme.get_z()
        pred_y = learnware.predict(test_x)
        loss = self.loss_func(rkme_label, pred_y, sample_weight=rkme.get_beta())
        return loss <= self.loss_threshold
    
    def _does_perform_similar(self, learnware1: Learnware, learnware2: Learnware, rkme: RKMETableSpecification) -> float:
        test_x = rkme.get_z()
        pred_y1 = learnware1.predict(test_x)
        pred_y2 = learnware2.predict(test_x)
        loss = self.loss_func(pred_y1, pred_y2, sample_weight=rkme.get_beta())
        return loss <= self.loss_threshold
    
    def get_task_set(self, learnware: Learnware) -> List[int]:
        result_set = SortedSet([])
        for rkme_id, (rkme, rkme_label) in self.rkme_list.items():
            if self._does_perform_well(learnware, rkme, rkme_label):
                result_set.add(rkme_id)
        return result_set

    def update_task_set(self, learnware: Learnware) -> None:
        RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")
        test_x = RKME.get_z()
        label_y = learnware.predict(test_x)
        self.rkme_list[learnware.id] = (RKME, label_y)

        for learnware_id, task_learnware in self.learnware_list.items():
            pred_y = task_learnware.predict(test_x)
            loss = self.loss_func(label_y, pred_y, sample_weight=RKME.get_beta())
            if loss <= self.loss_threshold:
                self.top_RKME_ids_map[learnware_id].add(learnware.id)

    def add_learnware(self, learnware: Learnware) -> bool:
        result_set = self.get_task_set(learnware)
        candidate_ids = []
        for learnware_id, task_set in self.top_RKME_ids_map.items():
            if result_set.issubset(task_set):
                candidate_ids.append(learnware_id)
        
        rkme_list = [learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")]
        for rkme_id in result_set:
            rkme_list.append(self.rkme_list[rkme_id][0])
        result_set.add(learnware.id)

        for learnware_id in candidate_ids:
            flag = True
            for rkme in rkme_list:
                if not self._does_perform_similar(learnware, self.learnware_list[learnware_id], rkme):
                    flag = False
                    break
            if flag:
                logger.info(f"New learnware {learnware.id} is covered by existing learnwares, total learnware count: {self.get_learnware_count()}")
                return False
            
        self.update_task_set(learnware)
        remove_ids = []
        for learnware_id, task_set in self.top_RKME_ids_map.items():
            if task_set.issubset(result_set):
                rkme_list = []
                for rkme_id in task_set:
                    rkme_list.append(self.rkme_list[rkme_id][0])
                
                flag = True
                for rkme in rkme_list:
                    if not self._does_perform_similar(self.learnware_list[learnware_id], learnware, rkme):
                        flag = False
                        break
                if flag:
                    remove_ids.append(learnware_id)
            
        for learnware_id in remove_ids:
            self.top_RKME_ids_map.pop(learnware_id)
            self.learnware_list.pop(learnware_id)
            logger.info(f"Learnware {learnware_id} is removed from the organizer due to the new learnware {learnware.id}, total learnware count: {self.get_learnware_count()}")

        self.learnware_list[learnware.id] = learnware
        self.top_RKME_ids_map[learnware.id] = result_set
        logger.info(f"Add learnware {learnware.id} to the organizer, total learnware count: {self.get_learnware_count()}")
        return True

    def get_learnware_list(self) -> List[Learnware]:
        return list(self.learnware_list.values())

    def get_learnware_count(self) -> int:
        assert len(self.learnware_list) == len(self.top_RKME_ids_map)
        return len(self.learnware_list)
    
    def get_learnware_ids(self) -> List[int]:
        return list(self.learnware_list.keys())