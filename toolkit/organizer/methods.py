from __future__ import annotations

import math
import heapq
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from sortedcontainers import SortedList, SortedSet
from typing import List, Optional, Tuple, Dict, Callable
from learnware.specification import RKMETableSpecification
from learnware.learnware import Learnware

from .base import BaseOrganizer
from .. import Benchmark
from ..config import MAX_LEVEL, MAX_RADIUS
from ..logger import get_module_logger

logger = get_module_logger("CheckModelCapOrganizer")


class LinkedChildrenMap:
    def __init__(self):
        self._dict = {} # level: list for child_idx
        self._size_dict = {} # level: size of descendants
        self._sorted_keys = SortedList(key=lambda x: -x) # sorted by level, descending

    def add(self, level, child_idx):
        """Add a new child."""
        if level not in self._dict:
            self._sorted_keys.add(level)
            self._dict[level] = []
            self._size_dict[level] = 0
        self._dict[level].append(child_idx)

    def increase_size(self, level, size):
        """Increase the size of descendants."""
        if level not in self._size_dict:
            raise ValueError(f"Level {level} does not exist.")
        self._size_dict[level] += size

    def get_descendant_size(self, level):
        """Retrieve the size of descendants."""
        return self._size_dict.get(level, 0)

    def get_idxes_by_level(self, level):
        """Retrieve the child idxes associated with the level."""
        return self._dict.get(level, [])

    def get_level_by_pos(self, pos):
        """Retrieve the level by the position."""
        if pos < 0 or pos >= len(self._sorted_keys):
            return None
        return self._sorted_keys[pos]

    def __len__(self):
        """Return the number of different levels."""
        return len(self._sorted_keys)

    def __str__(self):
        """Return the string representation of the map."""
        return str(self._dict)


@dataclass
class RKMECoverTreeNode:
    idx: int
    level: Optional[int] = None # The level of the cover tree node
    children_idx_map: LinkedChildrenMap = field(default_factory=LinkedChildrenMap)
    parent_idx: Optional[int] = None
    cover_radius: float = 0.0 # The radius of the cover tree node
    descendant_size: int = 1 # The number of descendants, including itself

    _iter_pos: int = -1
    _cumulative_size: int = 0

    def add_child(self, child: RKMECoverTreeNode):
        self.children_idx_map.add(child.level, child.idx)
        self.children_idx_map.increase_size(child.level, child.descendant_size)
        child.parent_idx = self.idx
        self.descendant_size += child.descendant_size

    def increase_descendant_size(self, level, size):
        self.descendant_size += size
        self.children_idx_map.increase_size(level, size)

    def is_leaf(self):
        return len(self.children_idx_map) == 0

    def is_root(self):
        return self.parent_idx is None

    def init_children_iter(self):
        self._iter_pos = -1
        self._cumulative_size = 0

    def get_next_level(self):
        next_level = self.children_idx_map.get_level_by_pos(self._iter_pos + 1)
        return next_level
    
    def get_distinctive_descendant_size(self, level):
        # Get the number of descendants lower than the level, including itself
        next_level = self.get_next_level()
        if next_level is not None and next_level > level:
            raise ValueError("The next level is greater than the current level.")
        return self.descendant_size - self._cumulative_size

    def get_next_children_idxes(self, level: int) -> List[int]:
        next_level = self.children_idx_map.get_level_by_pos(self._iter_pos + 1)
        if next_level is None or next_level < level:
            return []
        if next_level > level:
            logger.warning(f"_iter_pos: {self._iter_pos}, idx: {self.idx}, Next level: {next_level}, Current level: {level}, Sorted_keys: {list(self.children_idx_map._sorted_keys)}")
            raise ValueError("The next level is greater than the current level.")
        self._iter_pos += 1
        self._cumulative_size += self.children_idx_map.get_descendant_size(next_level)
        return self.children_idx_map.get_idxes_by_level(next_level)

    def get_children_idxes_by_level(self, level: int) -> List[int]:
        return self.children_idx_map.get_idxes_by_level(level)
    
    def get_all_children_idxes(self) -> List[int]:
        all_children_idxes = []
        for level, idx_list in self.children_idx_map._dict.items():
            all_children_idxes.extend(idx_list)
        return all_children_idxes

    @property
    def max_cover_radius(self):
        if self.level is None:
            raise ValueError("The level of the node is null.")
        if self.level >= MAX_LEVEL:
            return MAX_RADIUS
        return pow(2, self.level)


class RKMECoverTree:
    def __init__(
        self,
        loss_func: Optional[Callable] = None,
        loss_threshold: float = 10.0,
        dist_weight: float = 60,
        **kwargs
    ):
        self.root_idx: Optional[int] = None
        self.nodes: List[RKMECoverTreeNode] = []

        self.RKMEs: List[RKMETableSpecification] = []
        self.RKME_labels: List[np.ndarray] = []

        # The top learnware ids for each RKME
        self.top_learnware_ids_map: List[SortedSet] = []

        # Loss function
        self.loss_func = loss_func

        # Min level
        self.min_level = MAX_LEVEL

        # Hyperparameters
        self.params = {}
        self.params["loss_threshold"] = loss_threshold
        self.params["dist_weight"] = dist_weight

    def insert(self, learnware: Learnware, top_learnware_ids: List[int] = None, top_RKME_ids: List[int] = None) -> int:
        RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")
        RKME_label = learnware.predict(RKME.get_z())

        self.RKMEs.append(RKME)
        self.RKME_labels.append(RKME_label)
        if top_learnware_ids is None:
            self.top_learnware_ids_map.append(SortedSet([learnware.id]))
        else:
            assert learnware.id in top_learnware_ids
            self.top_learnware_ids_map.append(SortedSet(top_learnware_ids))

        # Update the top learnware ids for each RKME
        if top_RKME_ids is not None:
            for RKME_id in top_RKME_ids:
                self.top_learnware_ids_map[RKME_id].add(learnware.id)

        node_idx = len(self.nodes)
        new_node = RKMECoverTreeNode(idx=node_idx)
        self.nodes.append(new_node)

        def _assign_parent(new_node, curr_R):
            parent_idx, min_dist = None, None
            for node_idx, dist in curr_R:
                level = self.nodes[node_idx].level
                if node_idx == self.root_idx or dist <= pow(2, level):
                    if min_dist is None or dist < min_dist:
                        parent_idx, min_dist = node_idx, dist
            
            if min_dist == 0:
                raise ValueError(f"The minimum distance is zero for node {new_node.idx}, stop assigning parent.")

            if parent_idx is None:
                return False

            new_node.level = math.ceil(math.log2(min_dist)) - 1
            self.min_level = min(self.min_level, new_node.level)

            if parent_idx == self.root_idx:
                if self.nodes[self.root_idx].level == MAX_LEVEL:
                    self.nodes[self.root_idx].level = new_node.level + 1
                else:
                    self.nodes[self.root_idx].level = max(self.nodes[self.root_idx].level, new_node.level + 1)
            
            curr_node = self.nodes[parent_idx]
            curr_node.add_child(new_node)

            # Update the cover radius of the ancestors
            curr_node.cover_radius = max(curr_node.cover_radius, min_dist)
            parent_idx = curr_node.parent_idx
            son_level = curr_node.level
            while parent_idx is not None:
                # print(f"Parent idx: {parent_idx}")
                parent_node = self.nodes[parent_idx]
                parent_node.increase_descendant_size(son_level, 1)
                parent_node.cover_radius = max(parent_node.cover_radius, self._dist(parent_node, new_node))
                son_level = parent_node.level
                parent_idx = parent_node.parent_idx

            return True

        def _insert(i, new_node, old_R):
            if i >= self.min_level:
                new_R = []
                for node_idx, dist in old_R:
                    if dist <= pow(2, i+1):
                        new_R.append((node_idx, dist))

                    children_idxes = self.nodes[node_idx].get_next_children_idxes(i)
                    for child_idx in children_idxes:
                        dist2 = self._dist(self.nodes[child_idx], new_node)
                        if dist2 <= pow(2, i+1):
                            self.nodes[child_idx].init_children_iter()
                            new_R.append((child_idx, dist2))

                if len(new_R) == 0:
                    flag = _assign_parent(new_node, old_R)
                    return flag
            
                next_i = -MAX_LEVEL
                for node_idx, dist in new_R:
                    next_level = self.nodes[node_idx].get_next_level()
                    if next_level is not None:
                        next_i = max(next_i, next_level)

                flag = _insert(next_i, new_node, new_R)
                if flag:
                    return True
                else:
                    return _assign_parent(new_node, old_R)
            else:
                return _assign_parent(new_node, old_R)

        if self.root_idx is None:
            new_node.level = MAX_LEVEL
            self.root_idx = new_node.idx
        else:
            max_level = self.nodes[self.root_idx].level
            i = max_level - 1
            self.nodes[self.root_idx].init_children_iter()
            old_R = [(self.root_idx, self._dist(self.nodes[self.root_idx], new_node))]

            flag = _insert(i, new_node, old_R)
            if not flag:
                raise ValueError("Failed to insert the new node.")
        
        return node_idx

    def get_top_RKME_ids_for_learnware_dfs(self, learnware: Learnware, node_idx: int = None) -> List[int]:
        top_RKME_ids = []

        if self.root_idx is None:
            return top_RKME_ids

        if node_idx is None:
            node_idx = self.root_idx
        test_x = self.RKMEs[node_idx].get_z()
        test_y = self.RKME_labels[node_idx]
        pred_y = learnware.predict(test_x)
        loss = self.loss_func(test_y, pred_y, sample_weight=self.RKMEs[node_idx].get_beta())
        if loss <= self.params["loss_threshold"]:
            top_RKME_ids.append(node_idx)
        
        cur_node = self.nodes[node_idx]
        # logger.info(f"Node {node_idx}, loss: {loss}, cover_radius: {cur_node.cover_radius}, loss - dist_weight * cover_radius: {loss - self.params['dist_weight'] * cur_node.cover_radius}")
        if cur_node.is_root() or (not cur_node.is_leaf() and loss - self.params["dist_weight"] * cur_node.cover_radius <= self.params["loss_threshold"]):
            cur_node.init_children_iter()
            level = cur_node.get_next_level()
            while level is not None:
                children_idxes = cur_node.get_next_children_idxes(level)
                for child_idx in children_idxes:
                    top_RKME_ids.extend(self.get_top_RKME_ids_for_learnware_dfs(learnware, child_idx))
                level = cur_node.get_next_level()
        
        return top_RKME_ids

    def get_top_RKME_ids_for_learnware_bfs(self, learnware: Learnware) -> List[int]:
        top_RKME_ids = []
        if self.root_idx is None:
            return top_RKME_ids
        queue = [self.root_idx] + self.nodes[self.root_idx].get_all_children_idxes()

        while len(queue) > 0:
            test_x_list = [self.RKMEs[idx].get_z() for idx in queue]
            test_y_list = [self.RKME_labels[idx] for idx in queue]
            beta_list = [self.RKMEs[idx].get_beta() for idx in queue]
            pred_y_list = self._batch_predict(learnware, test_x_list)

            new_queue = []
            loss_list = [self.loss_func(test_y, pred_y, sample_weight=beta) for test_y, pred_y, beta in zip(test_y_list, pred_y_list, beta_list)]
            for idx, loss in zip(queue, loss_list):
                if loss <= self.params["loss_threshold"]:
                    top_RKME_ids.append(idx)
                
                cur_node = self.nodes[idx]
                if cur_node.is_root() or cur_node.is_leaf(): continue
                if loss - self.params["dist_weight"] * cur_node.cover_radius <= self.params["loss_threshold"]:
                    new_queue.extend(cur_node.get_all_children_idxes())
            queue = new_queue
        
        return top_RKME_ids

    def nearest_neighbor_search(self, query_RKME: RKMETableSpecification, K: int) -> Tuple[List[int], List[int]]:
        """Search for the nearest neighbors of the RKME in the cover tree.

        Parameters
        ----------
        query_RKME : RKMETableSpecification
            The RKME to be searched.
        K : int
            The number of nearest neighbors.
        Returns
        -------
        Tuple[List[int], List[int]]
            The RKME indexes and distances of the nearest neighbors, descending by distance.
        """
        res_heap = []

        def _add(heap, idx, dist, capacity):
            # Use negative distance to implement max heap
            if len(heap) < capacity:
                heapq.heappush(heap, (-dist, idx))
            else:
                heapq.heappushpop(heap, (-dist, idx))
        
        def _peek(heap, capacity):
            # Get the maximum distance
            if len(heap) < capacity:
                return None
            return -heap[0][0]
        
        def _pop(heap):
            dist, idx = heapq.heappop(heap)
            return idx, -dist

        def _get_lambda_k_distance(R_list, capacity, level):
            curr_heap = []
            for node_idx, dist in R_list:
                _add(curr_heap, node_idx, dist, capacity)
            reorder_list = []
            while len(curr_heap) > 0:
                reorder_list.append(_pop(curr_heap))
            size = 0
            for i in range(len(reorder_list)-1, -1, -1):
                idx, dist = reorder_list[i]
                size += self.nodes[idx].get_distinctive_descendant_size(level)
                if size >= capacity:
                    return dist
            return None

        if self.root_idx is not None:
            max_level = self.nodes[self.root_idx].level
            i = max_level - 1
            self.nodes[self.root_idx].init_children_iter()
            old_R = [(self.root_idx, self._dist_with_RKME(self.nodes[self.root_idx], query_RKME))]
            _add(res_heap, old_R[0][0], old_R[0][1], K)
            
            while i >= self.min_level:
                length = len(old_R)
                for pos in range(length):
                    node_idx, dist = old_R[pos]
                    children_idxes = self.nodes[node_idx].get_next_children_idxes(i)
                    for child_idx in children_idxes:
                        dist = self._dist_with_RKME(self.nodes[child_idx], query_RKME)
                        self.nodes[child_idx].init_children_iter()
                        old_R.append((child_idx, dist))
                        _add(res_heap, old_R[-1][0], old_R[-1][1], K)
                
                new_R = []
                k_distance = _peek(res_heap, K)
                lambda_distance = _get_lambda_k_distance(old_R, K, i)

                for node_idx, dist in old_R:
                    next_level = self.nodes[node_idx].get_next_level()
                    if next_level is not None:
                        if k_distance is None or dist - self.nodes[node_idx].cover_radius < k_distance:
                            if lambda_distance is None or dist - self.nodes[node_idx].cover_radius < lambda_distance + math.pow(2, i+1):
                                new_R.append((node_idx, dist))

                next_i = -MAX_LEVEL
                for node_idx, dist in new_R:
                    next_i = max(next_i, self.nodes[node_idx].get_next_level())
                i = next_i
                old_R = new_R
        
        idxs, dists = [], []
        while len(res_heap) > 0:
            idx, dist = _pop(res_heap)
            idxs.append(idx)
            dists.append(dist)
        return idxs, dists

    def get_counter_learnware_ids(self, RKME_ids: List[int]) -> Counter:
        counter = Counter()
        for RKME_id in RKME_ids:
            counter.update(self.top_learnware_ids_map[RKME_id])
        return counter

    def get_union_learnware_ids(self, RKME_ids: List[int]) -> List[int]:
        curr_set = None
        for RKME_id in RKME_ids:
            if curr_set is None:
                curr_set = self.top_learnware_ids_map[RKME_id]
            else:
                curr_set = curr_set | self.top_learnware_ids_map[RKME_id]
        
        if curr_set is None:
            return []
        return list(curr_set)

    def get_intersection_learnware_ids(self, RKME_ids: List[int]) -> List[int]:
        curr_set = None
        for RKME_id in RKME_ids:
            if curr_set is None:
                curr_set = self.top_learnware_ids_map[RKME_id]
            else:
                curr_set = curr_set & self.top_learnware_ids_map[RKME_id]
        
        if curr_set is None:
            return []
        return list(curr_set)

    def are_learnwares_similar(self, learnware1: Learnware, learnware2: Learnware, RKME_ids: List[int]) -> bool:
        """Check if two learnwares are similar.

        Parameters
        ----------
        learnware1 : Learnware
            The first learnware.
        learnware2 : Learnware
            The second learnware.
        RKME_ids : List[int]
            The RKME indices to be considered.
        Returns
        -------
        bool
            True if the two learnwares are similar, False otherwise.
        """
        test_x_list = [self.RKMEs[RKME_id].get_z() for RKME_id in RKME_ids]
        beta_list = [self.RKMEs[RKME_id].get_beta() for RKME_id in RKME_ids]
        pred_y1_list = self._batch_predict(learnware1, test_x_list)
        pred_y2_list = self._batch_predict(learnware2, test_x_list)

        for pred_y1, pred_y2, beta in zip(pred_y1_list, pred_y2_list, beta_list):
            loss = self.loss_func(pred_y1, pred_y2, sample_weight=beta)
            if loss > self.params["loss_threshold"]:
                return False
        return True

    def does_learnware_perform_well(self, learnware: Learnware, RKME: RKMETableSpecification, RKME_labels: np.ndarray) -> bool:
        test_x = RKME.get_z()
        pred_y = learnware.predict(test_x)
        loss = self.loss_func(RKME_labels, pred_y, sample_weight=RKME.get_beta())
        return loss <= self.params["loss_threshold"]

    def remove_learnware_id(self, learnware_id: int, RKME_ids: List[int]):
        for RKME_id in RKME_ids:
            self.top_learnware_ids_map[RKME_id].remove(learnware_id)

    def _batch_predict(self, learnware: Learnware, test_x_list: List[np.ndarray]) -> List[np.ndarray]:
        concatenated_data = np.vstack(test_x_list)
        output = learnware.predict(concatenated_data)
        split_idxes = np.cumsum([len(test_x) for test_x in test_x_list])
        return np.split(output, split_idxes[:-1], axis=0)

    def _dist(self, Node1: RKMECoverTreeNode, Node2: RKMECoverTreeNode) -> float:
        return math.sqrt(self.RKMEs[Node1.idx].dist(self.RKMEs[Node2.idx]))

    def _dist_with_RKME(self, Node: RKMECoverTreeNode, RKME: RKMETableSpecification) -> float:
        return math.sqrt(self.RKMEs[Node.idx].dist(RKME))
    
    def print_structure(self, node_idx: int = None, iter: int = 0):
        if node_idx is None:
            node_idx = self.root_idx
            print("Each node format: (idx, level, size, cover_radius), omit cover_radius if zero.")

        node = self.nodes[node_idx]
        if node.cover_radius == 0.0:
            print("|\t" * iter + f"|-[{node_idx, node.level, node.descendant_size}]")
        else:
            print("|\t" * iter + f"|-\033[90m[{node_idx, node.level, node.descendant_size, round(node.cover_radius, 2)}], {node.children_idx_map._size_dict}\033[0m")

        node.init_children_iter()
        level = node.get_next_level()
        while level is not None:
            children_idxes = node.get_next_children_idxes(level)
            for child_idx in children_idxes:
                self.print_structure(child_idx, iter + 1)
            level = node.get_next_level()


class CheckModelCapOrganizer(BaseOrganizer):
    # The organizer for dynamically checking the model capacity
    def __init__(
            self,
            dataset: str,
            nearest_neighbor_search_K: int = 5,
            rkme_cover_tree_params: Dict = {},
            auto_learnware_idx: bool = False
        ):
        self.dataset = dataset
        self.learnware_cnt = -1
        self.auto_learnware_idx = auto_learnware_idx
        self.nearest_neighbor_search_K = nearest_neighbor_search_K

        rkme_cover_tree_params["loss_func"] = Benchmark(dataset).score
        self.RKME_cover_tree = RKMECoverTree(**rkme_cover_tree_params)

        # Key is the learnware id
        self.learnware_list: Dict[int, Learnware] = {}
        self.top_RKME_ids_map: Dict[int, List[int]] = {}
        super(CheckModelCapOrganizer, self).__init__(type=self.__class__.__name__)

    def is_learnware_covered_by_others(self, learnware: Learnware) -> bool:
        # Check if the learnware is covered by existing learnwares
        learnware_RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")

        top_RKME_ids = self.RKME_cover_tree.get_top_RKME_ids_for_learnware_bfs(learnware)
        candidate_learnware_ids = self.RKME_cover_tree.get_intersection_learnware_ids(top_RKME_ids)

        if len(candidate_learnware_ids) == 0:
            if len(top_RKME_ids) != 0:
                return False, top_RKME_ids
            else:
                similar_RKME_ids, distances = self.RKME_cover_tree.nearest_neighbor_search(learnware_RKME, self.nearest_neighbor_search_K)
                candidate_learnware_ids = self.RKME_cover_tree.get_union_learnware_ids(similar_RKME_ids)

        # Filter out the candidate learnwares that not perform well on the new RKME
        filtered_candidate_learnware_ids = []
        RKME_label = learnware.predict(learnware_RKME.get_z())
        for idx in candidate_learnware_ids:
            if self.RKME_cover_tree.does_learnware_perform_well(self.learnware_list[idx], learnware_RKME, RKME_label):
                if len(top_RKME_ids) == 0:
                    return True, top_RKME_ids
                else:
                    filtered_candidate_learnware_ids.append(idx)
        
        for idx in filtered_candidate_learnware_ids:
            if self.RKME_cover_tree.are_learnwares_similar(learnware, self.learnware_list[idx], top_RKME_ids):
                return True, top_RKME_ids

        return False, top_RKME_ids

    def check_existing_learnwares(self, new_learnware: Learnware, top_RKME_ids: List[int]):
        # Check if existing learnwares are covered by the new learnware
        learnware_id_counter = self.RKME_cover_tree.get_counter_learnware_ids(top_RKME_ids)
        for learnware_id, count in learnware_id_counter.items():
            other_top_RKME_ids = self.top_RKME_ids_map[learnware_id]
            if count == len(other_top_RKME_ids):
                if self.RKME_cover_tree.are_learnwares_similar(new_learnware, self.learnware_list[learnware_id], other_top_RKME_ids):
                    # Remove the existing learnware since it is covered by the new learnware
                    del self.learnware_list[learnware_id]
                    del self.top_RKME_ids_map[learnware_id]
                    self.RKME_cover_tree.remove_learnware_id(learnware_id, other_top_RKME_ids)
                    logger.info(f"Learnware {learnware_id} is removed from the organizer due to the new learnware {new_learnware.id}, total learnware count: {self.get_learnware_count()}")

    def add_learnware(self, learnware: Learnware) -> bool:
        """Add a learnware to the organizer.

        Parameters
        ----------
        learnware : Learnware
            The learnware to be added

        Returns
        -------
        bool
            True if the learnware is successfully added, False otherwise
        """
        # Assign new id to the learnware
        if self.auto_learnware_idx:
            self.learnware_cnt += 1
            learnware.id = self.learnware_cnt

        # Check if the learnware is covered by existing learnwares
        flag, top_RKME_ids = self.is_learnware_covered_by_others(learnware)
        if flag:
            logger.info(f"New learnware {learnware.id} is covered by existing learnwares, total learnware count: {self.get_learnware_count()}")
            return False

        # Add the learnware to the list
        self.learnware_list[learnware.id] = learnware
        self.top_RKME_ids_map[learnware.id] = top_RKME_ids
        logger.info(f"Add learnware {learnware.id} to the organizer, total learnware count: {self.get_learnware_count()}")

        # Check if existing learnwares are covered by this learnware
        self.check_existing_learnwares(learnware, top_RKME_ids)

        # Insert the new RKME to the cover tree
        top_learnware_ids = [learnware.id]
        new_RKME = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")
        similar_RKME_ids, distances = self.RKME_cover_tree.nearest_neighbor_search(new_RKME, self.nearest_neighbor_search_K)
        if len(distances) == 0 or distances[-1] > 0.0:
            potential_learnware_ids = self.RKME_cover_tree.get_union_learnware_ids(similar_RKME_ids)
            RKME_label = learnware.predict(new_RKME.get_z())
            for learnware_id in potential_learnware_ids:
                if self.RKME_cover_tree.does_learnware_perform_well(self.learnware_list[learnware_id], new_RKME, RKME_label):
                    top_learnware_ids.append(learnware_id)
            new_RKME_idx = self.RKME_cover_tree.insert(learnware, top_learnware_ids, top_RKME_ids)
            assert new_RKME_idx is not None

            # Update the top RKME ids for related learnware
            for learnware_id in top_learnware_ids:
                self.top_RKME_ids_map[learnware_id].append(new_RKME_idx)
        
        return True
    
    def get_learnware_list(self) -> List[Learnware]:
        return list(self.learnware_list.values())

    def get_learnware_count(self) -> int:
        assert len(self.learnware_list) == len(self.top_RKME_ids_map)
        return len(self.learnware_list)
    
    def get_learnware_ids(self) -> List[int]:
        return list(self.learnware_list.keys())