import argparse
from tqdm import tqdm

import toolkit
from toolkit import Benchmark
from toolkit.evaluator import Evaluator, Plotter
from toolkit.organizer import NoCheckOrganizer, CheckRKMEOrganizer, CheckModelCapOrganizer


def run_evaluation(dataset, learnware_num, user_task_num, not_regenerate_data):
    benchmark = Benchmark(dataset)
    if not not_regenerate_data:
        benchmark.regenerate_data()

    learnware_ids = benchmark.get_learnware_ids(first_num=learnware_num)
    user_ids = benchmark.get_user_ids(first_num=user_task_num)
    evaluation_method = "online_evaluation"
    PARAM_DICT = toolkit.config.PARAM_DICT[dataset]

    # Get learnware list
    total_learnware_list = []
    for idx in tqdm(learnware_ids):
        learnware = benchmark.get_idx_learnware(idx)
        total_learnware_list.append(learnware)

    # Our method - CheckModelCapOrganizer
    nearest_neighbor_search_K = PARAM_DICT["nearest_neighbor_search_K"]
    rkme_cover_tree_params = PARAM_DICT["rkme_cover_tree_params"]
    path_suffix=f"_knn{nearest_neighbor_search_K}_thres{rkme_cover_tree_params['loss_threshold']}_weight{rkme_cover_tree_params['dist_weight']}"
    our_organizer = CheckModelCapOrganizer(
        dataset,
        nearest_neighbor_search_K,
        rkme_cover_tree_params,
        auto_learnware_idx=False
    )
    evaluator = Evaluator(dataset, our_organizer, total_learnware_list, user_ids, path_suffix=path_suffix)
    getattr(evaluator, evaluation_method)()
    del evaluator
    del our_organizer

    # Baseline - CheckRKME
    rkme_thres = PARAM_DICT["rkme_thres"]
    path_suffix=f"_{rkme_thres}"
    rkme_organizer = CheckRKMEOrganizer(rkme_thres)
    evaluator = Evaluator(dataset, rkme_organizer, total_learnware_list, user_ids, path_suffix=path_suffix)
    getattr(evaluator, evaluation_method)()
    del evaluator
    del rkme_organizer
    
    # Baseline - NoFilter
    base_organizer = NoCheckOrganizer()
    evaluator = Evaluator(dataset, base_organizer, total_learnware_list, user_ids)
    getattr(evaluator, evaluation_method)()
    del evaluator
    del base_organizer
    del total_learnware_list

    # Plot
    PARAM_DICT["dataset"] = dataset
    PARAM_DICT["learnware_num"] = learnware_num
    PARAM_DICT["user_task_num"] = user_task_num
    plotter = Plotter(PARAM_DICT)
    plotter.plot_online_ratio_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="covertype")
    parser.add_argument("--not_regenerate_data", action="store_true")
    args = parser.parse_known_args()[0]
    dataset = args.dataset
    not_regenerate_data = args.not_regenerate_data

    learnware_num = 2000
    user_task_num = 100
    dataset_list = ["ppg", "air_quality", "m5", "covertype", "har70", "diabetes"]
    assert dataset in dataset_list

    run_evaluation(dataset, learnware_num, user_task_num, not_regenerate_data)