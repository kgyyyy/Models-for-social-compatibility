# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module evaluates the forecasted trajectories against the ground truth."""

import math
from typing import Dict, List, Optional

import numpy as np

import argoverse
from argoverse.map_representation.map_api import ArgoverseMap

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


# 250226修改：去掉与city_name相关的所有信息
def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_displacement_errors_and_miss_rate(
    crash_flags,
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.
    The Brier Score is defined here:
        Brier, G. W. Verification of forecasts expressed in terms of probability. Monthly weather review, 1950.
        https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR, brier-minADE, brier-minFDE
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade, brier_min_ade = [], [], []
    min_fde, prob_min_fde, brier_min_fde = [], [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        # 250312修改：增加crash_flags参数，根据该参数划分horizon
        if crash_flags[k]:
            valid_future = ~((v[:, 0] == 0) & (v[:, 1] == 0))
            eval_horizon = np.sum(valid_future)
        else:
            eval_horizon = horizon

        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:eval_horizon], v[:eval_horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        curr_min_ade = get_ade(pruned_trajectories[min_idx][:eval_horizon], v[:eval_horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - pruned_probabilities[min_idx]))
            prob_min_ade.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_ade
            )
            brier_min_ade.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_ade)
            prob_min_fde.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_fde
            )
            brier_min_fde.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_fde)

    # 250308修改：key名增加步长以区分
    metric_results[f"{horizon}_minADE"] = sum(min_ade) / len(min_ade)
    metric_results[f"{horizon}_minFDE"] = sum(min_fde) / len(min_fde)
    metric_results[f"{horizon}_MR"] = sum(n_misses) / len(n_misses)
    if forecasted_probabilities is not None:
        metric_results[f"{horizon}_p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
        metric_results[f"{horizon}_p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
        metric_results[f"{horizon}_p-MR"] = sum(prob_n_misses) / len(prob_n_misses)
        metric_results[f"{horizon}_brier-minADE"] = sum(brier_min_ade) / len(brier_min_ade)
        metric_results[f"{horizon}_brier-minFDE"] = sum(brier_min_fde) / len(brier_min_fde)
    return metric_results


def get_drivable_area_compliance(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    max_n_guesses: int,
) -> float:
    """Compute drivable area compliance metric.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        max_n_guesses: Maximum number of guesses allowed.

    Returns:
        Mean drivable area compliance

    """
    avm = ArgoverseMap()

    dac_score = []

    for seq_id, trajectories in forecasted_trajectories.items():
        num_dac_trajectories = 0
        n_guesses = min(max_n_guesses, len(trajectories))
        for trajectory in trajectories[:n_guesses]:
            raster_layer = avm.get_raster_layer_points_boolean(trajectory, "driveable_area")
            if np.sum(raster_layer) == raster_layer.shape[0]:
                num_dac_trajectories += 1

        dac_score.append(num_dac_trajectories / n_guesses)

    return sum(dac_score) / len(dac_score)


def compute_forecasting_metrics(
    crash_flags,
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_n_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,

) -> Dict[str, float]:
    """Compute all the forecasting metrics.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_n_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Miss threshold
        forecasted_probabilities: Normalized Probabilities associated with each of the forecasted trajectories.

     Returns:
        metric_results: Dictionary containing values for all metrics.
    """
    metric_results = get_displacement_errors_and_miss_rate(
        crash_flags,
        forecasted_trajectories,
        gt_trajectories,
        max_n_guesses,
        horizon,
        miss_threshold,
        forecasted_probabilities,
    )
    # 250226修改：没有地图文件记录drivable_area  放弃该指标
    # metric_results["DAC"] = get_drivable_area_compliance(forecasted_trajectories, max_n_guesses)

    print("------------------------------------------------")
    print(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}")

    return metric_results


# 250225修改：格式整理便于打印
def format_metric_results(metric_results):
    """
    将 metric_results 转化为可打印输出的字符串。

    Args:
        metric_results: 包含评估指标的字典。

    Returns:
        metric_result_str: 格式化后的字符串，包含键名和对应的值。
    """
    metric_result_str = "Evaluation Metrics:\n"
    metric_result_str += "----------------------------------------\n"

    for key, value in metric_results.items():
        metric_result_str += f"{key}: {value:.4f}\n"  # 保留 4 位小数

    metric_result_str += "----------------------------------------\n"
    return metric_result_str


def social_aware_evaluation(pred_dicts):
    """
    评估预测轨迹，调用 eval_forecasting.py 中的评估指标，并返回结果字典。

    """
    # 初始化用于评估的输入数据结构
    forecasted_trajectories = {}
    gt_trajectories = {}
    forecasted_probabilities = {}
    crash_flags = {}
    # 遍历 pred_dicts，提取预测轨迹和真实轨迹
    for idx, pred_dict in enumerate(pred_dicts):
        # 提取预测轨迹和真实轨迹
        forecasted_trajectories[idx] = pred_dict['pred_trajs']  # 形状 (6, 100, 2)
        gt_trajectories[idx] = pred_dict['gt_trajs']  # 形状 (100, 2)

        # 提取预测轨迹的概率
        forecasted_probabilities[idx] = pred_dict['pred_scores']  # 形状 (6,)

        # 提取if_crash标签
        crash_flags[idx] = pred_dict['if_crash']

    # 250308修改：分别对40步长和100步长内都计算一次指标
    # 250312修改：增加对碰撞数据的适应，具体为如发生碰撞，以碰撞点截取horizon，无碰撞gt则计算40
    # 调用 compute_forecasting_metrics 计算评估指标
    metric_results = get_displacement_errors_and_miss_rate(
        crash_flags,
        forecasted_trajectories=forecasted_trajectories,
        gt_trajectories=gt_trajectories,
        max_guesses=6,
        horizon=40,  # 预测时间步长为 40
        miss_threshold=3,
        forecasted_probabilities=forecasted_probabilities,
    )
    # metric_results2 = compute_forecasting_metrics(
    #     forecasted_trajectories=forecasted_trajectories,
    #     gt_trajectories=gt_trajectories,
    #     max_n_guesses=6,  # 假设最多允许 6 个预测轨迹
    #     horizon=100,  # 预测时间步长为 100
    #     miss_threshold=3,  # 未命中阈值设为 3
    #     forecasted_probabilities=forecasted_probabilities,
    # )
    # metric_results = {**metric_results1, **metric_results2}
    metric_result_str = format_metric_results(metric_results)

    # 返回结果字典
    return metric_results, metric_result_str
