import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.colors as mcolors  # 新增导入
from matplotlib.collections import LineCollection
import pandas as pd


def calculate_theta(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    改进的航向角计算函数，处理小角度情况
    Args:
        vx: x方向速度数组
        vy: y方向速度数组
    Returns:
        平滑处理的航向角数组（弧度制）
    """
    theta = np.zeros_like(vx)
    theta[0] = np.arctan2(vy[0], vx[0])  # 初始值

    for i in range(1, len(vx)):
        # 当前时刻原始计算值
        current_theta = np.arctan2(vy[i], vx[i])

        # 当速度很小时（小于0.01m/s视为静止），沿用上一时刻角度
        if np.sqrt(vx[i] ** 2 + vy[i] ** 2) < 0.01:
            theta[i] = theta[i - 1]
        else:
            theta[i] = current_theta

    return theta


def calculate_yaw_rate(theta: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    计算横摆角速度（一阶差分法）
    Args:
        theta: 航向角数组（弧度）
        dt: 时间间隔（秒），默认0.1s对应10Hz数据
    Returns:
        yaw_rate: 横摆角速度数组（弧度/秒）
    """
    yaw_rate = np.zeros_like(theta)
    yaw_rate[1:] = np.diff(theta) / dt  # 前向差分

    # 处理角度跳变（如从-pi到pi的突变）
    jumps = np.abs(np.diff(theta)) > np.pi
    yaw_rate[1:][jumps] = (np.diff(theta)[jumps] - 2 * np.pi * np.sign(np.diff(theta)[jumps])) / dt

    return yaw_rate


def extract_data_frames_from_raw(data: np.ndarray, start: int, end: int):
    """辅助函数：从车辆数据中提取指定帧范围的6个特征列表"""
    return [
        data[start:end, 0],  # x
        data[start:end, 1],  # y
        data[start:end, 2],  # vx
        data[start:end, 3],  # vy
        data[start:end, 4],  # v
        data[start:end, 5]   # theta
    ]


def extract_raw_scenario_data(raw_scenario: np.ndarray) -> Dict:
    """
    Args:
        raw_scenario: 形状为(4,201,10)的numpy数组，包含4辆车201帧的10维数据

    Returns:
        Dict: 结构化数据字典，包含：
            - hist_ego_data: ego车0-100帧历史数据 [6列表]
            - gt_ego_data: ego车101-200帧未来数据 [6列表]
            - hist_oth_data: 其他车历史数据 [3车×6列表]
            - oth_data: 其他车未来数据 [3车×6列表]
    """
    # 提取ego车数据
    ego_data = raw_scenario[0]
    hist_ego_data = extract_data_frames_from_raw(ego_data, 0, 101)
    gt_ego_data = extract_data_frames_from_raw(ego_data, 101, 201)

    # 提取其他车辆数据 (risk, oth1, oth2)
    oth_vehicles = raw_scenario[1:]
    hist_oth_data = [extract_data_frames_from_raw(v, 0, 101) for v in oth_vehicles]
    oth_data = [extract_data_frames_from_raw(v, 101, 201) for v in oth_vehicles]

    return {
        'hist_ego_data': hist_ego_data,
        'gt_ego_data': gt_ego_data,
        'hist_oth_data': hist_oth_data,
        'oth_data': oth_data
    }


def extract_prediction_data(scenario: Dict, prefix: str = '') -> Dict:
    """
    从scenario中提取预测数据，可自定义key前缀

    Args:
        scenario: 包含预测数据的字典，需有以下键:
            - 'pred_trajs': ndarray(6, 100, 2)  # 6种预测轨迹, 100帧, (x,y)
            - 'pred_velo': ndarray(6, 100, 2)   # 对应速度 (vx,vy)
            - 'pred_scores': ndarray(6)         # 每种预测的得分
        prefix: 输出字典key的前缀，例如传入'social_aware_'则生成'SAMSH_pred_ego_data'

    Returns:
        包含预测数据的字典，key格式为{f"{prefix}pred_ego_data"}和{f"{prefix}pred_ego_yaw_rate"}
    """
    # 获取得分最高的预测索引
    pred_idx = np.argmax(scenario['pred_scores'])

    # 提取轨迹和速度
    traj = scenario['pred_trajs'][pred_idx]  # (100,2)
    velo = scenario['pred_velo'][pred_idx]  # (100,2)
    vx, vy = velo[:, 0], velo[:, 1]

    # 计算运动参数
    v = np.sqrt(vx ** 2 + vy ** 2)
    theta = calculate_theta(vx, vy)
    yaw_rate = calculate_yaw_rate(theta)

    # 构建返回字典（使用前缀）
    return {
        f"{prefix}pred_ego_data": [
            traj[:, 0],  # x
            traj[:, 1],  # y
            vx,  # vx
            vy,  # vy
            v,  # v
            theta  # theta
        ],
        f"{prefix}pred_ego_yaw_rate": yaw_rate
    }


def process_single_scenario(scenario_id: str,
                     social_aware_scenario: Dict,
                     ap_scenario: Dict,
                     raw_scenario: Dict) -> Dict:
    """处理单个场景的所有数据"""
    # 提取原始数据
    raw_data = extract_raw_scenario_data(raw_scenario)

    # 处理预测数据
    social_aware_pred = extract_prediction_data(social_aware_scenario, 'social_aware_')
    ap_pred = extract_prediction_data(ap_scenario, 'AP_')

    # 构建统一数据格式
    scenario_dict = {**raw_data, **social_aware_pred, **ap_pred}
    scenario_dict.update({'scenario_id': scenario_id})

    return scenario_dict


# ***************************************主函数******************************************


def results_process_and_file_rearrange():
    # 配置路径
    social_aware_result_path = "mtr_social_aware_train/result_social_aware_1/eval/eval_with_train/epoch_200/result.pkl"
    ap_result_path = 'mtr_baseline_train/result_baseline_2/eval/eval_with_train/epoch_200/result.pkl'

    social_aware_result_path = "mtr_social_aware_test/social_aware_test_crash_scenario/eval/epoch_3/default/result.pkl"
    ap_result_path = "mtr_social_aware_test/baseline_test_crash_scenario/eval/epoch_3/default/result.pkl"

    # social_aware_result_path = "mtr_social_aware_test/social_aware_test_brake/eval/epoch_3/default/result.pkl"
    # ap_result_path = "mtr_social_aware_test/baseline_test_brake/eval/epoch_3/default/result.pkl"

    raw_data_path = 'crash_data_all_raw.pkl'
    output_path = 'crash_scenario_results.pkl'

    # 加载数据
    with open(social_aware_result_path, 'rb') as f:
        social_aware_data = pickle.load(f)
    with open(ap_result_path, 'rb') as f:
        ap_data = pickle.load(f)
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)

    # 匹配并处理场景中的真值数据
    processed_scenarios = []

    # 建立原始数据的ID映射
    raw_data_map = {"_".join([str(scenario['ID'][i]) for i in range(3)]): scenario for scenario in raw_data}
    # raw_data_map = {scenario['ID']: scenario for scenario in raw_data}

    # 建立social_aware和AP数据的ID映射
    social_aware_map = {str(scenario['scenario_id']): scenario for scenario in social_aware_data}
    ap_map = {str(scenario['scenario_id']): scenario for scenario in ap_data}

    # 处理所有匹配的场景
    for id in set(social_aware_map.keys()) | set(ap_map.keys()):
        if id not in set(raw_data_map.keys()):
            continue

        social_aware_scenario = social_aware_map.get(id)
        ap_scenario = ap_map.get(id)
        raw_scenario = raw_data_map[id]['agent']
        complete_id = "_".join(map(str, raw_data_map[id]['ID']))
        
        processed = process_single_scenario(complete_id, social_aware_scenario, ap_scenario, raw_scenario)
        processed_scenarios.append(processed)

    # 保存结果
    with open(output_path, 'wb') as f:
        pickle.dump(processed_scenarios, f)

    print(f"处理完成，共{len(processed_scenarios)}个场景，结果已保存至{output_path}")


if __name__ == '__main__':
    results_process_and_file_rearrange()
    output_path = 'crash_scenario_results.pkl'
    with open(output_path, 'rb') as f:
        data_processed = pickle.load(f)
    print(len(data_processed))