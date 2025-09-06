import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import pandas as pd


def calculate_theta(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    Improved heading angle calculation function, handling small angle cases
    """
    theta = np.zeros_like(vx)
    theta[0] = np.arctan2(vy[0], vx[0])

    for i in range(1, len(vx)):
        current_theta = np.arctan2(vy[i], vx[i])

        # When the speed is very small (less than 0.01m/s is considered stationary), use the previous moment's angle
        if np.sqrt(vx[i] ** 2 + vy[i] ** 2) < 0.01:
            theta[i] = theta[i - 1]
        else:
            theta[i] = current_theta

    return theta


def calculate_yaw_rate(theta: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Calculate yaw rate (first-order difference method)
    """
    yaw_rate = np.zeros_like(theta)
    yaw_rate[1:] = np.diff(theta) / dt

    # Handle angle jumps (such as sudden changes from -pi to pi)
    jumps = np.abs(np.diff(theta)) > np.pi
    yaw_rate[1:][jumps] = (np.diff(theta)[jumps] - 2 * np.pi * np.sign(np.diff(theta)[jumps])) / dt

    return yaw_rate


def extract_data_frames_from_raw(data: np.ndarray, start: int, end: int):
    """Helper function: Extract 6 feature lists from vehicle data for specified frame range"""
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
    提取数据
    """
    # Extract ego vehicle data
    ego_data = raw_scenario[0]
    hist_ego_data = extract_data_frames_from_raw(ego_data, 0, 101)
    gt_ego_data = extract_data_frames_from_raw(ego_data, 101, 201)

    # Extract other vehicle data (risk, oth1, oth2)
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
    Extract prediction data from scenario, with customizable key prefix
    """
    # Get the index of the prediction with the highest score
    pred_idx = np.argmax(scenario['pred_scores'])

    # Extract trajectory and velocity
    traj = scenario['pred_trajs'][pred_idx]  # (100,2)
    velo = scenario['pred_velo'][pred_idx]  # (100,2)
    vx, vy = velo[:, 0], velo[:, 1]

    # Calculate motion parameters
    v = np.sqrt(vx ** 2 + vy ** 2)
    theta = calculate_theta(vx, vy)
    yaw_rate = calculate_yaw_rate(theta)

    # Build return dictionary (using prefix)
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
    """Process all data for a single scenario"""
    # Extract raw data
    raw_data = extract_raw_scenario_data(raw_scenario)

    # Process prediction data
    social_aware_pred = extract_prediction_data(social_aware_scenario, 'social_aware_')
    ap_pred = extract_prediction_data(ap_scenario, 'AP_')

    # Build unified data format
    scenario_dict = {**raw_data, **social_aware_pred, **ap_pred}
    scenario_dict.update({'scenario_id': scenario_id})

    return scenario_dict

# ***************************************Main Function******************************************

def results_process_and_file_rearrange():
    # Configure paths
    social_aware_result_path = "test/baseline_test_crash_scenario/eval/epoch_3/default/result.pkl"
    ap_result_path = "test/social_aware_test_crash_scenario/eval/epoch_3/default/result.pkl"
    raw_data_path = '../data/crash_data_raw_test.pkl'
    output_path = 'crash_scenario_results.pkl'

    # Load data
    with open(social_aware_result_path, 'rb') as f:
        social_aware_data = pickle.load(f)
    with open(ap_result_path, 'rb') as f:
        ap_data = pickle.load(f)
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)

    # Match and process ground truth data in scenarios
    processed_scenarios = []

    # Create ID mapping for raw data
    raw_data_map = {"_".join([str(scenario['ID'][i]) for i in range(3)]): scenario for scenario in raw_data}
    # raw_data_map = {scenario['ID']: scenario for scenario in raw_data}

    # Create ID mapping for social_aware and AP data
    social_aware_map = {str(scenario['scenario_id']): scenario for scenario in social_aware_data}
    ap_map = {str(scenario['scenario_id']): scenario for scenario in ap_data}

    # Process all matched scenarios
    for id in set(social_aware_map.keys()) | set(ap_map.keys()):
        if id not in set(raw_data_map.keys()):
            continue

        social_aware_scenario = social_aware_map.get(id)
        ap_scenario = ap_map.get(id)
        raw_scenario = raw_data_map[id]['agent']
        complete_id = "_".join(map(str, raw_data_map[id]['ID']))
        
        processed = process_single_scenario(complete_id, social_aware_scenario, ap_scenario, raw_scenario)
        processed_scenarios.append(processed)

    # Save results
    with open(output_path, 'wb') as f:
        pickle.dump(processed_scenarios, f)

    print(f"Processing completed, total {len(processed_scenarios)} scenarios, results saved to {output_path}")


if __name__ == '__main__':
    results_process_and_file_rearrange()
    output_path = 'crash_scenario_results.pkl'
    with open(output_path, 'rb') as f:
        data_processed = pickle.load(f)
    print(len(data_processed))