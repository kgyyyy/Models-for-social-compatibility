# coding=utf-8
import pandas as pd
import os
import numpy as np
import json
import time
import pickle
import random


def data_agent_process(data, id):
    """
    将carla的excel数据处理为三维的数据表
    """
    #********************读取、讨论、切割三类agent（自车、危险车、其他周车）的数据********************
    # 各对象数据列的起始列和终止列
    Ego_start = data.columns.get_loc("STEERING WHEEL")
    Ego_end = data.columns.get_loc("sign_risk")
    Risk_start = data.columns.get_loc("risk_id")
    Other_1_start = data.columns.get_loc("b_actor1_id")
    Other_2_start = data.columns.get_loc("b_actor2_id")
    Other_3_start = data.columns.get_loc("b_actor3_id")
    Other_4_start = data.columns.get_loc("b_actor4_id")
    Other_5_start = data.columns.get_loc("b_actor5_id")
    Other_6_start = data.columns.get_loc("b_actor6_id")
    Other_7_start = data.columns.get_loc("b_actor7_id")

    # 险态触发时刻
    risk_time = np.arange(len(data))[data["sign_risk"] != 0]
    risk_time = risk_time[np.argmin(np.abs(risk_time - 100))]
    risk_id = data["risk_id"][risk_time]

    # 碰撞时刻
    crash_time = data['timestamp_Collision_'].to_numpy()
    nan_value = np.isnan(crash_time[:].astype(float))
    crash_time = np.arange(len(data))[~nan_value]  # 筛选出 crash_time 中非 NaN 值对应的[索引]


    data_ego = data[list(data.columns[Ego_start:Ego_end + 1])].to_numpy()

    if len(data_ego) < 201:  # 按照预处理算法  正常数据应该为201行
        return False, None

    # 处理他车+危险车数据，整合为（8*N,M+1）的数组形式；危险车轨迹数据在risk列中可能没有完全记录，因为危险车会换，所以要和他车的一起处理后再提取出来
    data_oth_0 = data[list(data.columns[Risk_start:Other_1_start])].to_numpy()
    data_oth_1 = data[list(data.columns[Other_1_start:Other_2_start])].to_numpy()
    data_oth_2 = data[list(data.columns[Other_2_start:Other_3_start])].to_numpy()
    data_oth_3 = data[list(data.columns[Other_3_start:Other_4_start])].to_numpy()
    data_oth_4 = data[list(data.columns[Other_4_start:Other_5_start])].to_numpy()
    data_oth_5 = data[list(data.columns[Other_5_start:Other_6_start])].to_numpy()
    data_oth_6 = data[list(data.columns[Other_6_start:Other_7_start])].to_numpy()
    data_oth_7 = data[list(data.columns[Other_7_start:Other_7_start + 17])].to_numpy()
    data_oth = np.concatenate((data_oth_0, data_oth_1, data_oth_2, data_oth_3, data_oth_4, data_oth_5, data_oth_6, data_oth_7), axis=0)  # 垂直方向拼接
    data_oth_order = np.array(list(range(len(data_ego))) * 8)  # 生成time_step序号数组：0 到 len(data_ego) - 1 的重复序列，重复8次
    data_oth = np.concatenate((data_oth, data_oth_order[:, None]), axis=1)  # 把序号数组拼接到数据最右一列

    nan_value = np.isnan(data_oth[:, 0].astype(float))
    data_oth = data_oth[~nan_value]  # 过滤nan值
    data_oth[:, 0] = data_oth[:, 0].astype(int)

    agent_ids = np.unique(data_oth[:, 0])
    agent_ids = np.delete(agent_ids, np.where(agent_ids == risk_id))

    # 获得他车中危险车的data
    data_agent_risk = data_oth[data_oth[:, 0] == risk_id]
    data_agent_risk_index = data_agent_risk[:, -1].argsort()  # 按照最后一列的序号数组排序整理
    data_agent_risk = data_agent_risk[data_agent_risk_index]
    time_step = data_agent_risk[:, -1]
    time_step_ = np.concatenate((np.array([-1]), time_step[:-1]))  # time_step往前偏移一步
    data_agent_risk = data_agent_risk[time_step != time_step_]  # 去除重复的时间步数据
    data_agent_risk = data_agent_risk[:, 2:].astype(float)

    # 同样处理思路处理非危险车的他车，重新将各车的数据整理为单个列表
    data_agent_oth = []
    for agent_id in agent_ids[2:]:
        data_oth_ = data_oth[data_oth[:, 0] == agent_id]
        data_oth_ = data_oth_[:, 2:].astype(float)
        data_oth_index = data_oth_[:, -1].argsort()
        data_oth_ = data_oth_[data_oth_index]
        time_step = data_oth_[:, -1]
        time_step_ = np.concatenate((np.array([-1]), time_step[:-1]))
        data_oth_ = data_oth_[time_step != time_step_]
        data_agent_oth.append(data_oth_)

    # 根据crash_time对三类agent的数据做切割，截取至碰撞前
    if len(crash_time) > 0:
        if crash_time[0] < 103:  # 险态触发前自车发生碰撞  该数据不要
            return False, None
        data_ego = data_ego[:crash_time[0]]
        risk_nocrash = np.arange(len(data_agent_risk))[data_agent_risk[:, -1] < crash_time[0]]
        data_agent_risk = data_agent_risk[risk_nocrash]

        data_agent_oth_ = []
        for j in range(len(data_agent_oth)):
            data_agent_oth_j = data_agent_oth[j]
            oth_j_nocrash = np.arange(len(data_agent_oth_j))[data_agent_oth_j[:, -1] < crash_time[0]]
            if len(data_agent_oth_j[oth_j_nocrash]) > 0:
                data_agent_oth_.append(data_agent_oth_j[oth_j_nocrash])
        data_agent_oth = data_agent_oth_

    # ********************按照MTR输入数据的要求整理和补充计算********************
    traj_ego_x, traj_ego_y = data_ego[:, 9], data_ego[:, 10]
    traj_ego_vx, traj_ego_vy = data_ego[:, 6], data_ego[:, 7]
    traj_ego_v = (traj_ego_vx ** 2 + traj_ego_vy ** 2) ** 0.5
    traj_ego_theta = np.arctan2(traj_ego_vy, traj_ego_vx)
    traj_ego_control = data_ego[:, :3]

    # 检查单一时间步内的自车的位移，若位移过大，说明应用了restart，轨迹数据有突变，不适合加入学习
    if np.max(((traj_ego_x[1:] - traj_ego_x[:-1]) ** 2 + (traj_ego_y[1:] - traj_ego_y[:-1]) ** 2) ** 0.5) > 2.5:
        print('Ego Invalid!')
        print(id)
        return False, None

    traj_risk_x, traj_risk_y = data_agent_risk[:, 6], data_agent_risk[:, 7]
    traj_risk_vx, traj_risk_vy = data_agent_risk[:, 3], data_agent_risk[:, 4]
    traj_risk_v = (traj_risk_vx ** 2 + traj_risk_vy ** 2) ** 0.5
    traj_risk_theta = np.arctan2(traj_risk_vy, traj_risk_vx)
    traj_risk_control = data_agent_risk[:, :3]
    # mask数据列：除自车外，所有周车可能因不在记录模块的采样范围而没有充分采集，使用mask项标记
    traj_risk_mask = np.zeros(len(traj_ego_x)).astype(bool)
    traj_risk_mask[data_agent_risk[:, -1].astype(int)] = True

    if np.max(((traj_risk_x[1:] - traj_risk_x[:-1]) ** 2 + (traj_risk_y[1:] - traj_risk_y[:-1]) ** 2) ** 0.5) > 2.5:
        return False, None

    if traj_risk_mask[-1] == False:
        return False, None

    traj_risk_x_masked, traj_risk_y_masked = np.zeros(len(traj_ego_x)), np.zeros(len(traj_ego_x))
    traj_risk_vx_masked, traj_risk_vy_masked = np.zeros(len(traj_ego_x)), np.zeros(len(traj_ego_x))
    traj_risk_v_masked, traj_risk_theta_masked = np.zeros(len(traj_ego_x)), np.zeros(len(traj_ego_x))
    traj_risk_control_masked = np.zeros((len(traj_ego_x), 3))
    traj_risk_x_masked[traj_risk_mask], traj_risk_y_masked[traj_risk_mask] = traj_risk_x, traj_risk_y
    traj_risk_vx_masked[traj_risk_mask], traj_risk_vy_masked[traj_risk_mask] = traj_risk_vx, traj_risk_vy
    traj_risk_v_masked[traj_risk_mask], traj_risk_theta_masked[traj_risk_mask] = traj_risk_v, traj_risk_theta
    traj_risk_control_masked[traj_risk_mask] = traj_risk_control

    traj_oth_masked = []
    for data_agent_oth_i in data_agent_oth:
        traj_oth_x, traj_oth_y = data_agent_oth_i[:, 6], data_agent_oth_i[:, 7]
        traj_oth_vx, traj_oth_vy = data_agent_oth_i[:, 3], data_agent_oth_i[:, 4]
        traj_oth_v = (traj_oth_vx ** 2 + traj_oth_vy ** 2) ** 0.5
        traj_oth_theta = np.arctan2(traj_oth_vy, traj_oth_vx)
        traj_oth_mask = np.zeros(len(traj_ego_x)).astype(bool)
        traj_oth_mask[data_agent_oth_i[:, -1].astype(int)] = True

        traj_oth_x_masked, traj_oth_y_masked = np.zeros(len(traj_ego_x)), np.zeros(len(traj_ego_x))
        traj_oth_vx_masked, traj_oth_vy_masked = np.zeros(len(traj_ego_x)), np.zeros(len(traj_ego_x))
        traj_oth_v_masked, traj_oth_theta_masked = np.zeros(len(traj_ego_x)), np.zeros(len(traj_ego_x))
        traj_oth_x_masked[traj_oth_mask], traj_oth_y_masked[traj_oth_mask] = traj_oth_x, traj_oth_y
        traj_oth_vx_masked[traj_oth_mask], traj_oth_vy_masked[traj_oth_mask] = traj_oth_vx, traj_oth_vy
        traj_oth_v_masked[traj_oth_mask], traj_oth_theta_masked[traj_oth_mask] = traj_oth_v, traj_oth_theta
        traj_oth_masked.append((traj_oth_x_masked, traj_oth_y_masked, traj_oth_vx_masked, traj_oth_vy_masked,
                                traj_oth_v_masked, traj_oth_theta_masked, traj_oth_mask))

    # 低速数据点和null数据点处不再计算航向角，沿用上一时刻航向角
    for i in range(len(traj_ego_x)):
        if i > 0 and traj_ego_v[i] < 0.5:
            traj_ego_theta[i] = traj_ego_theta[i - 1]
    for i in range(len(traj_risk_x)):
        if i > 0 and traj_risk_v[i] < 0.5 and traj_risk_mask[i] and traj_risk_mask[i - 1]:
            traj_risk_theta_masked[i] = traj_risk_theta_masked[i - 1]
    for j in range(len(traj_oth_masked)):
        for i in range(len(traj_oth_masked[j][4])):
            if i > 0 and traj_oth_masked[j][4][i] < 0.5 and traj_oth_masked[j][-1][i] and traj_oth_masked[j][-1][i - 1]:
                traj_oth_masked[j][5][i] = traj_oth_masked[j][5][i - 1]

    # ********************数据格式组织********************
    traj_ego_mask = np.ones(len(traj_ego_x))  # 自车补充一列mask  数据格式一致
    data_ego = np.concatenate([traj_ego_x[:, None], traj_ego_y[:, None], traj_ego_vx[:, None], traj_ego_vy[:, None],
                               traj_ego_v[:, None], traj_ego_theta[:, None], traj_ego_control[:, :3],
                               traj_ego_mask[:, None]], axis=-1)
    data_risk = np.concatenate([traj_risk_x_masked[:, None], traj_risk_y_masked[:, None],
                                traj_risk_vx_masked[:, None], traj_risk_vy_masked[:, None],
                                traj_risk_v_masked[:, None], traj_risk_theta_masked[:, None],
                                traj_risk_control_masked[:, :3],
                                traj_risk_mask[:, None]], axis=-1)
    if len(traj_oth_masked) > 0:
        data_oth = np.array(traj_oth_masked)
        data_oth = np.transpose(data_oth, (0, 2, 1))
        data_oth = np.concatenate([data_oth[:, :, :6], -np.ones_like(data_oth)[:, :, :3], data_oth[:, :, 6:7]], axis=-1)
        data_oth[~data_oth[:, :, -4].astype(bool)] = 0
        data_final = np.concatenate([data_ego[None, :, :], data_risk[None, :, :], data_oth], axis=0)
    else:
        data_final = np.concatenate([data_ego[None, :, :], data_risk[None, :, :]], axis=0)

    # 长度短于201 即又发生碰撞  在最终数据中做一个标记
    if data_final.shape[1] < 201:
        data_final[:2, -1, -3] = 999
    data_final = np.concatenate([data_final, np.zeros((data_final.shape[0], 201 - data_final.shape[1],
                                                       data_final.shape[2]))], axis=1)

    return True, data_final


def data_map_process():
    """
    处理地图数据生成map.pkl的函数
    """
    with open("./map/Town04_new.json", 'r', encoding='UTF-8') as f:
        map = json.load(f)

    # lane_type_dict = {'Driving': 0}
    lane_change_dict = {'NONE': 0, 'Right': 0, 'Left': 0, 'Both': 0}
    inside_junction_dict = {True: 1, False: 0}

    map_lanes = []
    for map_i, map_i_ in map.items():
        map_i_xy = []
        for map_i_j, map_i_j_ in map_i_.items():
            map_lane = []
            pre_x, pre_y = map_i_j_[0]['x'], map_i_j_[0]['y']
            for map_i_j_k_ in map_i_j_:
                x, y = map_i_j_k_['x'], map_i_j_k_['y']
                theta = np.arctan2(y - pre_y, x - pre_x)
                lane_type = map_i_j_k_['lane_type']
                lane_id, road_id = map_i_j_k_['lane_id'], map_i_j_k_['road_id']
                lane_change = lane_change_dict[map_i_j_k_['lane_change']]
                inside_junction = inside_junction_dict[map_i_j_k_['inside_junction']]
                lane_width = map_i_j_k_['lane_width']
                map_lane.append([x, y, theta, pre_x, pre_y, lane_id, road_id, lane_change, inside_junction])
                pre_x, pre_y = x, y
            if len(map_lane) > 1:
                map_lane[0][2] = map_lane[1][2]
        map_lanes.append(map_lane)

    with open('../map.pkl', 'wb') as f:
        pickle.dump(map_lanes, f)


def main():
    time_start = time.time()

    dir = './social_awaredata'  # data folder to preprocess

    data_list = list()

    for item_0 in sorted(os.listdir(dir))[:]:
        dir_0 = os.path.join(dir, item_0)
        num_0 = int(item_0[7:10])  # 取被试编号
        print(dir_0)
        num_1 = int(item_0[0])  # 取险态类型
        num_2 = int(item_0.split('.')[-2].split('_')[-1])
        event_data = pd.read_excel(dir_0)
        id = (num_0, num_1, num_2)  # id ==（被试，险态类型，各个case）

        flag, data_agent = data_agent_process(event_data, id)
        if not flag:
            continue

        data_dict = {'ID': id, 'agent': data_agent}
        data_list.append(data_dict)

        print('Time: ', time.time() - time_start)

        with open("../data_raw.pkl", "wb") as file:
            pickle.dump(data_list, file)


    with open('../data_raw.pkl', 'rb') as f:
        data_list = pickle.load(f)

    # 对数据做一些统计，生成对应结构的info文件
    Event = [0, 0, 0]  #  统计三种险态的case数
    Crash = [0, 0, 0, 0]
    for data_i in data_list:
        id = data_i['ID']
        Event[id[1]-1] = Event[id[1]-1] + 1

        if_crash = 100 - np.sum(data_i['agent'][0, 101:, 0] == 0)  # 检查险态触发后是否有碰撞并根据碰撞时间分类统计
        if if_crash == 100:
            Crash[0] = Crash[0] + 1
        elif if_crash < 20:
            Crash[1] = Crash[1] + 1
        elif if_crash < 50:
            Crash[2] = Crash[2] + 1
        elif if_crash <= 100:
            Crash[3] = Crash[3] + 1

    Event_ = np.array(Event) / np.sum(Event)
    Crash_ = np.array(Crash) / np.sum(Crash)

    # 划分训练集  生成训练/验证文件
    random.seed(123)
    random.shuffle(data_list)
    num_val = int(0.2 * len(data_list))
    data_train = data_list[:-num_val]
    data_val = data_list[-num_val:]
    with open("../data_raw_train.pkl", "wb") as file:
        pickle.dump(data_train, file)
    with open("../data_raw_val.pkl", "wb") as file:
        pickle.dump(data_val, file)


if __name__ == "__main__":
    # data_map_process()
    main()
