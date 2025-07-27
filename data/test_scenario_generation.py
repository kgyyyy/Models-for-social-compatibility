import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_lanes_with_specified_length(pkl_path, specified_length):
    """
    读取 map.pkl 文件，绘制轨迹点数为指定长度的所有车道

    Args:
        pkl_path (str): map.pkl 文件路径
        specified_length (int): 指定的轨迹点数（如 10 表示只画点数为 10 的车道）
    """
    # 1. 读取 pkl 文件
    with open(pkl_path, 'rb') as f:
        map_lanes = pickle.load(f)

    # 2. 筛选符合长度的车道
    filtered_lanes = [lane for lane in map_lanes if len(lane) == specified_length]
    print(f"找到 {len(filtered_lanes)} 条轨迹点数为 {specified_length} 的车道")

    if not filtered_lanes:
        print("无匹配车道，请检查 specified_length 或数据！")
        return

    # 3. 绘制所有符合条件的车道
    plt.figure(figsize=(10, 8))
    for i, lane in enumerate(filtered_lanes):
        # 提取 x, y 坐标
        x_coords = [point[0] for point in lane]
        y_coords = [point[1] for point in lane]

        # 绘制折线图（带标记点）
        plt.plot(x_coords, y_coords, marker='o', label=f'Lane {i + 1}')

    # 添加图例和标题
    plt.title(f'Lanes with {specified_length} Points (Total: {len(filtered_lanes)})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_lanes_with_specified_length(pkl_path, specified_length):
    with open(pkl_path, 'rb') as f:
        map_lanes = pickle.load(f)
    filtered_lanes = [lane for lane in map_lanes if len(lane) == specified_length]
    return filtered_lanes  # 共8条，其中我们需要idx=4和5的

def generate_braking_data(target_lane, a, deceleration, brake_start_frame=100, fps=20):
    """
    修改后的制动工况生成函数，增加刹车起始帧参数
    """
    # 计算车道方向
    x_start, y_start = target_lane[0][0], target_lane[0][1]
    x_end, y_end = target_lane[-1][0], target_lane[-1][1]
    theta = np.arctan2(y_end - y_start, x_end - x_start)

    # 计算起始点坐标
    x0 = x_start + a * np.cos(theta)
    y0 = y_start + a * np.sin(theta)

    # 初始化运动参数
    v_total = 25.0  # 初始总速度(m/s)
    dt = 1.0 / fps  # 时间步长(s)

    # 生成201帧数据
    scenario = np.zeros((201, 10))

    # 前段匀速运动
    for i in range(brake_start_frame):
        t = i * dt
        x = x0 + v_total * np.cos(theta) * t
        y = y0 + v_total * np.sin(theta) * t

        scenario[i, :6] = [x, y,
                          v_total * np.cos(theta), v_total * np.sin(theta),
                          v_total,
                          theta]
        scenario[i, 6:9] = 0
        scenario[i, 9] = 1

    # 后段减速运动
    for i in range(brake_start_frame, 201):
        t = (i - brake_start_frame) * dt
        v = max(0, v_total - deceleration * t)

        # 计算位移
        s = v_total * t - 0.5 * deceleration * t**2
        x = scenario[brake_start_frame-1, 0] + s * np.cos(theta)
        y = scenario[brake_start_frame-1, 1] + s * np.sin(theta)

        scenario[i, :6] = [x, y,
                          v * np.cos(theta), v * np.sin(theta),
                          v,
                          theta]
        scenario[i, 6:9] = 0
        scenario[i, 9] = 1

    return scenario

def generate_uniform_motion_data(target_lane, a, fps=20):
    """
    生成匀速运动场景数据

    Args:
        target_lane: 目标车道数据（从map_lanes中提取的单条车道）
        a: 起始点距离车道起点的距离（米）
        fps: 帧率（默认100Hz）
        speed: 匀速运动速度（默认25m/s）

    Returns:
        scenario: (201,10)的ndarray，格式同急刹工况
                 每行[x, y, vx, vy, v_total, theta, 0, 0, 0, 1]
    """
    # 1. 计算车道方向
    x_start, y_start = target_lane[0][0], target_lane[0][1]
    x_end, y_end = target_lane[-1][0], target_lane[-1][1]
    theta = np.arctan2(y_end - y_start, x_end - x_start)

    # 2. 计算起始点坐标
    x0 = x_start + a * np.cos(theta)
    y0 = y_start + a * np.sin(theta)

    # 3. 初始化参数
    speed = 25.0
    dt = 1.0 / fps
    scenario = np.zeros((201, 10))

    # 4. 生成201帧匀速数据
    for i in range(201):
        t = i * dt
        x = x0 + speed * np.cos(theta) * t
        y = y0 + speed * np.sin(theta) * t

        scenario[i, :] = [x, y,
                          speed * np.cos(theta), speed * np.sin(theta),
                          speed,
                          theta,
                          0, 0, 0, 1]  # 后4列固定

    return scenario

def generate_scenarios_with_matrix(lanes):
    """
    生成参数矩阵组合的所有工况 (修改制动帧数和ID格式)
    """
    # 定义参数矩阵
    # a_values = [25, 35, 45, 55, 65]  # risk车距离
    # b_values = [-50, -40, -30, -20, -10, 10, 20]  # oth1车距离
    # c_values = [-50, -40, -30]  # oth2车距离
    a_values = np.linspace(25, 70, 9)
    b_values = np.linspace(-20, 10, 12)
    c_values = np.linspace(-50, -30, 5)

    test_scenario_list = []
    scenario_count = 0  # 场景计数器

    # 遍历所有组合
    for a in a_values:
        for b in b_values:
            for c in c_values:
                # 生成risk车(前车)数据 - 101帧开始制动
                risk_data= generate_braking_data(lanes[7], a, deceleration= 6, brake_start_frame=101)

                # 生成oth1车数据(匀速)
                oth1_data = generate_uniform_motion_data(lanes[6], b)

                # 生成oth2车数据(111帧开始制动)
                oth2_data= generate_braking_data(lanes[7], c, deceleration= 3, brake_start_frame=121)

                # 生成自车数据
                ego_data = generate_uniform_motion_data(lanes[7], 0)

                # 合并数据 (ego, risk, oth1, oth2)
                data_agent = np.concatenate([
                    ego_data[None, :, :],
                    risk_data[None, :, :],
                    oth1_data[None, :, :],
                    oth2_data[None, :, :]
                ], axis=0)

                # 创建场景字典 (ID格式为(i, a, b, c))
                data_dict = {
                    'ID': (scenario_count, a, b, c),  # 修改ID格式
                    'agent': data_agent
                }
                test_scenario_list.append(data_dict)
                scenario_count += 1  # 递增计数器

                if scenario_count == 1:
                    plt.figure(figsize=(10, 8))
                    plt.plot(ego_data[:, 0], ego_data[:, 1], label='ego')
                    plt.plot(risk_data[:, 0], risk_data[:, 1], label='risk')
                    plt.plot(oth1_data[:, 0], oth1_data[:, 1], label='oth1')
                    plt.plot(oth2_data[:, 0], oth2_data[:, 1], label='oth2')
                    plt.show()

    # 写入pkl文件
    test_file = 'test_data_v6_raw.pkl'
    with open(test_file, 'wb') as file:
        pickle.dump(test_scenario_list, file)

    print(f"已生成 {scenario_count} 种工况组合，制动时机：risk@101帧, oth2@121帧")


def main():
    # 获得车道数据
    map_file = 'map.pkl'
    specified_length = 237
    plot_lanes_with_specified_length(map_file, specified_length)
    lanes = get_lanes_with_specified_length(map_file, specified_length)
    # generate_scenarios_with_matrix(lanes)
    # # 循环生成测试集
    # test_scenario_list = []
    # for i in range(10000):
    #     risk_data, a = generate_braking_data(lanes[4])
    #     b = random.uniform(-60, -20)  # b的随机取值范围
    #     oth_data = generate_uniform_motion_data(lanes[5], b)
    #     ego_data = generate_uniform_motion_data(lanes[4], 0)
    #
    #     # 分别画出来看一下
    #     # plt.figure(figsize=(10, 8))
    #     # plt.plot(ego_data[:, 0], ego_data[:, 1], label='ego')
    #     # plt.plot(risk_data[:, 0], risk_data[:, 1], label='risk')
    #     # plt.plot(oth_data[:, 0], oth_data[:, 1], label='other')
    #     # plt.show()
    #
    #     data_agent= np.concatenate([ego_data[None, :, :], risk_data[None, :, :], oth_data[None, :, :]], axis=0)
    #     id = (i, a, b)
    #     data_dict = {'ID': id, 'agent': data_agent}
    #     test_scenario_list.append(data_dict)
    #
    # # 写入pkl文件
    # test_file = 'test_data_right_raw.pkl'
    # with open(test_file, 'wb') as file:
    #     pickle.dump(test_scenario_list, file)

if __name__ == "__main__":
    # map_file = 'map.pkl'
    # specified_length = 237
    # plot_lanes_with_specified_length(map_file, specified_length)
    main()
