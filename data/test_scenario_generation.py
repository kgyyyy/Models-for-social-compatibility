import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_lanes_with_specified_length(pkl_path, specified_length):
    """
    Read the map.pkl file and plot all lanes with the specified number of trajectory points.
    Use this function to manually find longer straight roads in the map as lanes for constructing test scenarios.
    """
    # read pkl
    with open(pkl_path, 'rb') as f:
        map_lanes = pickle.load(f)

    # Filter lanes that meet the length criteria
    filtered_lanes = [lane for lane in map_lanes if len(lane) == specified_length]
    print(f"find {len(filtered_lanes)} lanse with the length of {specified_length}")

    if not filtered_lanes:
        print("No matching lanes found, please check specified_length or the data!")
        return

    # plot
    plt.figure(figsize=(10, 8))
    for i, lane in enumerate(filtered_lanes):
        # Extract x, y coordinates
        x_coords = [point[0] for point in lane]
        y_coords = [point[1] for point in lane]

        # Plot line chart (with markers)
        plt.plot(x_coords, y_coords, marker='o', label=f'Lane {i + 1}')

    # Add legend and title
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
    return filtered_lanes

def generate_braking_data(target_lane, a, deceleration, brake_start_frame=100, fps=20):
    """
    Braking vehicle data constructor.
    """
    # Calculate lane direction
    x_start, y_start = target_lane[0][0], target_lane[0][1]
    x_end, y_end = target_lane[-1][0], target_lane[-1][1]
    theta = np.arctan2(y_end - y_start, x_end - x_start)

    # Calculate starting point coordinates
    x0 = x_start + a * np.cos(theta)
    y0 = y_start + a * np.sin(theta)

    # Initialize motion parameters
    v_total = 25.0
    dt = 1.0 / fps

    # Generate 201 frames of data
    scenario = np.zeros((201, 10))

    # First segment: uniform motion
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

    # Later segment: deceleration motion
    for i in range(brake_start_frame, 201):
        t = (i - brake_start_frame) * dt
        v = max(0, v_total - deceleration * t)

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
    Surrounding vehicle uniform motion data constructor
    """
    # direction
    x_start, y_start = target_lane[0][0], target_lane[0][1]
    x_end, y_end = target_lane[-1][0], target_lane[-1][1]
    theta = np.arctan2(y_end - y_start, x_end - x_start)

    # init point
    x0 = x_start + a * np.cos(theta)
    y0 = y_start + a * np.sin(theta)

    # init
    speed = 25.0
    dt = 1.0 / fps
    scenario = np.zeros((201, 10))

    # data generation
    for i in range(201):
        t = i * dt
        x = x0 + speed * np.cos(theta) * t
        y = y0 + speed * np.sin(theta) * t

        scenario[i, :] = [x, y,
                          speed * np.cos(theta), speed * np.sin(theta),
                          speed,
                          theta,
                          0, 0, 0, 1]

    return scenario

def generate_scenarios_with_matrix(lanes):
    """
    生成brake参数矩阵组合的所有工况
    """
    # Define parameter matrix
    # a_values = [25, 35, 45, 55, 65]  # risk vehicle distance
    # b_values = [-50, -40, -30, -20, -10, 10, 20]  # oth1 vehicle distance
    # c_values = [-50, -40, -30]  # oth2 vehicle distance
    a_values = np.linspace(25, 70, 9)
    b_values = np.linspace(-20, 10, 12)
    c_values = np.linspace(-50, -30, 5)

    test_scenario_list = []
    scenario_count = 0

    for a in a_values:
        for b in b_values:
            for c in c_values:
                # Generate risk vehicle (front vehicle) data - braking starts at frame 101
                risk_data= generate_braking_data(lanes[7], a, deceleration= 6, brake_start_frame=101)

                # Generate oth1 vehicle data (uniform motion)
                oth1_data = generate_uniform_motion_data(lanes[6], b)

                # Generate oth2 vehicle data (braking starts at frame 121)
                oth2_data= generate_braking_data(lanes[7], c, deceleration= 3, brake_start_frame=121)

                # Generate ego vehicle data
                ego_data = generate_uniform_motion_data(lanes[7], 0)

                # Merge data (ego, risk, oth1, oth2)
                data_agent = np.concatenate([
                    ego_data[None, :, :],
                    risk_data[None, :, :],
                    oth1_data[None, :, :],
                    oth2_data[None, :, :]
                ], axis=0)

                # Create scenario dictionary (ID format is (i, a, b, c))
                data_dict = {
                    'ID': (scenario_count, a, b, c),
                    'agent': data_agent
                }
                test_scenario_list.append(data_dict)
                scenario_count += 1

                if scenario_count == 1:
                    plt.figure(figsize=(10, 8))
                    plt.plot(ego_data[:, 0], ego_data[:, 1], label='ego')
                    plt.plot(risk_data[:, 0], risk_data[:, 1], label='risk')
                    plt.plot(oth1_data[:, 0], oth1_data[:, 1], label='oth1')
                    plt.plot(oth2_data[:, 0], oth2_data[:, 1], label='oth2')
                    plt.show()

    # Write to pkl file
    test_file = 'brake_test_data_raw.pkl'
    with open(test_file, 'wb') as file:
        pickle.dump(test_scenario_list, file)

    print(f"Generated {scenario_count} scenario combinations, braking timing: risk@frame 101, oth2@frame 121")

def main():
    # Obtain lane data
    map_file = 'map.pkl'
    # plot_lanes_with_specified_length(map_file, specified_length)
    specified_length = 237
    lanes = get_lanes_with_specified_length(map_file, specified_length)
    generate_scenarios_with_matrix(lanes)

if __name__ == "__main__":
    main()
