import os
import math
import zlib
import random
import matplotlib.pyplot as plt
import torch
import pickle
import pandas as pd
import numpy as np
from tensorflow.python.ops.gen_stateful_random_ops import rng_read_and_skip
from tqdm import tqdm
from pathlib import Path
import multiprocessing
from multiprocessing import Process

from mtr.utils import common_utils
from mtr.datasets.dataset import DatasetTemplate
from mtr.config import cfg


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


class social_awareDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        # 选择数据集路径
        # 250221修改：目前的代码中training是bool变量，暂时先按这个逻辑修改
        # 250226修改：路径修改  通过cfg.ROOT_DIR改为绝对路径  把train工作区放到tools文件夹
        # if self.training == 'train':
        #     data_dir = dataset_cfg.train_data_dir
        #     data_raw_dir = dataset_cfg.raw_train_data_dir
        # elif self.training == 'val':
        #     data_dir = dataset_cfg.val_data_dir
        #     data_raw_dir = dataset_cfg.raw_val_data_dir
        # elif self.training == 'test':
        #     data_dir = dataset_cfg.test_data_dir
        #     data_raw_dir = dataset_cfg.raw_test_data_dir
        if self.training:
            data_dir = cfg.ROOT_DIR / dataset_cfg.train_data_dir
            data_raw_dir = cfg.ROOT_DIR / dataset_cfg.raw_train_data_dir
        else:
            data_dir = cfg.ROOT_DIR / dataset_cfg.val_data_dir
            data_raw_dir = cfg.ROOT_DIR / dataset_cfg.raw_val_data_dir

        self.ex_list = []  # 处理后的数据实例
        self.cfg = dataset_cfg

        self.dim = 3
        self.num_historical_steps = 50
        self.num_future_steps = 60
        self.num_steps = 50 + 60 if self.training in ['train', 'val'] else 50
        # 250226修改：路径修改  将工作文件夹设到tools下
        if dataset_cfg.reuse_temp_file:
            with open(cfg.ROOT_DIR / data_dir, 'rb') as f:
                self.ex_list = pickle.load(f)
                # self.ex_list = pickle.load(f)[:10]
        else:
            with open(cfg.ROOT_DIR / data_raw_dir, 'rb') as f:
                files = pickle.load(f)
                # files = pickle.load(f)[:50]
            with open(cfg.ROOT_DIR / dataset_cfg.map_data_dir, 'rb') as f:
                map = pickle.load(f)

            # 用多进程并行处理数据
            pbar = tqdm(total=len(files))  # 初始化一个进度条对象 pbar，用于显示数据处理的进度
            queue = multiprocessing.Queue(dataset_cfg.core_num)  # 初始化一个多进程队列 queue，用于在多进程之间传递数据
            queue_res = multiprocessing.Queue()  # 初始化另一个多进程队列 queue_res，用于存储处理后的结果

            def calc_ex_list(queue, queue_res, dataset_cfg):
                dis_list = []
                while True:
                    file = queue.get()
                    if file is None:
                        break
                    # print(file)

                    id = file['ID']
                    agent = file['agent']

                    instance = dict()
                    instance['scenario_id'] = np.array([f'{id[0]}_{id[1]}_{id[2]}'])

                    instance_agent = self.get_agent_features(agent)
                    instance_map = self.get_map_features(map, instance_agent['center_objects_world'][0], instance_agent['center_objects_velo_world'][0])
                    instance.update(instance_agent)
                    instance.update(instance_map)

                    queue_res.put(instance)

            processes = [Process(target=calc_ex_list, args=(queue, queue_res, dataset_cfg,)) for _ in range(dataset_cfg.core_num)]
            for each in processes:
                each.start()
            for file in files:
                assert file is not None
                queue.put(file)
                pbar.update(1)

            while not queue.empty():
                pass
            pbar.close()

            self.ex_list = []
            pbar = tqdm(total=len(files))
            for i in range(len(files)):
                t = queue_res.get()
                if t is not None:
                    self.ex_list.append(t)
                pbar.update(1)
            pbar.close()

            for _ in range(dataset_cfg.core_num):
                queue.put(None)
            for each in processes:
                each.join()

            # cache the ex_list
            with open(data_dir, "wb") as f:
                pickle.dump(self.ex_list, f)

        assert len(self.ex_list) > 0
        # print("valid data size is", len(self.ex_list))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        data = self.ex_list[idx]
        while True:
            try:
                instance = data
                break
            except:
                data = self.ex_list[idx + 1]
        return instance

    def get_agent_features(self, data_agent):
        """
        从pkl数据继续做一些转化和整理
        """
        # v_x = (data_agent[0, 1:, 0] - data_agent[0, :-1, 0]) / 0.05
        # v_x_ = data_agent[0, :-1, 2]
        # v_x__ = v_x - v_x_
        # 250219修改：数据结构中最后三列risk相关删除，因此做一些对应修改
        time_history = 100 + 1  # 险态发生时刻，后100的数据为待预测数据的gt
        obj_trajs_mask = data_agent[:, :, -1].astype(bool)
        # 训练数据要转化为以自车为中心的相对值
        center_gt_trajs_world = data_agent[0, time_history:, :4][None, :, :]  # 切片后变为二维矩阵，[None, :, :]增加上新的维度变回三维矩阵
        center_gt_trajs_mask = obj_trajs_mask[0, time_history:][None, :]
        center_gt_final_valid_idx = np.array([np.arange(201 - time_history)[center_gt_trajs_mask[0]][-1]])
        target_gt_trajs_world = data_agent[1, time_history:, :4][None, :, :]
        # 250307修改：补上center_objects_velo_world
        center_objects_velo_world = data_agent[0, time_history - 1, [2, 3]]

        center_objects_world = data_agent[0, time_history - 1, [0, 1, 5]]  # 初始位置xy与航向角
        center_objects_world[2] = -center_objects_world[2]
        norm_x, norm_y = rotate(data_agent[:, :, 0] - center_objects_world[0],
                                data_agent[:, :, 1] - center_objects_world[1],
                                center_objects_world[2])  # 求相对值并旋转至正
        norm_theta = data_agent[:, :, 5] + center_objects_world[2]
        norm_vx, norm_vy = rotate(data_agent[:, :, 2], data_agent[:, :, 3], center_objects_world[2])

        # v_x = (norm_x[0, 1:] - norm_x[0, :-1]) / 0.05
        # v_x_ = norm_vx[0, :-1]
        # v_x__ = v_x - v_x_

        # import matplotlib.pyplot as plt
        # plt.plot(position_norm_x[0], position_norm_y[0], 'r', alpha=0.5)
        # plt.axis('equal')
        # plt.show()

        data_agent = np.concatenate([norm_x[:, :, None], norm_y[:, :, None], norm_theta[:, :, None],
                                     norm_vx[:, :, None], norm_vy[:, :, None],
                                     data_agent[:, :, 4:5], data_agent[:, :, 6:9],
                                     np.arange(len(data_agent[0]))[None, :, None].repeat(len(data_agent), axis=0)],
                                     axis=-1)   # data_agent[:, :, -3:] 需要删掉
        data_agent[~obj_trajs_mask] = 0
        obj_trajs = np.array(data_agent, dtype=np.float32)[None, :, :, :]  # 【？】这里为什么要再增加一个维度
        obj_trajs_mask = obj_trajs_mask[None, :, :].astype(bool)


        track_index_to_predict = np.array([0])
        obj_trajs_pos = obj_trajs[:, :, :, :3]
        obj_trajs_last_pos = obj_trajs_pos[:, :, time_history - 1]
        # obj_types = np.array(['vehicle'] * len(data_agent))[None, :]
        obj_types = np.array(['TYPE_VEHICLE'] * len(data_agent))[None, :]
        center_objects_type = np.array(['TYPE_VEHICLE'])

        if_crash = np.array([not obj_trajs_mask[0, 0, -1]])
        crash_time = np.array([(np.arange(201) - 100)[obj_trajs_mask[0, 0]][-1] if if_crash else -1])

        ret_dict_agent = {
            'obj_trajs': obj_trajs[:, :, :time_history],
            'obj_trajs_mask': obj_trajs_mask[:, :, :time_history],
            'track_index_to_predict': track_index_to_predict,
            'obj_trajs_pos': obj_trajs_pos[:, :, :time_history],
            'obj_trajs_last_pos': obj_trajs_last_pos,
            # 'obj_types': obj_types,
            'center_objects_world': center_objects_world[None, :],
            # 250307修改：补上center_objects_velo_world
            'center_objects_velo_world': center_objects_velo_world[None, :],

            'obj_trajs_future_state': obj_trajs[:, :, time_history:][:, :, :, [0, 1, 3, 4]],
            'obj_trajs_future_mask': obj_trajs_mask[:, :, time_history:],
            'center_gt_trajs': obj_trajs[:, 0, time_history:][:, :, [0, 1, 3, 4]],
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_world': np.array(center_gt_trajs_world, dtype=np.float32),
            'center_objects_type': center_objects_type,

            'if_crash': if_crash,
            'crash_time': crash_time,
            # 250307修改：测试obj_risk_future_state这一项有没有用
            'obj_risk_future_state': obj_trajs[:, :, time_history:][:, :, :, -3:],

            'target_gt_trajs_world': target_gt_trajs_world,
        }

        return ret_dict_agent

    # 250307修改：替换新的get_map_features函数
    def old_get_map_features(self, map, center_objects_world):
        num_map_polylines = 8

        map_new = []
        sub_map_point = 20
        for map_i in map:
            sub_map = (len(map_i) - 1) // sub_map_point + 1
            order = np.arange(len(map_i))
            for j in range(sub_map):
                map_j = map_i[j * sub_map_point:(j + 1) * sub_map_point]
                map_j = np.array([map_j_k + [1] for map_j_k in map_j])
                map_j_order = order[j * sub_map_point:(j + 1) * sub_map_point]
                map_j = np.concatenate([map_j[:, :-1], map_j_order[:, None], map_j[:, -1:]], axis=-1)
                dist = np.sqrt(np.square(map_j[:, 0] - center_objects_world[0]) +
                               np.square(map_j[:, 1] - center_objects_world[1]))
                map_j = np.concatenate([map_j[:, :-1], dist[:, None], map_j[:, -1:]], axis=1)
                if len(map_j) > 3:
                    map_new.append(np.concatenate([map_j, np.zeros((sub_map_point - len(map_j), map_j.shape[-1]))], axis=0))
        map = np.array(map_new)

        map[:, :, -2][~map[:, :, -1].astype(bool)] = 9999
        sorted_indices = np.argsort(np.min(map[:, :, -2], axis=1))
        map = map[sorted_indices][:num_map_polylines]

        norm_x, norm_y = rotate(map[:, :, 0] - center_objects_world[0],
                                map[:, :, 1] - center_objects_world[1],
                                center_objects_world[2])
        norm_theta = map[:, :, 2] + center_objects_world[2]
        map[:, :, 0], map[:, :, 1] = norm_x, norm_y
        map[:, :, 2] = norm_theta
        map_mask = map[:, :, -1].astype(bool)
        map[~map_mask] = 0
        map_center = []
        for i in range(len(map)):
            map_center.append(np.average(map[i, :, :3][map_mask[i, :]], axis=0))
        map_center = np.array(map_center, dtype=np.float32)

        # import  matplotlib.pyplot as plt
        # for map_i in map:
        #     plt.plot(map_i[:, 0][map_i[:, -1].astype(bool)], map_i[:, 1][map_i[:, -1].astype(bool)])
        # plt.axis('equal')
        # plt.show()

        ret_dict_map = {}
        ret_dict_map['map_polylines'] = np.array(map, dtype=np.float32)[None, :, :, :]
        ret_dict_map['map_polylines_mask'] = map_mask[None, :, :]
        ret_dict_map['map_polylines_center'] = map_center[None, :, :]

        return ret_dict_map

    def get_map_features(self, map, center_objects_world, center_objects_velo_world):
        # num_map_polylines = 16
        # pred_center_objects_world = center_objects_world
        # map_new = []
        # sub_map_point = 20
        # for map_i in map:
        #     sub_map = (len(map_i) - 1) // sub_map_point + 1
        #     order = np.arange(len(map_i))
        #     for j in range(sub_map):
        #         map_j = map_i[j * sub_map_point:(j + 1) * sub_map_point]
        #         map_j = np.array([map_j_k + [1] for map_j_k in map_j])
        #         map_j_order = order[j * sub_map_point:(j + 1) * sub_map_point]
        #         map_j = np.concatenate([map_j[:, :-1], map_j_order[:, None], map_j[:, -1:]], axis=-1)
        #         dist = np.sqrt(np.square(map_j[:, 0] - pred_center_objects_world[0]) +
        #                        np.square(map_j[:, 1] - pred_center_objects_world[1]))
        #         map_j = np.concatenate([map_j[:, :-1], dist[:, None], map_j[:, -1:]], axis=1)
        #         if len(map_j) > 3:
        #             map_new.append(np.concatenate([map_j, np.zeros((sub_map_point - len(map_j), map_j.shape[-1]))], axis=0))
        # map = np.array(map_new)

        num_map_polylines = 6
        pred_time = 1
        pred_center_objects_world = np.array([center_objects_world[0] + center_objects_velo_world[0] * pred_time,
                                              center_objects_world[1] + center_objects_velo_world[1] * pred_time])
        map_new = []
        sub_map_point = 20
        for map_i in map:
            map_i = map_i[::3]
            sub_map = (len(map_i) - 1) // sub_map_point + 1
            order = np.arange(len(map_i))
            for j in range(sub_map):
                map_j = map_i[j * sub_map_point:(j + 1) * sub_map_point]
                map_j = np.array([map_j_k + [1] for map_j_k in map_j])
                map_j_order = order[j * sub_map_point:(j + 1) * sub_map_point]
                map_j = np.concatenate([map_j[:, :-1], map_j_order[:, None], map_j[:, -1:]], axis=-1)
                dist = np.sqrt(np.square(map_j[:, 0] - pred_center_objects_world[0]) +
                               np.square(map_j[:, 1] - pred_center_objects_world[1]))
                map_j = np.concatenate([map_j[:, :-1], dist[:, None], map_j[:, -1:]], axis=1)
                if len(map_j) > 3:
                    map_new.append(np.concatenate([map_j, np.zeros((sub_map_point - len(map_j), map_j.shape[-1]))], axis=0))
        map = np.array(map_new)

        map[:, :, -2][~map[:, :, -1].astype(bool)] = 9999
        sorted_indices = np.argsort(np.min(map[:, :, -2], axis=1))
        map = map[sorted_indices][:num_map_polylines]

        norm_x, norm_y = rotate(map[:, :, 0] - center_objects_world[0],
                                map[:, :, 1] - center_objects_world[1],
                                center_objects_world[2])
        norm_theta = map[:, :, 2] + center_objects_world[2]
        map[:, :, 0], map[:, :, 1] = norm_x, norm_y
        map[:, :, 2] = norm_theta
        map_mask = map[:, :, -1].astype(bool)
        map[~map_mask] = 0
        map_center = []
        for i in range(len(map)):
            map_center.append(np.average(map[i, :, :3][map_mask[i, :]], axis=0))
        map_center = np.array(map_center, dtype=np.float32)

        ret_dict_map = {}
        ret_dict_map['map_polylines'] = np.array(map, dtype=np.float32)[None, :, :, :]
        ret_dict_map['map_polylines_mask'] = map_mask[None, :, :]
        ret_dict_map['map_polylines_center'] = map_center[None, :, :]

        return ret_dict_map

    def generate_prediction_dicts(self, batch_dict, epoch_id, output_path=None, if_test=False):
        """
        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 3)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']

        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        ## test CV model
        if False:
        # if True:
            time_1 = 0.05 * torch.arange(1, 101)[None, :].repeat(num_center_objects, 1)
            time_2 = torch.ones(100)[None, :].repeat(num_center_objects, 1)
            pred_trajs_x = input_dict['obj_trajs'][:, 0, -1, 0][:, None].repeat(1, 100) + \
                           input_dict['obj_trajs'][:, 0, -1, 3][:, None].repeat(1, 100) * time_1
            pred_trajs_y = input_dict['obj_trajs'][:, 0, -1, 1][:, None].repeat(1, 100) + \
                           input_dict['obj_trajs'][:, 0, -1, 4][:, None].repeat(1, 100) * time_1
            # import matplotlib.pyplot as plt
            # pred_trajs_x, pred_trajs_y = pred_trajs_x.cpu().numpy(), pred_trajs_y.cpu().numpy()
            # for i in range(len(pred_trajs_x)):
            #     history_x = input_dict['obj_trajs'][i, 0, :, 0].cpu().numpy()
            #     history_y = input_dict['obj_trajs'][i, 0, :, 1].cpu().numpy()
            #     plt.plot(history_x, history_y, 'b.')
            #     plt.plot(pred_trajs_x[i], pred_trajs_y[i], 'r.')
            #     plt.axis('equal')
            #     plt.show()
            #     plt.close()
            pred_trajs_vx = input_dict['obj_trajs'][:, 0, -1, 3][:, None].repeat(1, 100) * time_2
            pred_trajs_vy = input_dict['obj_trajs'][:, 0, -1, 4][:, None].repeat(1, 100) * time_2
            pred_trajs = torch.cat([pred_trajs_x[:, None, :, None], pred_trajs_y[:, None, :, None],
                                    pred_trajs_vx[:, None, :, None], pred_trajs_vy[:, None, :, None]],
                                   dim=-1).repeat(1, num_modes, 1, 1).cuda()
            pred_scores = torch.zeros(num_center_objects, num_modes).cuda()
            pred_scores[:, 0] = 1
            num_feat = 4

        ## test CA model
        if False:
        # if True:
            time_1 = 0.05 * torch.arange(1, 101)[None, :].repeat(num_center_objects, 1)
            ax = (input_dict['obj_trajs'][:, 0, -3, 3] - input_dict['obj_trajs'][:, 0, -1, 3]) / 0.1
            ay = (input_dict['obj_trajs'][:, 0, -3, 4] - input_dict['obj_trajs'][:, 0, -1, 4]) / 0.1
            pred_trajs_vx = input_dict['obj_trajs'][:, 0, -1, 3][:, None].repeat(1, 100) + \
                            ax[:, None].repeat(1, 100) * time_1
            pred_trajs_vy = input_dict['obj_trajs'][:, 0, -1, 4][:, None].repeat(1, 100) + \
                            ay[:, None].repeat(1, 100) * time_1
            pred_trajs_vx = np.clip(pred_trajs_vx, 0, 30)
            pred_trajs_vy = np.clip(pred_trajs_vy, 0, 30)
            pred_trajs_x = input_dict['obj_trajs'][:, 0, -1, 0][:, None].repeat(1, 100) + \
                           np.cumsum(pred_trajs_vx, axis=-1) * 0.05
            pred_trajs_y = input_dict['obj_trajs'][:, 0, -1, 1][:, None].repeat(1, 100) + \
                           np.cumsum(pred_trajs_vy, axis=-1) * 0.05
            # import matplotlib.pyplot as plt
            # pred_trajs_x, pred_trajs_y = pred_trajs_x.cpu().numpy(), pred_trajs_y.cpu().numpy()
            # for i in range(len(pred_trajs_x)):
            #     history_x = input_dict['obj_trajs'][i, 0, :, 0].cpu().numpy()
            #     history_y = input_dict['obj_trajs'][i, 0, :, 1].cpu().numpy()
            #     plt.plot(history_x, history_y, 'b.')
            #     plt.plot(pred_trajs_x[i], pred_trajs_y[i], 'r.')
            #     plt.axis('equal')
            #     plt.show()
            #     plt.close()

            pred_trajs = torch.cat([pred_trajs_x[:, None, :, None], pred_trajs_y[:, None, :, None],
                                    pred_trajs_vx[:, None, :, None], pred_trajs_vy[:, None, :, None]],
                                   dim=-1).repeat(1, num_modes, 1, 1).cuda()
            pred_scores = torch.zeros(num_center_objects, num_modes).cuda()
            pred_scores[:, 0] = 1
            num_feat = 4


        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=-center_objects_world[:, 2].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]
        gt_trajs_world = input_dict['center_gt_trajs_world'][:, :, :2]

        pred_velo_world = common_utils.rotate_points_along_z(
            points=pred_trajs[:, :, :, -2:].view(num_center_objects, num_modes * num_timestamps, 2),
            angle=-center_objects_world[:, 2].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, 2)
        gt_velo_world = input_dict['center_gt_trajs_world'][:, :, -2:]


        obj_trajs_hist = input_dict['obj_trajs'][:, :, :, :2]
        _, num_agents, num_timestamps_, _ = obj_trajs_hist.shape
        obj_trajs_hist_world = common_utils.rotate_points_along_z(
            points=obj_trajs_hist.view(num_center_objects, num_agents * num_timestamps_, 2),
            angle=-center_objects_world[:, 2].view(num_center_objects).cpu()
        ).view(num_center_objects, num_agents, num_timestamps_, 2)
        obj_trajs_hist_world += center_objects_world[:, None, None, 0:2].cpu()
        obj_trajs_hist[~input_dict['obj_trajs_mask']] = 0

        obj_trajs_futu = input_dict['obj_trajs_future_state'][:, :, :, :2]
        obj_trajs_futu_world = common_utils.rotate_points_along_z(
            points=obj_trajs_futu.view(num_center_objects, num_agents * num_timestamps, 2),
            angle=-center_objects_world[:, 2].view(num_center_objects).cpu()
        ).view(num_center_objects, num_agents, num_timestamps, 2)
        obj_trajs_futu_world += center_objects_world[:, None, None, 0:2].cpu()
        obj_trajs_futu_world[~input_dict['obj_trajs_future_mask']] = 0

        map_polylines = input_dict['map_polylines'][:, :, :, :2]
        _, num_polylines, num_points, _ = map_polylines.shape
        map_polylines_world = common_utils.rotate_points_along_z(
            points=map_polylines.view(num_center_objects, num_polylines * num_points, 2),
            angle=-center_objects_world[:, 2].view(num_center_objects).cpu()
        ).view(num_center_objects, num_polylines, num_points, 2)
        map_polylines_world += center_objects_world[:, None, None, 0:2].cpu()
        map_polylines_world[~input_dict['map_polylines_mask']] = 0


        pred_dict_list = []
        for obj_idx in range(num_center_objects):
            single_pred_dict = {
                # 250225修改：注释掉所有没有的项，对非risk项标注一下
                'scenario_id': input_dict['scenario_id'][obj_idx],
                'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                # 'center_objects_id': input_dict['center_objects_id'][obj_idx],
                # 'object_type': input_dict['center_objects_type'][obj_idx],
                'gt_trajs': gt_trajs_world[obj_idx].cpu().numpy(),
                'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy(),
                'gt_trajs_mask': input_dict['center_gt_trajs_mask'][obj_idx].cpu().numpy(),

                'if_crash': input_dict['if_crash'][obj_idx],
                'crash_time': input_dict['crash_time'][obj_idx],
                'pred_velo': pred_velo_world[obj_idx].cpu().numpy(),
                'gt_velo': gt_velo_world[obj_idx].cpu().numpy(),
                # 'gt_risk': input_dict['center_gt_risks'][obj_idx].cpu().numpy(),
                'target_gt_trajs': input_dict['target_gt_trajs_world'][obj_idx].cpu().numpy(),
                'center_gt_final_valid_idx': input_dict['center_gt_final_valid_idx'][obj_idx].cpu().numpy(),

                'obj_trajs_hist': obj_trajs_hist_world[obj_idx].numpy(),
                'obj_trajs_hist_mask': input_dict['obj_trajs_mask'][obj_idx].numpy(),
                'obj_trajs_futu': obj_trajs_futu_world[obj_idx].numpy(),
                'obj_trajs_futu_mask': input_dict['obj_trajs_future_mask'][obj_idx].numpy(),
                'map_polylines': map_polylines_world[obj_idx].numpy(),
                'map_polylines_mask': input_dict['map_polylines_mask'][obj_idx].numpy(),
                # 'selected_idxs': batch_dict['selected_idxs'][obj_idx].cpu().numpy(),
                # selected_idxs看起来不是risk相关的项  待进一步考虑

                # 'gt_hist_risks_m': input_dict['obj_trajs'][obj_idx, 0, :, -3].cpu().numpy(),
                # 'gt_futu_risks_m': input_dict['center_gt_risks'][obj_idx, :, 0].cpu().numpy(),
                # 'pred_risks_m': pred_risks[obj_idx, :, :].cpu().numpy(),
            }
            pred_dict_list.append(single_pred_dict)

        # 250228修改：增加预测轨迹的可视化
        if epoch_id in [190]:
            # pred_line_idx = np.argmax(pred_dict_list[plot_idx]['pred_scores'])
            # pred_line = pred_dict_list[plot_idx]['pred_trajs'][pred_line_idx, :, :]
            # plt.plot(history[:, 0], history[:, 1], 'k-', linewidth=10, alpha=0.5)
            # plt.plot(history[-1, 0], history[-1, 1], 'ko', linewidth=13)
            # plt.plot(pred_line[:, 0], pred_line[:, 1], 'r--', linewidth=10, alpha=0.5)
            # plt.plot(gt_line[:, 0], gt_line[:, 1], 'g-.', linewidth=10, alpha=0.5)
            plot_idx = random.randint(0, len(pred_dict_list) - 1)  # 每个batch随机选一个scenario画一下看看
            history = pred_dict_list[plot_idx]['obj_trajs_hist'][0, :, :]
            gt_line = pred_dict_list[plot_idx]['gt_trajs']

            plt.figure(figsize=(10, 8))
            plt.plot(history[:, 0], history[:, 1], 'k-', linewidth=20, alpha=0.5)  # 历史数据
            plt.plot(history[-1, 0], history[-1, 1], marker='.', ms=40, c='black')  # 开始规划的点
            plt.plot(gt_line[:, 0], gt_line[:, 1], 'g-', linewidth=20, alpha=0.5)  # 真值轨迹
            colorset = [(1, 0.647, 0), (1, 0.549, 0), (1, 0.392, 0), (1, 0.271, 0), (1, 0.157, 0), (1, 0, 0)]
            for i in range(6):
                pred_line = pred_dict_list[plot_idx]['pred_trajs'][i, :, :]
                plt.plot(pred_line[:, 0], pred_line[:, 1], ls='-.', c=colorset[i], linewidth=8, alpha=0.4)  # 预测的六条轨迹
                plt.plot(pred_line[-1, 0], pred_line[-1, 1], marker='*', ms=18, c=colorset[i])  # 预测轨迹的终点
            plt.tick_params(axis='both',labelsize=30)
            plt.axis('equal')
            plt.show()
            plt.close()

        return pred_dict_list

    # 250225修改：尝试模仿waymo_dataset，调用eval_forcasting内的函数，编写evaluation
    def evaluation(self, pred_dicts, output_path=None, eval_method='social_aware', **kwargs):
        if eval_method == 'social_aware':
            from .eval_forecasting import social_aware_evaluation
            # try:
            #     num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
            # except:
            #     num_modes_for_eval = 6
            metric_results, metric_result_str= social_aware_evaluation(pred_dicts=pred_dicts)

        else:
            raise NotImplementedError

        return metric_result_str, metric_results


