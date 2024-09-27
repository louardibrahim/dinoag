import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from dinoag.augment import hard as augmenter

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class CARLA_Data(Dataset):

    def __init__(self, root, data_folders, img_aug=False):
        self.root = root
        self.img_aug = img_aug
        self._batch_read_number = 0

        self.front_img = []
        self.x = []
        self.y = []
        self.command = []
        self.target_command = []
        self.target_gps = []
        self.theta = []
        self.speed = []
        self.value = []
        self.feature = []
        self.x_command = []
        self.y_command = []
        self.only_ap_brake = []
        self.future_x = []
        self.future_y = []
        self.future_theta = []
        self.future_only_ap_brake = []

        for sub_root in data_folders:
            data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()

            self.x_command += data['x_target']
            self.y_command += data['y_target']
            self.command += data['target_command']
            self.front_img += data['front_img']
            self.x += data['input_x']
            self.y += data['input_y']
            self.theta += data['input_theta']
            self.speed += data['speed']
            self.future_x += data['future_x']
            self.future_y += data['future_y']
            self.future_theta += data['future_theta']
            self.future_only_ap_brake += data['future_only_ap_brake']
            self.value += data['value']
            self.feature += data['feature']
            self.only_ap_brake += data['only_ap_brake']

        self._im_transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.front_img)

    def __getitem__(self, index):
        """Returns the item at index idx."""
        data = dict()
        data['front_img'] = self.front_img[index]

        if self.img_aug:
            data['front_img'] = self._im_transform(
                augmenter(self._batch_read_number).augment_image(
                    np.array(Image.open(self.root + self.front_img[index][0]))
                )
            )
        else:
            data['front_img'] = self._im_transform(
                np.array(Image.open(self.root + self.front_img[index][0]))
            )

        # Fix for theta=nan in some measurements
        if np.isnan(self.theta[index][0]):
            self.theta[index][0] = 0.0

        ego_x = self.x[index][0]
        ego_y = self.y[index][0]
        ego_theta = self.theta[index][0]

        waypoints = []
        for i in range(4):
            R = np.array([
                [np.cos(np.pi/2 + ego_theta), -np.sin(np.pi/2 + ego_theta)],
                [np.sin(np.pi/2 + ego_theta), np.cos(np.pi/2 + ego_theta)]
            ])
            local_command_point = np.array([self.future_y[index][i] - ego_y, self.future_x[index][i] - ego_x])
            local_command_point = R.T.dot(local_command_point)
            waypoints.append(local_command_point)

        data['waypoints'] = np.array(waypoints)

        R = np.array([
            [np.cos(np.pi/2 + ego_theta), -np.sin(np.pi/2 + ego_theta)],
            [np.sin(np.pi/2 + ego_theta), np.cos(np.pi/2 + ego_theta)]
        ])
        local_command_point = np.array([-1 * (self.x_command[index] - ego_x), self.y_command[index] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = local_command_point[:2]

        local_command_point_aim = np.array([(self.y_command[index] - ego_y), self.x_command[index] - ego_x])
        local_command_point_aim = R.T.dot(local_command_point_aim)
        data['target_point_aim'] = local_command_point_aim[:2]

        data['speed'] = self.speed[index]
        data['feature'] = self.feature[index]
        data['value'] = self.value[index]
        command = self.command[index]

        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        data['target_command'] = torch.tensor(cmd_one_hot)

        self._batch_read_number += 1
        return data
