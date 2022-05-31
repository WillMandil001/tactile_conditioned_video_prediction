# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import os
import cv2
import csv
import glob
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pickle import dump
from sklearn import preprocessing
from datetime import datetime
from scipy.spatial.transform import Rotation as R

dataset_path = "/home/willmandil/Robotics/Data_sets/PRI/shelf_can/"
train_data_dir = dataset_path + 'train/'
test_data_dir = dataset_path + 'test/'
train_out_dir  = dataset_path + 'train_formatted/'
test_out_dir   = dataset_path + 'test_formatted/'
scaler_out_dir = dataset_path + 'scaler/'

for path in [train_out_dir, test_out_dir, scaler_out_dir]:
    try:
        os.mkdir(path)
    except FileExistsError or FileNotFoundError:
        pass

smooth = False
image = True
image_height = 64
image_width = 64
context_length = 0
horrizon_length = 70
one_sequence_per_test = True
data_train_percentage = 1.0

lines = ["smooth: " + str(smooth),
         "image: " + str(image),
         "image_height: " + str(image_height),
         "image_width: " + str(image_width),
         "context_length: " + str(context_length),
         "horrizon_length: " + str(horrizon_length),
         "one_sequence_per_test: " + str(one_sequence_per_test),
         "data_train_percentage: " + str(data_train_percentage),
         "dataset_path: " + str(dataset_path),
         "train_data_dir: " + str(train_data_dir),
         "test_data_dir: " + str(test_data_dir),
         "train_out_dir: " + str(train_out_dir),
         "test_out_dir: " + str(test_out_dir),
         "scaler_out_dir: " + str(scaler_out_dir)]
with open(scaler_out_dir + "dataset_info.txt", 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

class data_formatter:
    def __init__(self):
        self.files_train = []
        self.files_test = []
        self.files_test_2 = []
        self.full_data_tactile = []
        self.full_data_robot = []
        self.full_data_image = []
        self.full_data_depth = []
        self.smooth = smooth
        self.image = image
        self.image_height = image_height
        self.image_width = image_width
        self.context_length = context_length
        self.horrizon_length = horrizon_length
        self.one_sequence_per_test = one_sequence_per_test
        self.data_train_percentage = data_train_percentage

    def create_map(self):
        for stage in [train_out_dir, test_out_dir]:
            self.path_file = []
            index_to_save = 0
            print(stage)
            if stage == train_out_dir:
                files_to_run = self.files_train
            elif stage == test_out_dir:
                files_to_run = self.files_test
            print(files_to_run)
            path_save = stage
            for experiment_number, file in tqdm(enumerate(files_to_run), total=len(files_to_run)):

                tactile, robot, image, depth, side_image = self.load_file_data(file)

                # scale the data
                for index, (standard_scaler, min_max_scalar) in enumerate(zip(self.tactile_standard_scaler, self.tactile_min_max_scalar)):
                    tactile[:, index] = standard_scaler.transform(tactile[:, index])
                    tactile[:, index] = min_max_scalar.transform(tactile[:, index])

                for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
                    robot[:, index] = np.squeeze(min_max_scalar.transform(robot[:, index].reshape(-1, 1)))

                # save images and save space:
                image_names = []
                for time_step in range(len(image)):
                    image_name = "image_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
                    image_names.append(image_name)
                    np.save(path_save + image_name, image[time_step])

                if self.one_sequence_per_test:
                    sequence_length = self.context_length + self.horrizon_length - 1
                else:
                    sequence_length = self.context_length + self.horrizon_length
                for time_step in range(len(tactile) - sequence_length):
                    robot_data_euler_sequence   = [robot[time_step + t] for t in range(sequence_length)]
                    tactile_data_sequence       = [tactile[time_step + t] for t in range(sequence_length)]
                    image_name_sequence = [image_names[time_step + t] for t in range(sequence_length)]
                    experiment_data_sequence    = experiment_number
                    time_step_data_sequence     = [time_step + t for t in range(sequence_length)]

                    ####################################### Save the data and add to the map ###########################################
                    np.save(path_save + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
                    np.save(path_save + 'tactile_data_sequence_' + str(index_to_save), tactile_data_sequence)
                    np.save(path_save + 'image_name_sequence_' + str(index_to_save), image_name_sequence)
                    np.save(path_save + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
                    np.save(path_save + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
                    np.save(path_save + 'side_view_image_' + str(index_to_save), np.array(side_image))
                    ref = []
                    ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
                    ref.append('tactile_data_sequence_' + str(index_to_save) + '.npy')
                    ref.append('image_name_sequence_' + str(index_to_save) + '.npy')
                    ref.append('experiment_number_' + str(index_to_save) + '.npy')
                    ref.append('time_step_data_' + str(index_to_save) + '.npy')
                    ref.append('side_view_image_' + str(index_to_save) + '.npy')
                    self.path_file.append(ref)
                    index_to_save += 1
                    if self.one_sequence_per_test:
                        break

            self.save_map(path_save)

    def save_map(self, path, test=False):
        if test:
            with open(path + '/map_' + str(self.test_no) + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'image_name_sequence', 'experiment_number', 'time_steps', "side_view_image"])
                for row in self.path_file:
                    writer.writerow(row)
        else:
            with open(path + '/map.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'image_name_sequence', 'experiment_number', 'time_steps', 'side_view_image'])
                for row in self.path_file:
                    writer.writerow(row)

    def scale_data(self):
        files = self.files_train + self.files_test
        for file in tqdm(files):
            tactile, robot, image, depth, side_image = self.load_file_data(file)
            self.full_data_tactile += list(tactile)
            self.full_data_robot += list(robot)

        self.full_data_robot = np.array(self.full_data_robot)
        self.full_data_tactile = np.array(self.full_data_tactile)

        self.robot_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot[:, feature].reshape(-1, 1)) for feature in range(6)]
        self.tactile_standard_scaler = [preprocessing.StandardScaler().fit(self.full_data_tactile[:, feature]) for feature in range(3)]
        self.tactile_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.tactile_standard_scaler[feature].transform(self.full_data_tactile[:, feature])) for feature in range(3)]

        self.save_scalars()

    def load_file_data(self, file):
        # print(file)
        robot_state = np.array(pd.read_csv(file + '/robot_states.csv', header=None))
        xela_sensor = np.array(np.load(file + '/tactile_states.npy'))
        image_data = np.array(np.load(file + '/color_images.npy'))
        depth_data = np.array(np.load(file + '/depth_images.npy'))
        side_image = np.array(np.load(file + '/start_color_side.npy'))

        # convert orientation to euler, and remove column labels:
        robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in robot_state[1:]]).astype(float)

        # split tactile sensor into the three forces | find start value average for each force | find offsets for each taxel | take away start value average from the tactile data:
        tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
        tactile_mean_start_values = [int(sum(tactile_data_split[feature][0]) / len(tactile_data_split[feature][0])) for feature in range(3)]
        tactile_offsets = [[tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data_split[feature][0]] for feature in range(3)]
        tactile_data = [[tactile_data_split[feature][ts] + tactile_offsets[feature] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]

        # Resize the image using PIL antialiasing method (Copied from CDNA data formatting)
        raw = []
        for k in range(len(image_data)):
            tmp = Image.fromarray(image_data[k])
            tmp = tmp.resize((image_height, image_width), Image.ANTIALIAS)
            tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((image_height, image_width, 3))
            tmp = tmp.astype(np.float32) / 255.0
            raw.append(tmp)
        image_data = np.array(raw)

        tmp = Image.fromarray(side_image)
        tmp = tmp.resize((image_height, image_width), Image.ANTIALIAS)
        tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
        tmp = tmp.reshape((image_height, image_width, 3))
        side_image = tmp.astype(np.float32) / 255.0

        return np.array(tactile_data), robot_task_space, image_data, depth_data, side_image

    def load_file_names(self):
        self.files_train = glob.glob(train_data_dir + '/*')
        self.files_test = glob.glob(test_data_dir + '/*')
        self.files_train = random.sample(self.files_train, int(len(self.files_train) * self.data_train_percentage))

    def smooth_the_trial(self, tactile_data):
        for force in range(tactile_data.shape[1]):
            for taxel in range(tactile_data.shape[2]):
                tactile_data[:, force, taxel] = [None for i in range(3)] + list(self.smooth_func(tactile_data[:, force, taxel], 6)[3:-3]) + [None for i in range(3)]

        return tactile_data

    def smooth_func(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def save_scalars(self):
        # save the scalars
        dump(self.tactile_standard_scaler[0], open(scaler_out_dir + 'tactile_standard_scaler_x.pkl', 'wb'))
        dump(self.tactile_standard_scaler[1], open(scaler_out_dir + 'tactile_standard_scaler_y.pkl', 'wb'))
        dump(self.tactile_standard_scaler[2], open(scaler_out_dir + 'tactile_standard_scaler_z.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[0], open(scaler_out_dir + 'tactile_min_max_scalar_x.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[1], open(scaler_out_dir + 'tactile_min_max_scalar_y.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[2], open(scaler_out_dir + 'tactile_min_max_scalar.pkl', 'wb'))

        dump(self.robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', 'wb'))
        dump(self.robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', 'wb'))
        dump(self.robot_min_max_scalar[2], open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', 'wb'))
        dump(self.robot_min_max_scalar[3], open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', 'wb'))
        dump(self.robot_min_max_scalar[4], open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', 'wb'))
        dump(self.robot_min_max_scalar[5], open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', 'wb'))

    def create_image(self, tactile_x, tactile_y, tactile_z):
        # convert tactile data into an image:
        image = np.zeros((4, 4, 3), np.float32)
        index = 0
        for x in range(4):
            for y in range(4):
                image[x][y] = [tactile_x[index], tactile_y[index], tactile_z[index]]
                index += 1
        reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(self.image_height, self.image_width), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
        return reshaped_image


def main():
    df = data_formatter()
    df.load_file_names()
    df.scale_data()
    df.create_map()


if __name__ == "__main__":
    main()