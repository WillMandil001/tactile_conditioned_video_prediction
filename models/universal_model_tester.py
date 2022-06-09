# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import math
import click
import random
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

import os
import csv
import copy
import utils
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# standard video and tactile prediction models:
from universal_networks.SVG import Model as SVG

# tactile conditioned models:
from universal_networks.SVG_TC import Model as SVG_TC
from universal_networks.SVG_TC_TE import Model as SVG_TC_TE

# non-stochastic models:
from universal_networks.VG import Model as VG
from universal_networks.VG_MMMM import Model as VG_MMMM

# artificial occlusion training:
from universal_networks.SVG_TC_occ import Model as SVG_TC_occ
from universal_networks.SVG_TC_TE_occ import Model as SVG_TC_TE_occ


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)


class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)


class BatchGenerator:
    def __init__(self, test_data_dir, batch_size, image_size, occlusion_test, occlusion_size):
        self.occlusion_size = occlusion_size
        self.occlusion_test = occlusion_test
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_map = []
        with open(test_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_data(self):
        dataset_test = FullDataSet(self.data_map, self.test_data_dir, self.occlusion_size, image_size=self.image_size, occlusion_test=self.occlusion_test)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map, test_data_dir, occlusion_size=0, image_size=64, occlusion_test=False):
        self.occlusion_test = occlusion_test
        self.test_data_dir = test_data_dir
        self.image_size = image_size
        self.occlusion_size = occlusion_size
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.test_data_dir + value[0])

        tactile_data = np.load(self.test_data_dir + value[1])
        tactile_images = []
        for tactile_data_sample in tactile_data:
            tactile_images.append(create_image(tactile_data_sample, image_size=self.image_size))

        images = []
        occ_images = []
        if self.occlusion_test:
            # random start point:
            min_pixel = int(((self.occlusion_size / 2) + 1))
            max_pixel = int(self.image_size - ((self.occlusion_size / 2) + 1))
            if min_pixel >= max_pixel:
                rand_x = None
                rand_y = None
            else:
                rand_x = random.randint(min_pixel, max_pixel)
                rand_y = random.randint(min_pixel, max_pixel)
        for image_name in np.load(self.test_data_dir + value[2]):
            images.append(np.load(self.test_data_dir + image_name))
            if self.occlusion_test:
                occ_images.append(self.add_occlusion(np.load(self.test_data_dir + image_name), min_pixel, max_pixel, rand_x, rand_y))

        experiment_number = np.load(self.test_data_dir + value[3])
        time_steps = np.load(self.test_data_dir + value[4])
        # print(robot_data.astype(np.float32).shape,
        #       np.array(images).astype(np.float32).shape,
        #       np.array(tactile_images).astype(np.float32).shape,
        #       np.array(tactile_data).astype(np.float32).shape,
        #       experiment_number.shape, time_steps.shape,
        #       np.array(occ_images).astype(np.float32).shape)
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), np.array(tactile_data).astype(np.float32), experiment_number, time_steps, np.array(occ_images).astype(np.float32)]

    def add_occlusion(self, image, min_pixel, max_pixel, rand_x, rand_y):
        if min_pixel >= max_pixel:
            image[:,:,:] = 0.0
        else:
            for i in range(rand_x - int(self.occlusion_size / 2), rand_x + int(self.occlusion_size / 2)):
                for j in range(rand_y - int(self.occlusion_size / 2), rand_y + int(self.occlusion_size / 2)):
                    try:
                        image[i,j,:] = 0.0
                    except:
                        pass  # out of bounds
        return image

def create_image(tactile, image_size):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] = [tactile[0][index],
                           tactile[1][index],
                           tactile[2][index]]
            index += 1
    reshaped_image = cv2.resize(image.astype(np.float32), dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return reshaped_image


class UniversalTester():
    def __init__(self, data_save_path, model_save_path, test_data_dir, scaler_dir, model_save_name, model_folder_name, test_folder_name, model_stage, quant_analysis, qual_analysis, qual_tactile_analysis, quant_test, model_name_save_appendix):
        self.gain = None
        self.stage = None
        self.scaler_dir = scaler_dir
        self.quant_test = quant_test
        self.model_stage = model_stage
        self.test_data_dir = test_data_dir
        self.qual_analysis = qual_analysis
        self.quant_analysis = quant_analysis
        self.data_save_path = data_save_path
        self.model_save_path = model_save_path
        self.test_folder_name = test_folder_name
        self.model_folder_name = model_folder_name
        self.qual_tactile_analysis = qual_tactile_analysis

        saved_model = torch.load(model_save_path + model_save_name + model_name_save_appendix)
        features = saved_model["features"]

        # load features
        self.lr = features["lr"]
        self.beta = features["beta"]
        self.seed = features["seed"]
        self.beta1 = features["beta1"]
        self.niter = features["niter"]
        self.z_dim = features["z_dim"]
        self.g_dim = features["g_dim"]
        self.device = features["device"]
        self.n_past = features["n_past"]
        self.n_eval = features["n_eval"]
        self.epochs = features["epochs"]
        self.log_dir = features["log_dir"]
        self.dataset = features["dataset"]
        self.channels = features["channels"]
        self.n_future = features["n_future"]
        self.model_dir = features["model_dir"]
        self.data_root = features["data_root"]
        self.rnn_size = features["rnn_size"]
        self.optimizer = features["optimizer"]
        self.criterion = features["criterion"]
        self.batch_size = features["batch_size"]
        self.model_name = features["model_name"]
        self.scaler_dir = features["scaler_dir"]
        self.num_digits = features["num_digits"]
        self.image_width = features["image_width"]
        self.out_channels = features["out_channels"]
        self.tactile_size = features["tactile_size"]
        self.data_threads = features["data_threads"]
        self.train_data_dir = features["train_data_dir"]
        self.training_stages = features["training_stages"]
        self.last_frame_skip = features["last_frame_skip"]
        self.prior_rnn_layers = features["prior_rnn_layers"]
        self.train_percentage = features["train_percentage"]
        self.state_action_size = features["state_action_size"]
        self.posterior_rnn_layers = features["posterior_rnn_layers"]
        self.predictor_rnn_layers = features["predictor_rnn_layers"]
        self.validation_percentage = features["validation_percentage"]
        self.training_stages_epochs = features["training_stages_epochs"]
        # self.model_name_save_appendix = features["model_name_save_appendix"]

        print("self.batch_size", self.batch_size)

        try:
            self.occlusion_test = features["occlusion_test"]
            self.occlusion_max_size = features["occlusion_max_size"]
            self.occlusion_start_epoch = features["occlusion_start_epoch"]
            self.occlusion_gain_per_epoch = features["occlusion_gain_per_epoch"]
            self.occlusion_size = int(self.occlusion_max_size * self.image_width)
        except:
            self.occlusion_test = False
            self.occlusion_max_size = 0
            self.occlusion_start_epoch = 0
            self.occlusion_gain_per_epoch = 0
            self.occlusion_size = None

        print(self.occlusion_test)

        self.scene_loss_titles = ["Scene MAE: "] + ["Scene MAE T" + str(i) + ": " for i in range(self.n_future - 1)] + ["Scene MSE: "] + ["Scene MSE T" + str(i) + ": " for i in range(self.n_future - 1)] + ["Scene PSNR: "] +  ["Scene PSNR T" + str(i) + ": " for i in range(self.n_future - 1)] + ["Scene SSIM: "] + ["Scene SSIM T" + str(i) + ": " for i in range(self.n_future - 1)]
        self.tactile_loss_titles = ["Tactile MAE: ", "Tactile MAE T1: ", "Tactile MAE T5: ", "Tactile MAE T10: ",
                                    "Tactile MAE Shear X: ", "Tactile MAE Shear Y: ", "Tactile MAE Normal Z: ",
                                    "Tactile MSE: ", "Tactile MSE T1: ", "Tactile MSE T5: ", "Tactile MSE T10: ",
                                    "Tactile MSE Shear X: ", "Tactile MSE Shear Y: ", "Tactile MSE Normal Z: ",
                                    "Tactile PSNR: ", "Tactile PSNR T1: ", "Tactile PSNR T5: ", "Tactile PSNR T10: ",
                                    "Tactile PSNR Shear X: ", "Tactile PSNR Shear Y: ", "Tactile PSNR Normal Z: ",
                                    "Tactile SSIM: ", "Tactile SSIM T1: ", "Tactile SSIM T5: ", "Tactile SSIM T10: ",
                                    "Tactile SSIM Shear X: ", "Tactile SSIM Shear Y: ", "Tactile SSIM Normal Z: "]

        print(self.occlusion_test)

        # load model
        print(self.model_name)
        if self.model_name == "SVG":
            self.model = SVG(features)
        elif self.model_name == "SVG_TC":
            self.model = SVG_TC(features)
        elif self.model_name == "SVG_TC_TE":
            self.model = SVG_TC_TE(features)
        elif self.model_name == "VG":
            self.model = VG(features)
        elif self.model_name == "VG_MMMM":
            self.model = VG_MMMM(features)
        elif self.model_name == "SVG_TC_occ":
            self.model = SVG_TC_occ(features)
        elif self.model_name == "SVG_TC_TE_occ":
            self.model = SVG_TC_TE_occ(features)

        self.model.load_model(full_model = saved_model)
        # [saved_model[name].to("cpu") for name in saved_model if name != "features"]
        saved_model = []

        # load test set:
        print(self.occlusion_test, self.occlusion_size)
        BG = BatchGenerator(self.test_data_dir, self.batch_size, self.image_width, self.occlusion_test, self.occlusion_size)
        self.test_full_loader = BG.load_data()

        # test dataset
        if self.quant_analysis == True:
            self.test_model()
        if self.qual_analysis == True:
            self.test_model_qualitative()

    def test_model(self):
        self.objects = []
        self.performance_data_scene = []
        self.performance_data_tactile = []
        self.prediction_data = []
        self.current_exp = 0
        self.objects = []

        self.model.set_test()
        with torch.no_grad ():
            for index, batch_features in enumerate(self.test_full_loader):
                if batch_features[1].shape[0] == self.batch_size:
                    cut_size = -1
                    scene_MAE, tactile_MAE, predictions, tactile_predictions, images_occ = self.format_and_run_batch(batch_features, test=True)
                    self.performance_data_scene.append(scene_MAE)
                    self.performance_data_tactile.append(tactile_MAE)
                else:
                    cut_size = batch_features[0].shape[0]
                    print(cut_size)
                    batch_features[0] = torch.cat((batch_features[0], torch.zeros(self.batch_size - batch_features[0].shape[0], batch_features[0].shape[1], batch_features[0].shape[2])), 0)
                    batch_features[1] = torch.cat((batch_features[1], torch.zeros(self.batch_size - batch_features[1].shape[0], batch_features[1].shape[1], batch_features[1].shape[2], batch_features[1].shape[3], batch_features[1].shape[4])), 0)
                    batch_features[2] = torch.cat((batch_features[2], torch.zeros(self.batch_size - batch_features[2].shape[0], batch_features[2].shape[1], batch_features[2].shape[2], batch_features[2].shape[3], batch_features[2].shape[4])), 0)
                    batch_features[3] = torch.cat((batch_features[3], torch.zeros(self.batch_size - batch_features[3].shape[0], batch_features[3].shape[1], batch_features[3].shape[2], batch_features[3].shape[3])), 0)
                    batch_features[4] = torch.cat((batch_features[4], torch.zeros(self.batch_size - batch_features[4].shape[0])), 0)
                    batch_features[5] = torch.cat((batch_features[5], torch.zeros(self.batch_size - batch_features[5].shape[0], batch_features[5].shape[1])), 0)

                    scene_MAE, tactile_MAE, predictions, tactile_predictions, images_occ = self.format_and_run_batch(batch_features, cut_size=0, test=True)

                    self.performance_data_scene.append(scene_MAE)
                    self.performance_data_tactile.append(tactile_MAE)

        # Calculate losses across batches
        self.performance_data_scene = np.array(self.performance_data_scene)
        self.performance_data_tactile = np.array(self.performance_data_tactile)
        self.performance_data_scene_average = [sum(self.performance_data_scene[:, i]) / self.performance_data_scene.shape[0] for i in range(self.performance_data_scene.shape[1])]
        self.performance_data_tactile_average = [sum(self.performance_data_tactile[:, i]) / self.performance_data_tactile.shape[0] for i in range(self.performance_data_tactile.shape[1])]
        # Add titles
        self.performance_data_scene_average = [[title, i] for i, title in zip(self.performance_data_scene_average, self.scene_loss_titles)]
        self.performance_data_tactile_average = [[title, i]for i, title in zip(self.performance_data_tactile_average, self.tactile_loss_titles)]
        # Save losses
        np.save(self.data_save_path + self.test_folder_name + self.model_stage + "_losses_per_trial", np.array(self.performance_data_scene_average + self.performance_data_tactile_average))

        lines = list(self.performance_data_scene_average) + list(self.performance_data_tactile_average)
        with open (self.data_save_path + self.test_folder_name + self.model_stage + "_losses_per_trial.txt", 'w') as f:
            for line in lines:
                f.write(line[0] + str(line[1]))
                f.write('\n')
        print(np.array(self.performance_data_scene_average + self.performance_data_tactile_average))

    def test_model_qualitative(self):
        self.model.set_test()

        qual_save_path = self.model_save_path + "qualitative_analysis/"
        try:
            os.mkdir(qual_save_path)
        except FileExistsError or FileNotFoundError:
            pass

        if self.model_stage != "":
            qual_save_path = qual_save_path + self.model_stage + "/"
            try:
                os.mkdir(qual_save_path)
            except FileExistsError or FileNotFoundError:
                pass

        qual_save_path = qual_save_path + self.test_folder_name + "/"
        try:
            os.mkdir(qual_save_path)
        except FileExistsError or FileNotFoundError:
            pass

        with torch.no_grad():
            for index, batch_features in enumerate(self.test_full_loader):
                if batch_features[1].shape[0] != self.batch_size:
                    cut_size = batch_features[0].shape[0]
                    batch_features[0] = torch.cat((batch_features[0], torch.zeros(self.batch_size - batch_features[0].shape[0], batch_features[0].shape[1], batch_features[0].shape[2])), 0)
                    batch_features[1] = torch.cat((batch_features[1], torch.zeros(self.batch_size - batch_features[1].shape[0], batch_features[1].shape[1], batch_features[1].shape[2], batch_features[1].shape[3], batch_features[1].shape[4])), 0)
                    batch_features[2] = torch.cat((batch_features[2], torch.zeros(self.batch_size - batch_features[2].shape[0], batch_features[2].shape[1], batch_features[2].shape[2], batch_features[2].shape[3], batch_features[2].shape[4])), 0)
                    batch_features[3] = torch.cat((batch_features[3], torch.zeros(self.batch_size - batch_features[3].shape[0], batch_features[3].shape[1], batch_features[3].shape[2], batch_features[3].shape[3])), 0)
                    batch_features[4] = torch.cat((batch_features[4], torch.zeros(self.batch_size - batch_features[4].shape[0])), 0)
                    batch_features[5] = torch.cat((batch_features[5], torch.zeros(self.batch_size - batch_features[5].shape[0], batch_features[5].shape[1])), 0)
                else:
                    cut_size = -1
                predictions, tactile_predictions, images, tactile, images_occ = self.format_and_run_batch(batch_features, test=True, qualitative=True)
                predictions = predictions[:, :cut_size]
                # tactile_predictions = tactile_predictions[:cut_size]
                images = images[:, :cut_size]
                # tactile = tactile[:cut_size]
                # images_occ = images_occ[:cut_size]
                # if index in self.quant_test[:,0]:
                # list_of_sub_batch_trials_to_test = [i[1] for i in self.quant_test if i[0] == index]
                # for test_trial in list_of_sub_batch_trials_to_test:
                for test_trial in range(predictions.shape[1]):
                    sequence_save_path = qual_save_path + "batch_" + str(index) + "sub_batch_" + str(test_trial) + "/"
                    try:
                        os.mkdir(sequence_save_path)
                    except FileExistsError or FileNotFoundError:
                        pass

                    for i in range(self.n_future - 1):
                        # plt.figure(1)
                        # if self.occlusion_test and self.qual_tactile_analysis:
                        #     f, axarr = plt.subplots(1, 4, constrained_layout=True)
                        # elif self.qual_tactile_analysis or self.occlusion_test:
                        #     f, axarr = plt.subplots(1, 3, constrained_layout=True)
                        # else:
                        #     f, axarr = plt.subplots(1, 2, constrained_layout=True)
                        # axarr[0].set_title("predictions: t_" + str(i))
                        # axarr[0].imshow(np.array(predictions[i][test_trial].permute(1, 2, 0).cpu().detach()))
                        # axarr[1].set_title("ground truth: t_" + str(i))
                        # axarr[1].imshow(np.array(images[i+self.n_past][test_trial].permute(1, 2, 0).cpu().detach()))
                        # if self.occlusion_test:
                        #     axarr[2].set_title("Occluded input: t_" + str(i))
                        #     axarr[2].imshow(np.array(images_occ[i+self.n_past][test_trial].permute(1, 2, 0).cpu().detach()))
                        # if self.qual_tactile_analysis:
                        #     axarr[-1].set_title("Tactile sequence:")
                        #     # create tactile plot here:
                        #     test_taxel1 = 12
                        #     test_taxel2 = 25
                        #     test_taxel3 = 40
                        #     axarr[-1].set_ylim((0,1))
                        #     axarr[-1].plot([i for i in range(tactile.shape[0])], tactile[:, test_trial, test_taxel1].cpu().detach(), label="GT taxel " + str(test_taxel1))
                        #     axarr[-1].plot([i for i in range(tactile.shape[0])], [None for i in range(tactile.shape[0] - tactile_predictions.shape[0])] + list(tactile_predictions[:, test_trial, test_taxel1].cpu().detach()), label="Pred taxel " + str(test_taxel1))
                        #     axarr[-1].plot([i for i in range(tactile.shape[0])], tactile[:, test_trial, test_taxel2].cpu().detach(), label="GT taxel " + str(test_taxel2))
                        #     axarr[-1].plot([i for i in range(tactile.shape[0])], [None for i in range(tactile.shape[0] - tactile_predictions.shape[0])] + list(tactile_predictions[:, test_trial, test_taxel2].cpu().detach()), label="Pred taxel " + str(test_taxel2))
                        #     axarr[-1].plot([i for i in range(tactile.shape[0])], tactile[:, test_trial, test_taxel3].cpu().detach(), label="GT taxel " + str(test_taxel3))
                        #     axarr[-1].plot([i for i in range(tactile.shape[0])], [None for i in range(tactile.shape[0] - tactile_predictions.shape[0])] + list(tactile_predictions[:, test_trial, test_taxel3].cpu().detach()), label="Pred taxel " + str(test_taxel3))
                        #     # axarr[-1].legend(loc="lower right")
                        #     axarr[-1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
                        #     # axarr[-1].
                        #     np.save(sequence_save_path + "gt_tactile_time_step", np.array(tactile[:, test_trial].cpu().detach()))
                        #     np.save(sequence_save_path + "pred_tactile_time_step", np.array(tactile_predictions[:, test_trial].cpu().detach()))
                        # plt.savefig(sequence_save_path + "scene_time_step_" + str(i) + ".png")
                        np.save(sequence_save_path + "pred_scene_time_step_" + str(i), np.array(predictions[i][test_trial].permute(1, 2, 0).cpu().detach()))
                        np.save(sequence_save_path + "gt_scene_time_step_" + str(i), np.array(images[i+self.n_past][test_trial].permute(1, 2, 0).cpu().detach()))
                        if self.occlusion_test:
                            np.save(sequence_save_path + "occluded_scene_time_step_" + str(i), np.array(images_occ[i+self.n_past][test_trial].permute(1, 2, 0).cpu().detach()))
                        # plt.close('all')

    def format_and_run_batch(self, batch_features, test, cut_size=0, qualitative=False):
        mae, kld, mae_tactile, predictions, tactile_predictions, tactile, scene_occ, images_occ = None, None, None, None, None, None, None, None
        if self.model_name == "SVG":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images, actions=action, test=test)
            if cut_size !=0:
                print("cut_size ", cut_size)
                print("predictions.shape ", predictions.shape)
                predictions = predictions[:, :cut_size]
            if not qualitative:
                scene_MAE, tactile_MAE = self.calculate_losses(images[self.n_past:], predictions)

        elif self.model_name == "VG" or self.model_name == "VG_MMMM":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, predictions = self.model.run(scene=images, actions=action, test=test)
            if cut_size != 0:
                predictions = predictions[:, :cut_size]
            if not qualitative:
                scene_MAE, tactile_MAE = self.calculate_losses(images[self.n_past:], predictions)

        elif self.model_name == "SVG_TC" or self.model_name == "SVG_TC_TE" or self.model_name == "SVG_TE":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images, tactile=tactile, actions=action, gain=self.gain, test=test, stage=self.stage)
            if cut_size != 0:
                predictions = predictions[:, :cut_size]
            if not qualitative:
                scene_MAE, tactile_MAE = self.calculate_losses(images[self.n_past:], predictions)

        elif self.model_name == "SVG_TC_occ" or self.model_name == "SVG_TC_TE_occ":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            images_occ  = batch_features[6].permute(1, 0, 4, 3, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images_occ, tactile=tactile, actions=action, scene_gt=images, gain=self.gain, test=test, stage=self.stage)
            if cut_size != 0:
                predictions = predictions[:, :cut_size]
            if not qualitative:
                scene_MAE, tactile_MAE = self.calculate_losses(images[self.n_past:], predictions)

        if qualitative:
            return predictions, tactile_predictions, images, tactile, images_occ

        return scene_MAE, tactile_MAE, predictions, tactile_predictions, images_occ

    def calculate_losses(self, groundtruth_scene, prediction_scene, groundtruth_tactile=None, predictions_tactile=None):
        scene_losses, tactile_losses = [],[]
        # scene:
        for criterion in [nn.L1Loss(), nn.MSELoss(), PSNR(), SSIM(window_size=self.image_width)]:  #, SSIM(window_size=self.image_width)]:
            batch_loss = []
            for i in range(prediction_scene.shape[0]):
                batch_loss.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)
            scene_losses.append(sum(batch_loss) / len(batch_loss))
            for itr in range(prediction_scene.shape[0]):
                scene_losses.append(criterion(prediction_scene[itr], groundtruth_scene[itr]).cpu().detach().data)
        # tactile:
        if groundtruth_tactile is not None:
            if self.tactile_size == 48:
                criterions = [nn.L1Loss(), nn.MSELoss()]
            else:
                criterions = [nn.L1Loss (), nn.MSELoss (), PSNR (), SSIM(window_size=self.image_width)]
            for criterion in criterions:
                batch_loss = []
                for i in range(prediction_scene.shape[0]):
                    batch_loss.append(criterion(predictions_tactile[i], groundtruth_tactile[i]).cpu().detach().data)
                tactile_losses.append(sum(batch_loss) / len(batch_loss))
                tactile_losses.append(criterion(predictions_tactile[0], groundtruth_tactile[0]).cpu().detach().data)
                tactile_losses.append(criterion(predictions_tactile[2], groundtruth_tactile[2]).cpu().detach().data)
                tactile_losses.append(criterion(predictions_tactile[4], groundtruth_tactile[4]).cpu().detach().data)
                tactile_losses.append(criterion(predictions_tactile[:,:,0], groundtruth_tactile[:,:,0]).cpu().detach().data)  # Shear X
                tactile_losses.append(criterion(predictions_tactile[:,:,1], groundtruth_tactile[:,:,1]).cpu().detach().data)  # Shear Y
                tactile_losses.append(criterion(predictions_tactile[:,:,2], groundtruth_tactile[:,:,2]).cpu().detach().data)  # Normal Z

        return scene_losses, tactile_losses


@click.command()
@click.option('--model_name', type=click.Path(), default="SVG", help='Set name for prediction model, SVG, SVTG_SE, SVG_TC, SVG_TC_TE, SPOTS_SVG_ACTP')
@click.option('--model_stage', type=click.Path(), default="", help='what stage of model should you test? BEST, stage1 etc.')
@click.option('--model_folder_name', type=click.Path(), default="model_31_05_2022_14_02", help='Folder name where the model is stored')
@click.option('--test_folder_name', type=click.Path(), default="train_formatted", help='Folder name where the test data is stored, test_no_new_formatted, test_novel_formatted')
@click.option('--scalar_folder_name', type=click.Path(), default="scaler", help='Folder name where the test data is stored, test_no_new_formatted, test_novel_formatted')
@click.option('--quant_analysis', type=click.BOOL, default=True, help='Perform quantitative analysis on the test data')
@click.option('--qual_analysis', type=click.BOOL, default=True, help='Perform qualitative analysis on the test data')
@click.option('--qual_tactile_analysis', type=click.BOOL, default=False, help='Perform qualitative analysis on the test tactile data')
@click.option('--test_sample_time_step', type=click.Path(), default="[1, 2, 10]", help='which time steps in prediciton sequence to calculate performance metrics for.')
@click.option('--model_name_save_appendix', type=click.Path(), default = "", help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--test_data_dir', type=click.Path(), default = "/home/willmandil/Robotics/Data_sets/PRI/shelf_can_aid/", help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--scaler_dir', type=click.Path(), default = "/home/willmandil/Robotics/Data_sets/PRI/shelf_can_aid/", help = "What to add to the save file to identify the model as a specific subset, _1c")
def main(model_name, model_stage, model_folder_name, test_folder_name, scalar_folder_name, quant_analysis, qual_analysis, qual_tactile_analysis, test_sample_time_step, model_name_save_appendix, test_data_dir, scaler_dir):
    # Lab PC
    model_save_path = "/home/willmandil/Robotics/tactile_conditioned_video_prediction/models/saved_models/" + model_name + "/" + model_folder_name + "/"
    test_data_dir   =  test_data_dir + test_folder_name + "/"
    scaler_dir      = scaler_dir + model_folder_name + "/"

    data_save_path = model_save_path + "performance_data/"
    try:
        os.mkdir(data_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    model_save_name = model_name + "_model"
    print(model_save_name)
    print(test_folder_name)
    quant_test = np.array([[0, i] for i in range(0, 2)])

    MT = UniversalTester(data_save_path, model_save_path, test_data_dir, scaler_dir, model_save_name, model_folder_name, test_folder_name, model_stage, quant_analysis, qual_analysis, qual_tactile_analysis, quant_test, model_name_save_appendix)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    main()