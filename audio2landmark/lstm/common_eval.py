#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import warnings
import numpy as np
from sklearn.metrics import auc
import torch

def evaluate_normalized_mean_error(predictions, groundtruth, log):
    ## compute total average normlized mean error
    assert len(predictions) == len(
        groundtruth
    ), "The lengths of predictions and ground-truth are not consistent : {} vs {}".format(
        len(predictions), len(groundtruth)
    )
    assert (
        len(predictions) > 0
    ), "The length of predictions must be greater than 0 vs {}".format(len(predictions))

    num_images = len(predictions)
    for i in range(num_images):
        c, g = predictions[i], groundtruth[i]
        assert isinstance(c, np.ndarray) and isinstance(
            g, np.ndarray
        ), "The type of predictions is not right : [{:}] :: {} vs {} ".format(
            i, type(c), type(g)
        )

    num_points = predictions[0].shape[0]
    error_per_image = np.zeros((num_images, 1))
    for i in range(num_images):
        detected_points = predictions[i]
        ground_truth_points = groundtruth[i]
        if num_points == 68:
            interocular_distance = np.linalg.norm(
                ground_truth_points[36, :2] - ground_truth_points[45, :2]
            )
        else:
            raise Exception("----> Unknown number of points : {}".format(num_points))
        dis_sum, pts_sum = 0, 0
        for j in range(num_points):
            dis_sum = dis_sum + np.linalg.norm(
                detected_points[j, :2] - ground_truth_points[j, :2]
            )
            pts_sum = pts_sum + 1
        error_per_image[i] = dis_sum / (pts_sum * interocular_distance)

    normalise_mean_error = error_per_image.mean()
    # calculate the auc for 0.07
    max_threshold = 0.07
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = (
            np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
        )
    area_under_curve07 = auc(threshold, accuracys) / max_threshold
    # calculate the auc for 0.08
    max_threshold = 0.08
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = (
            np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
        )
    area_under_curve08 = auc(threshold, accuracys) / max_threshold

    accuracy_under_007 = np.sum(error_per_image < 0.07) * 100.0 / error_per_image.size
    accuracy_under_008 = np.sum(error_per_image < 0.08) * 100.0 / error_per_image.size

    log.log(
        "Compute NME and AUC for {:} images with {:} points :: [(NME): mean={:.3f}, std={:.3f}], auc@0.07={:.3f}, auc@0.08-{:.3f}, acc@0.07={:.3f}, acc@0.08={:.3f}".format(
            num_images,
            num_points,
            normalise_mean_error * 100,
            error_per_image.std() * 100,
            area_under_curve07 * 100,
            area_under_curve08 * 100,
            accuracy_under_007,
            accuracy_under_008,
        )
    )

    for_pck_curve = []
    for x in range(0, 3501, 1):
        error_bar = x * 0.0001
        accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
        for_pck_curve.append((error_bar, accuracy))

    return normalise_mean_error, accuracy_under_008, for_pck_curve



def evaluate_normalized_mean_error_numpy(predictions, groundtruth):
    batch_size, num_images, _ = predictions.shape
    groundtruth = groundtruth.reshape((batch_size, num_images, 68, 2))
    predictions = predictions.reshape((batch_size, num_images, 68, 2))

    gt_norm = groundtruth[:, :, 36] - groundtruth[:, :, 45]
    gt_norm[gt_norm == 0] = 1e-5
    interocular_distance = np.linalg.norm(gt_norm, axis=2)

    df_norm = predictions - groundtruth
    df_norm[df_norm == 0] = 1e-5
    distance = np.linalg.norm(df_norm, axis=3)  # Sum(SQRT(dx**2 + dy**2))
    dis_mean = np.mean(distance, axis=2)
    norm_per_frame = np.divide(dis_mean, interocular_distance)
    norm_per_batch = np.mean(norm_per_frame, axis=1)
    norm_all_batch = np.mean(norm_per_batch, axis=0)
    return norm_all_batch


def evaluate_normalized_mean_error_torch(predictions, groundtruth):
    batch_size, num_images, _ = predictions.shape
    groundtruth = groundtruth.reshape((batch_size, num_images, 68, 2))
    predictions = predictions.reshape((batch_size, num_images, 68, 2))

    gt_norm = groundtruth[:, :, 36] - groundtruth[:, :, 45]
    gt_norm[gt_norm == 0] = 1e-5
    interocular_distance = torch.linalg.norm(gt_norm, dim=2)

    df_norm = predictions - groundtruth
    df_norm[df_norm == 0] = 1e-5
    distance = torch.linalg.norm(df_norm, dim=3)  # Sum(SQRT(dx**2 + dy**2))
    dis_mean = torch.mean(distance, dim=2)
    norm_per_frame = torch.div(dis_mean, interocular_distance)
    norm_per_batch = torch.mean(norm_per_frame, dim=1)
    norm_all_batch = torch.mean(norm_per_batch, dim=0)

    return norm_all_batch
