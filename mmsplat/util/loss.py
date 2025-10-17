
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copied from: https://github.com/chen-hangyu/Thermal-Gaussian-main/blob/main/utils/loss_utils.py

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import cv2
import torch.nn as nn

def generate_adj_neighbors(image_map, k):
    C, H, W = image_map.shape
    adj = torch.zeros((C, H, W, k)).cuda()

    for i in range(C):
        image_pixel = image_map[i].detach().cpu().numpy()
        if k == 4:
            neighbors = [
                (np.roll(image_pixel, 1, axis=0), 0),
                (np.roll(image_pixel, -1, axis=0), 1),
                (np.roll(image_pixel, 1, axis=1), 2),
                (np.roll(image_pixel, -1, axis=1), 3)
            ]
        if k == 8:
            neighbors = [
                (np.roll(image_pixel, 1, axis=0), 0),
                (np.roll(image_pixel, -1, axis=0), 1),
                (np.roll(image_pixel, 1, axis=1), 2),
                (np.roll(image_pixel, -1, axis=1), 3),
                (np.roll(np.roll(image_pixel, 1, axis=0), 1, axis=1), 4),
                (np.roll(np.roll(image_pixel, 1, axis=0), -1, axis=1), 5),
                (np.roll(np.roll(image_pixel, -1, axis=0), 1, axis=1), 6),
                (np.roll(np.roll(image_pixel, -1, axis=0), -1, axis=1), 7)
            ]

        for neighbor, idx in neighbors:
            adj[i, ..., idx] = torch.tensor(neighbor).cuda()

    return adj


def smoothness_loss(image_map):
    C, H, W = image_map.shape
    adj = generate_adj_neighbors(image_map, 4)

    K = adj.shape[-1]

    loss = 0.0
    for i in range(K):
        neighbor_image = adj[..., i]
        diff = image_map - neighbor_image
        loss += torch.sum(torch.abs(diff))

    loss /= (C * H * W)

    return loss


def unit_norm_regularization_loss(feature_vectors: torch.Tensor, norm_target: float = 1.0) -> torch.Tensor:
    """
    Computes a soft regularization loss encouraging feature vectors to have a specified norm (default = 1).

    Args:
        feature_vectors (torch.Tensor): Tensor of shape [batch_size, feature_dim].
        norm_target (float): Desired norm of each feature vector (default is 1.0).

    Returns:
        torch.Tensor: Scalar regularization loss.
    """
    # Compute the L2 norm of each vector in the batch
    vector_norms = torch.norm(feature_vectors, p=2, dim=-1)

    # Compute mean squared deviation from the target norm
    norm_penalty = ((vector_norms - norm_target) ** 2).mean()

    return norm_penalty


def cosine_neighbor_loss_batched(features):
    """
    Computes cosine similarity loss between center features and their neighbors.

    Args:
        features (torch.Tensor): Tensor of shape (N, k, D), where
            - N = number of points,
            - k = number of neighbors (first is the center),
            - D = feature dimension.

    Returns:
        torch.Tensor: Scalar cosine similarity loss.
    """
    centers = features[:, 0, :]  # (N, D)
    neighbors = features[:, 1:, :]  # (N, k-1, D)

    # Expand center to (N, k-1, D) to match neighbor shape
    centers_expanded = centers.unsqueeze(1).expand_as(neighbors)

    # Compute cosine similarity for each neighbor
    cos_sim = F.cosine_similarity(centers_expanded, neighbors, dim=-1)  # (N, k-1)

    # Loss: 1 - cosine similarity
    loss = 1 - cos_sim

    # Return mean loss over all pairs
    return loss.mean()