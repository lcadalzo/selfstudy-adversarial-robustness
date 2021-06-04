# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Placeholder for L_{inf} attack."""

import common.framework
from defense_discretize.task_definition import LINF_THRESHOLD as eps
import numpy as np
import torch
from torch import nn


def get_thermometer_gradient(model, x, y):
    x_tensor = torch.tensor(x, requires_grad=True)
    shape = x_tensor.shape
    y = torch.LongTensor(y)
    thresholds = np.arange(0, 1, 0.05) + 0.05
    threshold_tensor = torch.tensor(thresholds, dtype=x_tensor.dtype)
    x_sub_threshold = (
        -x_tensor[:, :, None, :, :] + threshold_tensor[None, None, :, None, None]
    )
    x_reshaped = torch.reshape(
        x_sub_threshold, [-1, shape[1] * len(thresholds), shape[2], shape[3]]
    )
    x_sigmoid = torch.sigmoid(x_reshaped * 1000)
    y_pred = model.convnet(x_sigmoid)
    loss = nn.NLLLoss()(torch.log(y_pred) + 1e-14, y)
    loss.backward()
    gradient = x_tensor.grad.detach().numpy()
    return gradient


class LinfAttack(common.framework.Attack):
    def attack(self, model, x, y):
        x = torch.tensor(x)
        x_orig = torch.tensor(x)

        steps = 10
        eps_step = eps / 2
        for i in range(steps):
            print(i)
            x.requires_grad = True
            gradient = get_thermometer_gradient(model, x, y)
            x = x.detach() + np.sign(gradient) * eps_step
            x = torch.max(torch.min(x, x_orig + eps), x_orig - eps)
            x = torch.clip(x, 0.0, 1.0)

        return x.detach().numpy()


# # If writing attack which operates on batch of examples is too complicated
# # then remove LinfAttack and uncommend LinfAttackNonBatched from below:
#
# class LinfAttackNonBatched(common.framework.NonBatchedAttack):
#
#     def attack_one_example(self, model, x,  y):
#         # TODO: Write your attack code here
#         # You can query model by calling `model(x)`
#
#         return x
