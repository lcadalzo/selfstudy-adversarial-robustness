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
from common.networks import AllConvModelTorch

import torch

from defense_jump.model import Defense
from defense_jump.task_definition import LINF_THRESHOLD as eps

MODEL_PATH = "checkpoints/jump/final_checkpoint-1"


class DefenseTorchProxy(Defense):
    def __init__(self):
        import torch

        class Jump(torch.nn.Module):
            def forward(self, x):
                activation = torch.nn.LeakyReLU(0.2)
                x = activation(x)
                x += (x > 0).float() * 5
                return x

        class JumpApprox(torch.nn.Module):
            def forward(self, x):
                activation1 = torch.nn.LeakyReLU(0.2)
                activation2 = torch.nn.Sigmoid()
                x = activation1(x) + activation2(100000 * x) * 5
                return x

        self.convnet = AllConvModelTorch(
            num_classes=10,
            num_filters=64,
            input_shape=[3, 32, 32],
            activation=JumpApprox(),
        )
        self.convnet.load_state_dict(torch.load(MODEL_PATH + ".torchmodel"))


proxy = torch.nn.Sequential(*DefenseTorchProxy().convnet.layers[:-1])


class LinfAttack(common.framework.Attack):
    def attack(self, model, x, y):
        x_orig = torch.tensor(x)
        x = torch.tensor(x)
        y = torch.LongTensor(y)
        loss = torch.nn.CrossEntropyLoss()
        # proxy = torch.nn.Sequential(*model.convnet.layers[:-1])

        steps = 10
        eps_step = eps / 2

        for i in range(steps):
            print(i)
            x.requires_grad = True
            output = proxy(x)
            l = loss(output, y)
            l.backward()

            x = x.detach() + torch.sign(x.grad) * eps_step
            # x = np.clip(x, x_orig - eps, x_orig + eps)
            x = torch.max(torch.min(x, x_orig + eps), x_orig - eps)
            x = torch.clip(x, 0.0, 1.0)

        return x.numpy()


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
