from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .ctdet import CtdetTrainer

class SMDoubleModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(SMDoubleModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input']) # changed - cascaded input
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class SMDoubleTrainer(CtdetTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(SMDoubleTrainer, self).__init__(opt, model, optimizer=optimizer)
    self.model_with_loss = SMDoubleModelWithLoss(model, self.loss)
