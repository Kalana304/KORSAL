from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .double import DoubleTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .SMDouble import SMDoubleTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'ctdet_smd': CtdetTrainer,
  'double': DoubleTrainer,
  'multi_pose': MultiPoseTrainer, 
  'doubleSM' : SMDoubleTrainer
}
