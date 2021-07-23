from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os,sys

import torch
from torch import nn
import torch.utils.data as data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
# from data import v2, UCF24Detection, AnnotationTransform, detection_collate, CLASSES, BaseTransform

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  # train_dataset, val_dataset, epoch_size, dataset_opts = load_datasets(opt.batch_size)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  data_str = 'dataset: {}, task: {}, input_type: {}, double: True, gpu: {}, epochs: {}\n'
  print(data_str.format(opt.dataset, opt.task, opt.input_type, opt.gpus, opt.num_epochs))
  logger.write(data_str.format(opt.dataset, opt.task, opt.input_type, opt.gpus, opt.num_epochs))

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  ###Replace the first layer to take 6 channels
  model.base.base_layer = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True))

  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, os.path.join('lib', 'data', opt.dataset) , 'test', opt.input_type), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, os.path.join('lib', 'data', opt.dataset), 'train', opt.input_type), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  if opt.test:
    _, preds = trainer.val(0, train_loader)
    train_loader.dataset.run_eval(preds, opt.save_dir)
    return

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      logger.write('\n')
      val_loader.dataset.run_eval_without_save(preds, logger)
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)