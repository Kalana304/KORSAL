from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from torch import nn
import os

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
from skimage.metrics import structural_similarity as ssim
import random

class BaseDetectorSMDouble(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model.base.base_layer = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=7, stride=1,
                        padding=3, bias=False),
                nn.BatchNorm2d(16, momentum=0.1),
                nn.ReLU(inplace=True))
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

        #------------------- small motion kernels -------------------#
        self.kernels =  {  
                            0: np.array([[1,0,0], [0,0,0],[0,0,0]]),
                            1: np.array([[0,1,0], [0,0,0],[0,0,0]]),
                            2: np.array([[0,0,1], [0,0,0],[0,0,0]]),
                            3: np.array([[0,0,0], [0,0,1],[0,0,0]]),
                            4: np.array([[0,0,0], [0,0,0],[0,0,1]]),
                            5: np.array([[0,0,0], [0,0,0],[0,1,0]]),
                            6: np.array([[0,0,0], [0,0,0],[1,0,0]]),
                            7: np.array([[0,0,0], [1,0,0],[0,0,0]]),
                            8: np.array([[0,0,0], [0,1,0],[0,0,0]])     
                        }
        #------------------------------------------------------------#

    def _ssim_calc(self, var_img, ref_img, Gaussian = False, return_map = False):
        """
            this function calculates the ssim based on skimage implementation and 
            return the ssim index value/ssm based on return_map flag.

            Inputs:
                var_img     : variable image whose difference should be measured w.r.t ref_img
                ref_img     : reference image (usually prev_inp)
                Gaussian    : using gaussian kernels to smoothing else uniform kernels
                return_map  : return ssm map if true. else return ssim index value only
            
            Outputs:
                ssim iddex value and/or ssm
        """
        assert ref_img.shape == var_img.shape, f"reference shape :: {ref_img.shape}\nvariable shape :: {var_img.shape}\n"
        
        if ref_img.shape[0] == 3:
            ref_img = ref_img.transpose(1,2,0)      # converting the images to the format width x height x n_ch
            var_img = var_img.transpose(1,2,0)
        
        if return_map:
            index_value, ss_map = ssim(ref_img, var_img, multichannel=True, data_range = var_img.max()-var_img.min(), gaussian_weights = Gaussian, full = return_map)
            ss_map = ss_map.transpose(2, 0, 1)
            ss_map = ss_map.astype(np.float32)
            return index_value, ss_map
        else:
            return ssim(ref_img, var_img, multichannel=True, data_range = var_img.max()-var_img.min(), gaussian_weights = Gaussian, full = return_map)

            

    def pre_process(self, image, prev_image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width  = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        resized_prev_image = cv2.resize(prev_image, (new_width, new_height))
        
        inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
        prev_inp_image = cv2.warpAffine(
        resized_prev_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
        
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        prev_inp_image = ((prev_inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        prev_images = prev_inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        #----------------------- small motion cost volume creation -----------------------#
        if self.opt.variant == 2 and 8 in self.kernels.keys():
            # variant 1 does not use the identity frame stacked
            del self.kernels[8]
        
        smotion = np.zeros((len(self.kernels), )+inp_image.shape, dtype=np.float32)
        ssim_index = []

        for i in self.kernels.keys():
            smotion[i] = cv2.filter2D(images.transpose(1,2,0), -1, self.kernels[i]).transpose(2,0,1)
            assert smotion[i].shape[0] == 3, "wrong dimensional shift"
            #------------------ ssim calculation ------------------#
            ssim_index.append(self._ssim_calc(smotion[i], prev_images))
        
        ssim_index = np.array(ssim_index)
        ssim_id_sort = np.argsort(ssim_index)[::-1]

        if self.split == 'train':
            candidate = random.sample(list(ssim_id_sort[:3]), k=1)  # output [img_index]
        else:
            candidate = ssim_id_sort[0]     # output img_index

        if isinstance(candidate, list):
            ssm = smotion[candidate[0]]
        else:
            ssm = smotion[candidate]
        
        _, concat_candid = self._ssim_calc(ssm, prev_images, Gaussian = self.opt.use_gaussian, return_map = True)

        if self.opt.variant == 3:
            # variant 3 is using structural dissimilarity map
            concat_candid = 0.5*(1 - concat_candid)
        
        #----------------------------------------------------------------------------------#
        
        concat_images = np.concatenate((images, concat_candid), axis = 0)

        if self.opt.flip_test:
            concat_images = np.concatenate((concat_images, concat_images[:, :, :, ::-1]), axis=0)
        concat_images = torch.from_numpy(concat_images)
        meta = {'c': c, 's': s, 
                'out_height': inp_height // self.opt.down_ratio, 
                'out_width': inp_width // self.opt.down_ratio}
        return concat_images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run_from_trainer(self, batch):

        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        # debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                            # theme=self.opt.debugger_theme)
        start_time = time.time()
        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        
        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            images = batch['input']
            meta = {k: v.cpu().numpy()[0] for k, v in batch['meta'].items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            
            output, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
            
            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)
            
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)
            
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, images, results)
        
        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
                

    def run(self, image_or_path_or_tensor, meta=None):

        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        # debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                            # theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
            prev_image = None
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)
        
            path_components = image_or_path_or_tensor.split('/')
            file_name = path_components[-1]
            frame_num = int(file_name.split('.')[0])
            prev_frame_num = frame_num - 1
            if prev_frame_num < 1:
                prev_image = np.zeros_like(image)
            else:
                prev_image_path = os.path.join(*path_components[:-1], '{:05d}.jpg'.format(prev_frame_num))
                
                prev_image = cv2.imread(prev_image_path)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            prev_image = None
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True
        
        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        
        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, prev_image, scale, meta)

            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            
            output, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
        
            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)
        
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)
        
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)
        
        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
