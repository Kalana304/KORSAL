from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import random

class CTDetDataset_SMDouble(data.Dataset):
    def __init__(self):
        super(CTDetDataset_SMDouble).__init__()
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

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _transform_to_coco(self, bboxs, labels):
        anns = []
        for t in range(len(labels)):
            bbox = bboxs[t, :]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            label = labels[t]
            anns.append({'bbox': bbox, 'category_id': label + 1})
        return anns
    
    def _scale_bbox(self, bbox, i_h, i_w, h, w):
        bbox[0] = float(bbox[0])*i_w/w
        bbox[2] = float(bbox[2])*i_w/w
        bbox[1] = float(bbox[1])*i_h/h
        bbox[3] = float(bbox[3])*i_h/h
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

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


    def __getitem__(self, index):
        img_id = index
        annot_info = self.ids[index]
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        img_path = os.path.join(self._imgpath, videoname, '{:05d}{}'.format(frame_num, self.extension))

        ### To load the previous frame
        prev_frame_num = frame_num - self.opt.frame_gap
        prev_img_path = os.path.join(self._imgpath, videoname, '{:05d}{}'.format(prev_frame_num, self.extension))

        # ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        # anns = self.coco.loadAnns(ids=ann_ids)
        anns = self._transform_to_coco(annot_info[3], annot_info[2])
        num_objs = min(len(anns), self.max_objs)
        
        img = cv2.imread(img_path)
        if prev_frame_num > 0:
            prev_img = cv2.imread(prev_img_path)
        else:
            prev_img = np.zeros_like(img)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w
        
        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                prev_img = prev_img[:, ::-1, :]
                c[0] =  width - c[0] - 1


        trans_input = get_affine_transform( c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
        prev_inp = cv2.warpAffine(prev_img, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        prev_inp = (prev_inp.astype(np.float32) / 255.)

        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            color_aug(self._data_rng, prev_inp, self._eig_val, self._eig_vec)   # Color Augmentation for previous image

        inp = (inp - self.mean) / self.std
        prev_inp = (prev_inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)                # Image size n_channels x width x height
        prev_inp = prev_inp.transpose(2, 0, 1)      # Image size n_channels x width x height

        #----------------------- small motion cost volume creation -----------------------#
        if self.opt.variant == 2 and 8 in self.kernels.keys():
            # variant 1 does not use the identity frame stacked
            del self.kernels[8]
        
        smotion = np.zeros((len(self.kernels), )+inp.shape, dtype=np.float32)
        ssim_index = []

        for i in self.kernels.keys():
            smotion[i] = cv2.filter2D(inp.transpose(1,2,0), -1, self.kernels[i]).transpose(2,0,1)
            assert smotion[i].shape[0] == 3, "wrong dimensional shift"
            #------------------ ssim calculation ------------------#
            ssim_index.append(self._ssim_calc(smotion[i], prev_inp))
        
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
        
        _, concat_candid = self._ssim_calc(ssm, prev_inp, Gaussian = self.opt.use_gaussian, return_map = True)

        if self.opt.variant == 3:
            # variant 3 is using structural dissimilarity map
            concat_candid = 0.5*(1 - concat_candid)
        
        #----------------------------------------------------------------------------------#
        
        # creating the concatenated input
        concat_inp = np.concatenate((inp, concat_candid), axis = 0)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                        draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            # bbox = self._scale_bbox(bbox, input_h, input_w, height, width)
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                            ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
            
        #ret = {'input': inp, 'prev_input': prev_inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'index':img_id, 'wh': wh}
        ret = {'input': concat_inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'index':img_id, 'wh': wh}
        
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                    np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id, 'out_height':output_h, 'out_width':output_w}
            ret['meta'] = meta
        return ret