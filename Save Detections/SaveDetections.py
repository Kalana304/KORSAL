from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __init__path                           # Initializing the paths from the current directory

import os, sys
import cv2
import time
import pickle
import argparse
import numpy as np 
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data 
import scipy.io as sio

#### Imports for CenterNet
import _init_paths
from opts import opts
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset

from layers.box_utils import decode, nms


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#--------------------------------------- arguements to be passed ---------------------------------------#
parser = argparse.ArgumentParser(description="Online Activity Detection Testing")
parser.add_argument('--dataset', default = 'ucf24', help = 'pretrained base model')
parser.add_argument('--input_type', default = 'rgb', type = str, help = 'Input tyep default rgb can take flow as well')
parser.add_argument('--task', default='doubleSM',     # change to ctdet_smd
                             help='ctdet | ddd | multi_pose | exdet | double | doubleSM')
parser.add_argument('--jaccard_threshold', default = 0.5, type = float, help = 'Min Jaccard index for matching')
parser.add_argument('--batch_size', default = 1, type = int, help = 'Batch size for evaluating')
parser.add_argument('--num_workers', default = 0, type = int, help = 'Number of workers used in dataloading')
parser.add_argument('--eval_iter', default = '120000', type = str, help = 'Number of training iterations')
parser.add_argument('--cuda', default = True, type = str2bool, help = 'Use cuda to train model')
parser.add_argument('--subsample', default = False, type = str2bool, help = 'Use cuda to train model')
parser.add_argument('--ngpu', default = 1, type = str2bool, help = 'Use cuda to train model')
parser.add_argument('--visdom', default = False, type = str2bool, help = 'Use visdom to for loss visualization')
parser.add_argument('--data_root', default = '../../CenterNet/src/lib/data/', help = 'Location of input data directory')
parser.add_argument('--save_root', default = '../../CenterNet/exp/', help = 'Location to save checkpoint models')
parser.add_argument('--results_root', default = './Centernet/results', help = 'Location to save results')
parser.add_argument('--iou_thresh', default = 0.5, type = float, help = 'Evaluation threshold')
parser.add_argument('--conf_thresh', default = 0.01, type = float, help = 'Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default = 0.45, type = float, help = 'NMS threshold')
parser.add_argument('--topk', default = 20, type = int, help = 'topk for evaluation')

## New Arguments added
parser.add_argument('--exp_id', default='coco_dla')
parser.add_argument('--frame_gap', type=int, default=1, help = 'Frame gap to be loaded. Default is current and previous.')
parser.add_argument('--variant', type = int, default=1, 
                                    help = '1 - variant 1 with identity and with similarity map'
                                           '2 - variant 2 without identity and with dissimilarity map'
                                           '3 - variant 3 with identity and with dissimilarity map')
parser.add_argument('--load_model', default='', help='path to pretrained model')
parser.add_argument('--frame_eval', default = True, type = str2bool, help = 'To evaluate the frame wise mAP score')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
#-----------------------------------------------------------------------------------------------------------#

args = parser.parse_args()

if args.input_type != 'rgb':
    args.conf_thresh = 0.05
    
if  not os.path.isdir(args.results_root):
    os.makedirs(args.results_root)
    
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def normalize(bbox, w, h):
    '''
        Normalizes the bounding box values to the height and width of the image
        Inputs:
            bbox: bounding box of format [x1, y1, x2, y2]
            w   : width
            h   : height
    
        Outputs:
            normalized_bbox : format [x1, y1, x2, y2]
    '''
    bbox[0] /= w
    bbox[2] /= w
    bbox[1] /= h
    bbox[3] /= h
    return bbox

def process_centernet_results(results, width, height, num_classes = 24):
    '''
        This function converts the results output by the detector into 
        the required format for the realtime pipeline

        Inputs: 
            results     : A dictionary containing an entry for each class
                results[class_index]    :   an array containing predictions for class_index
                                            each entry in the array is an array of length 5 of the format
                                            [x1, y1, x2, y2, conf]
                                            class_index starts from 1
                                            coordinates are absolute values (not normalized)
            width       : Width of the image
            height      : Height of the image
            num_classes : Number of Classes (Excluding the background class)

        Outputs:
            decoded_boxes : np.array - Predicted Bounding boxes: Tensor of size (n, 4), n is the number of predictions
                            Format [x1, y1, x2, y2] Normalized
            conf_scores   : np.array - Predicted class confidences: Tensor of size (n, num_classes + 1), n is the number of predictions
    '''

    conf_scores = []
    decoded_boxes = []

    for class_idx, preds in results.items():
        for pred in preds:
            conf = pred[4]
            #Normalize the predictions
            bbox = normalize(pred[:4], width, height)
            scores = np.zeros(num_classes + 1, dtype=float) #Set all other class confidence scores to zero
            scores[class_idx] = conf

            #Write the resuts into the main arrays
            conf_scores.append(scores)
            decoded_boxes.append(bbox)
    return np.array(decoded_boxes), np.array(conf_scores)

# test script for action detection and saving detected actions
def test(detector, dataset, iteration, num_classes, opt,
         thresh = 0.5, print_time = True, width = 320 , height = 240, val_steps = 250):
    '''
        This function loads frames of a video one-by-one, runs CenterNet to 
        detect the actions and save them as .mat files.
        
        Inputs:
            network         : loaded CenterNet network
            dataset         : loaded dataset instance 
            iteration       : 120000 (default)
            num_classes     : 24 (action classes)
            thresh          : IoU threshold
            print_time      : parameter to enable printing time
            
        Outputs:
            None
    '''
    print (args.data_root + args.dataset + '/')
    
    data_loader = data.DataLoader(
                                    dataset= dataset, 
                                    batch_size=1, 
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                )
    
    image_ids = dataset.ids
    nframe_vid = dataset.test_nframes
    video_list = dataset.video_list
    n_images = len(dataset)
    n_batches = len(data_loader)
    
    # Changes to save detection outputs as .mat files
    frame_save_dir = os.path.join(args.results_root, '_'.join([args.dataset,args.input_type, args.exp_id, '100dets']))
    print ("Detection outputs will be saved to {}".format(frame_save_dir))
    
    if not os.path.exists(frame_save_dir):
        os.makedirs(frame_save_dir)

    torch.cuda.synchronize()
    ts = time.perf_counter()
    
    print('Number of images: {}\nNumber of batches: {}\n'.format(n_images,n_batches))
    
    with torch.no_grad():
        for iter_id, batch in enumerate(data_loader):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
                       
            #------------ postprocessing output from centernet ------------#
            prediction = detector.run_from_trainer(batch)
            results = prediction['results']
            decoded_boxes, conf_scores = process_centernet_results(results, width, height)
            
            if print_time and iter_id%val_steps == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf - t1))
            #--------------------------------------------------------------#
            
            index = batch['index'].cpu().numpy()
            annot_infor = image_ids[index[0]]
            n_frame, video_id, videoname = annot_infor[1]-1, annot_infor[0], video_list[annot_infor[0]]
                       
            output_dir = os.path.join(frame_save_dir, videoname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            #------------------- saving the detections as .mat file -------------------#
            output_file_name = os.path.join(output_dir, '{:05d}.mat'.format(int(n_frame + 1)))
            sio.savemat(output_file_name, mdict={'scores':conf_scores, 'loc':decoded_boxes})
            print("Done - Video :: {} - Frame :: {}\r".format(videoname, n_frame), end = '')        
    return

def main():
    means = (104, 117, 123)     # channel wise mean values
    dataset = args.dataset

    #--------------- updating the opts() from CenterNet ---------------#
    parse_opts = [  '--exp_id', args.exp_id, 
                    '--task', args.task, 
                    '--dataset', dataset, 
                    '--K', '2000', 
                    '--frame_gap', str(args.frame_gap), 
                    '--load_model', args.load_model, 
                    '--variant', str(args.variant)
                  ]
    opt = opts().parse(parse_opts)
    #-----------------------------------------------------------------#
    args.listid = '01'  
    
    for n_iter in [int(itr) for itr in args.eval_iter.split(',')]:    ## if multiple iterations given
        log_file = open(args.results_root + '/testing-{:d}.log'.format(n_iter), 'w', 1)   
        log_file.write(exp_name + '\n')                                                   
        
        # Changed to load the correct model
        checkpoint_path = os.path.join(args.save_root, opt.dataset, opt.arch, opt.input_type, opt.exp_id, opt.load_model)
        opt.load_model = os.path.abspath(checkpoint_path)

        Dataset = get_dataset(opt.dataset, opt.task)

        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        log_file.write(checkpoint_path + '\n')       

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
        
        
        #-------------------- loading the correct model from the checkpoint --------------------#
        print("Loading the pre-trained model from", opt.load_model)
        Detector = detector_factory[opt.task] 
        detector = Detector(opt)
        print("Finished loading the model") 
        
        ## creating dataset loader instance
        dataset = Dataset(opt, args.data_root + args.dataset + '/', 'test', input_type = args.input_type, full_test = True, subsample = args.subsample)
        CLASSES = dataset.CLASSES
        n_classes = len(CLASSES) + 1  # additional 1 is for the background class
        print("number of activity classes: {}".format(n_classes))
        
        ## evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        test(detector, dataset, n_iter, n_classes, opt)
    return

if __name__ == '__main__':
    main()
    