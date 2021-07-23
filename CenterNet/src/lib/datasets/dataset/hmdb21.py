from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import os.path
import torch
import cv2, pickle
from collections import defaultdict

import torch.utils.data as data

CLASSES = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 
'pick', 'pour', 'pullup', 'push', 'run','shoot_ball', 'shoot_bow', 
'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']


def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        trainvideos.append(vid)
    return trainvideos


def make_lists(rootpath, imgtype, split=1, fulltest=False, subsample = False):
    imagesDir = os.path.join(rootpath, imgtype)
    assert (subsample == False)
    splitfile = os.path.join(rootpath,'splitfiles', 'trainlist{:02d}.txt'.format(split))
    trainvideos = readsplitfile(splitfile)
    trainlist = []
    testlist = []
    test_nframes = defaultdict(int)

    with open(os.path.join(rootpath, 'splitfiles','hmdb_annots.pkl'),'rb') as fff:
        database = pickle.load(fff)

    train_action_counts = np.zeros(len(CLASSES), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES), dtype=np.int32)

    #4500ratios = np.asarray([1.1, 0.8, 4.7, 1.4, 0.9, 2.6, 2.2, 3.0, 3.0, 5.0, 6.2, 2.7,
    #                     3.5, 3.1, 4.3, 2.5, 4.5, 3.4, 6.7, 3.6, 1.6, 3.4, 0.6, 4.3])
    # ratios = np.asarray([1.03, 0.75, 4.22, 1.32, 0.8, 2.36, 1.99, 2.66, 2.68, 4.51, 5.56, 2.46, 3.17, 2.76, 3.89, 2.28, 4.01, 3.08, 6.06, 3.28, 1.51, 3.05, 0.6, 3.84])
    ratios = [0.23, 0.23,  0.23, 0.21, 0.23, 0.16, 0.14, 0.18, 0.24, 0.30,
                0.22,  0.18, 0.15, 0.27, 0.22, 0.17, 0.15, 0.26, 0.25,  0.21,0.2 ]#[1]*21s
    #ratios = np.ones_like(ratios) #TODO:uncomment this line and line 155, 156 to compute new ratios might be useful for JHMDB21
    video_list = []
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        actidx = database[videoname]['label']
        istrain = True
        step = ratios[actidx]
        numf = database[videoname]['numf']
        lastf = numf-1
        if videoname not in trainvideos:
            istrain = False
            step = max(1, ratios[actidx])
        if fulltest:
            step = 1
            lastf = numf

        annotations = database[videoname]['annotations']
        num_tubes = len(annotations)

        tube_labels = np.zeros((numf,num_tubes),dtype=np.int16) # check for each tube if present in
        tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]
        for tubeid, tube in enumerate(annotations):
            # print('numf00', numf, tube['sf'], tube['ef'])
            for frame_id, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube
                label = tube['label']
                assert actidx == label, 'Tube label and video label should be same'
                box = tube['boxes'][frame_id, :]  # get the box as an array
                box = box.astype(np.float32)
                box[2] += box[0]  #convert width to xmax
                box[3] += box[1]  #converst height to ymax
                tube_labels[frame_num, tubeid] = 1 #label+1  # change label in tube_labels matrix to 1 form 0
                tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists

        possible_frame_nums = np.arange(0, lastf, step)
        # print('numf',numf,possible_frame_nums[-1])
        for frame_num in possible_frame_nums: # loop from start to last possible frame which can make a legit sequence
            frame_num = int(frame_num)
            check_tubes = tube_labels[frame_num,:]

            if np.sum(check_tubes)>0:  # check if there aren't any semi overlapping tubes
                all_boxes = []
                labels = []
                image_name = os.path.join(imagesDir, videoname, '{:05d}.png'.format(frame_num+1))
                #label_name = rootpath + 'labels/' + videoname + '/{:05d}.txt'.format(frame_num + 1)
                # assert os.path.isfile(image_name), 'Image does not exist'+image_name
                for tubeid, tube in enumerate(annotations):
                    label = tube['label']
                    if tube_labels[frame_num, tubeid]>0:
                        box = np.asarray(tube_boxes[frame_num][tubeid])
                        all_boxes.append(box)
                        labels.append(label)

                if istrain: # if it is training video
                    trainlist.append([vid, frame_num+1, np.asarray(labels), np.asarray(all_boxes)])
                    train_action_counts[actidx] += 1 #len(labels)
                else: # if test video and has micro-tubes with GT
                    testlist.append([vid, frame_num+1, np.asarray(labels), np.asarray(all_boxes)])
                    test_nframes[vid] += 1 
                    test_action_counts[actidx] += 1 #len(labels)
            elif fulltest and not istrain: # if test video with no ground truth and fulltest is trues
                testlist.append([vid, frame_num+1, np.asarray([0]), np.zeros((1,4))])
                test_nframes[vid] += 1 

    # for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
    #     print('train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES[actidx]))

    newratios = train_action_counts/5000
    #print('new   ratios', newratios)
    # line = '['
    # for r in newratios:
    #     line +='{:0.2f}, '.format(r)
    # print(line+']')
    print('Trainlistlen', len(trainlist), ' testlist ', len(testlist))

    return trainlist, testlist, video_list, test_nframes



class HMDB21(data.Dataset):
  num_classes = 21
  default_resolution = [256, 256]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, root, split, input_type='rgb', full_test = True, subsample = False):
    super(HMDB21, self).__init__()
    self.data_dir = root
    self.input_type = input_type
    input_type = input_type+'-images'
    self.root = root
    self.CLASSES = CLASSES
    self.image_set = split
    anot_name = 'fulltest' if full_test else 'parttest'
    annotations_file = '-'.join(('hmdb21', split, 'rgb-images', anot_name))
    self.coco_annots_path = os.path.join(root, 'splitfiles', annotations_file + '.json') 
    self._imgpath = os.path.join(root, input_type)
    self._annopath = os.path.join(root, 'labels/', '%s.txt')
    trainlist, testlist, video_list, test_nframes = make_lists(root, input_type, split=1, fulltest=full_test, subsample = subsample)
    self.video_list = video_list
    if self.image_set == 'train':
        self.ids = trainlist
    elif self.image_set == 'test':
        self.ids = testlist
        self.test_nframes = test_nframes
    else:
        print('specify correct subset ')
    
    self.coco = coco.COCO(self.coco_annots_path)
    
    self.max_objs = 128
    self.class_name = ['__background__', 'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 
        'pick', 'pour', 'pullup', 'push', 'run','shoot_ball', 'shoot_bow', 
        'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
    self.extension = '.png'


    self._valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
      14, 15, 16, 17, 18, 19, 20, 21]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing hmdb21 {} data.'.format(split))
    self.num_samples = self.__len__()

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    '''
    Used to convert the evaluation formats when saving the detection results
    '''  
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.4f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return len(self.ids)

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

  def run_eval_without_save(self, results, logger):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    detections = self.convert_eval_format(results)
    coco_dets = self.coco.loadRes(detections)
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize(logger)