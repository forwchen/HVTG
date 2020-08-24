#!/usr/bin/env python
# coding: utf-8

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import pylab
import time
import os
import sys
import copy
import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# Change dir to caffe root or prototxt database paths won't work wrong
import os
os.chdir('..')


# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import caffe


data_path = './data/genome/1600-400-20'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())
# Check object extraction
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
import cv2

GPU_ID = 0  # if we have multiple GPUs, pick one
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
net = None
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = None
init = False

def init_net():
    global init
    global net
    if not init:
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        init = True


def ext_obj_feat(im_file):
    ###########################
    # Similar to get_detections_from_im
    conf_thresh=0.4
    min_boxes=16
    max_boxes=32 # 32

    im = cv2.imread(im_file)
    #print(im.shape)
    max_s = max(im.shape[0], im.shape[1])
    max_s_limit = 480. # 480
    #if max_s > max_s_limit:
    shrink_h = int(im.shape[0] * (max_s_limit / max_s))
    shrink_w = int(im.shape[1] * (max_s_limit / max_s))
    im = cv2.resize(im, (shrink_w, shrink_h))
        #print(im.shape)


    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data.copy()
    attr_prob = net.blobs['attr_prob'].data.copy()
    #pool5 = net.blobs['pool5_flat'].data

    #print(net.blobs['res4b22'].data.shape) # (1, 1024, 36, 63)
    feat_data = net.blobs['res4b22'].data.copy()

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]
    ############################

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]

    boxes = cls_boxes[keep_boxes]
    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)

    #for i in range(len(keep_boxes)):

    bboxes = []
    tags = []

    for i in range(min_boxes):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i]+1]
        if attr_conf[i] > attr_thresh:
            cls = attributes[attr[i]+1] + " " + cls
        #print(bbox, cls)

        bbox[[1,3]] /= h
        bbox[[0,2]] /= w
        bboxes.append(bbox) # x1,y1,x2,y2
        tags.append(cls)

    #print 'boxes=%d' % (len(keep_boxes))
    return feat_data, np.array(bboxes), np.array(tags)


#conn = rpyc.connect('localhost', 18870+int(sys.argv[1]), config = rpyc.core.protocol.DEFAULT_CONFIG)

#im_file_sample = '/home/forwchen/mnt/s1/store/anet_frames/1vTHJMMPZN0/004160.jpg'
#im_file_sample = '/home/forwchen/mnt/s1/store/anet_frames/MmipoQF8EJs/001536.jpg'
import ipdb
ims = [os.path.join(sys.argv[2], x) for x in sorted(os.listdir(sys.argv[2]))]
import h5py
import pickle
from tqdm import tqdm
#db = h5py.File(sys.argv[3], 'w')
db = {}
db_path = sys.argv[3]+'.pkl'

if os.path.exists(db_path):
    db = pickle.load(open(db_path, 'rb'))
    print('restore db')
    print('has %d frames' % (len(db)//2,))
cnt = 0
for im_file in tqdm(ims, ncols=64):

    vid = im_file.split('/')[-2]
    fid = im_file.split('/')[-1].split('.')[0]

    ifid = int(fid)
    if not ((ifid-1)%4 == 0):
        continue

    k = vid+'/'+fid
    if (k+'/box') in db:
        continue
    init_net()
    f,b,t = ext_obj_feat(im_file)

    db[k+'/box'] = b
    db[k+'/tag'] = t
    cnt += 1
    #if cnt % 20 == 0:
    pickle.dump(db, open(db_path, 'wb'))

#pickle.dump(db, open(db_path, 'wb'))
