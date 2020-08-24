import json
import os
import sys
import math
import time
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
from pathlib import Path
import argparse
import pickle
import ipdb
import tensorflow as tf
#tf.set_random_seed(1234)
from utils import *
from multiprocessing import Pool
from tqdm import tqdm

logger = get_logger()

def pad_feat(f, max_l):
    return np.pad(f, [(0,max_l-f.shape[0])] + [(0,0)]*(f.ndim-1), mode='constant')

def sample_feat(feat, max_l):
    len_this = len(feat)
    idxs = np.linspace(0, len_this, max_l, endpoint=False, dtype=np.int32)
    feat = feat[idxs]
    return feat

def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou


class Dataset(object):
    def __init__(self, args, mode):
        self.args = args
        self.data_p = Path(args.data_dir)
        if mode == 'train':
            self.annos = pickle.load(open(self.data_p/(args.train_set+'_sents.pkl'),'rb'))
            random.shuffle(self.annos)
        elif mode == 'test':
            self.annos = pickle.load(open(self.data_p/(args.test_set+'_sents.pkl'),'rb'))
        self.ids = set([x['vid'] for x in self.annos])

        self.mode = mode
        logger.info(f"[*] num samples: {len(self)}")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.annos)

    def __getitem__(self, index):
        'Generates one sample of data'
        anno = self.annos[index]
        vid = anno['vid']
        seg = anno['seg']
        sent = anno['sent']
        idx = anno['id']

        MAXL = self.args.max_feat_len
        feats = []
        for fn in self.args.feat_name.split(','):
            feat = np.load(self.data_p/'features'/fn/(vid+'.npy'))
            feat = sample_feat(feat, MAXL)
            feats.append(feat)
        feats = np.concatenate(feats, axis=-1)

        if self.args.use_motion:
            mot = np.load(self.data_p/'features'/'c6'/(vid+'.npy'))
            mot = sample_feat(mot, MAXL)

        lbl = [seg[0]*1./seg[2], seg[1]*1./seg[2]]
        lbl = np.array(lbl, dtype=np.float32)

        M = feats.shape[0]
        m = np.zeros((M,), dtype=np.float32)
        l = max(0, int(lbl[0]*M)-1)
        r = min(M, int(lbl[1]*M)+1)
        m[l:r] = 1.

        ret =  (feats, sent, lbl, m)
        if self.args.use_motion:
            ret += (mot, )
        ret += (vid, idx)
        return ret

    def recall_at_iou(self, iou, x1, x2, idx):
        s = self.annos[idx]
        #n = s['duration']
        #n = s['seg'][2]

        x1 = (x1[0], x1[1])
        x2 = (x2[0], x2[1])

        iou_this = calculate_IoU(x1, x2)
        if iou_this >= iou:
            return 1.
        else:
            return 0.

    def gen_data(self):
        idxs = list(range(len(self)))
        if self.mode == 'train':
            #print('shuffling training set.')
            random.shuffle(idxs)
        for idx in idxs:
            yield self[idx]

    def get_data_type_and_shape(self):
        dp = self[0]
        type_dict = {np.dtype('float32'): tf.float32,
                     np.dtype('float16'): tf.float16,
                     np.dtype('int32'): tf.int32,
                     np.dtype('int64'): tf.int64,
                     np.dtype('int8'): tf.int8,
                     np.dtype('uint8'): tf.uint8,
                     np.dtype('bool'): tf.bool,
                     int: tf.int32,
                     float: tf.float32,
                     str: tf.string}

        types = []
        shapes = []
        for p in dp:
            if type(p) is np.ndarray:
                types.append(type_dict[p.dtype])
                shapes.append(tf.TensorShape(p.shape))
            elif type(p) in type_dict:
                types.append(type_dict[type(p)])
                shapes.append(tf.TensorShape([]))
            else:
                print(type(p))
                raise Exception('type not included')
        return tuple(types), tuple(shapes)
