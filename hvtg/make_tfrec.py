import json
import os
import sys
import math
import time
import config
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

def sample_feat(feat, max_l):
    len_this = len(feat)
    idxs = np.linspace(0, len_this, max_l, endpoint=False, dtype=np.int32)
    feat = feat[idxs]
    return feat

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def to_tfexample(feats):
    feature = {}
    for k in feats:
        p = feats[k]
        if type(p) is np.ndarray:
            feature[k] = _bytes_feature(p.tobytes())
        elif type(p) is int:
            feature[k] = _int64_feature(p)
        elif type(p) is str:
            feature[k] = _bytes_feature(p.encode())
        else:
            raise Exception('type not included')
    return tf.train.Example(
        features=tf.train.Features(feature=feature))

class Dataset(object):
    def __init__(self, args, mode):
        self.args = args
        self.data_p = Path(args.data_dir)
        if mode == 'train':
            self.annos = pickle.load(open(self.data_p/(args.train_set+'_sents.pkl'),'rb'))
            random.shuffle(self.annos)
        elif mode == 'test':
            self.annos = pickle.load(open(self.data_p/(args.test_set+'_sents.pkl'),'rb'))
        elif mode == 'infer':
            self.annos = pickle.load(open(self.data_p/'infer.pkl','rb'))
        self.ids = set([x['vid'] for x in self.annos])

        self.rois = pickle.load(open(self.data_p/'roi_db_npy.pkl', 'rb'))
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

        roi = self.rois[vid]
        #adj = self.adj[vid]
        MAXL = self.args.max_feat_len
        feats = []
        for fn in self.args.feat_name.split(','):
            feat = np.load(self.data_p/'features'/fn/(vid+'.npy'))
            assert len(roi) == len(feat)
            feat = sample_feat(feat, MAXL)
            feats.append(feat)
        feats = np.concatenate(feats, axis=-1)

        lbl = [seg[0]*1./seg[2], seg[1]*1./seg[2]]
        lbl = np.array(lbl, dtype=np.float32)

        M = feats.shape[0]
        m = np.zeros((M,), dtype=np.float32)
        l = max(0, int(lbl[0]*M)-1)
        r = min(M, int(lbl[1]*M)+1)
        m[l:r] = 1.

        ret =  {'feats':feats,
                'sent':sent,
                'lbl':lbl,
                'm':m}
        ret['vid'] = vid
        ret['idx'] = idx
        return ret

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


if __name__ == "__main__":
    args, _ = config.get_args()
    ds = Dataset(args, args.mode)

    num_shards = 100
    n_per_shard = len(ds) // num_shards
    for s in tqdm(range(num_shards)):
        st_idx = s * n_per_shard
        ed_idx = min((s+1) * n_per_shard, len(ds))

        filename = os.path.join(args.output_dir, "data-%04d-%04d.tfrecord" % (s, num_shards))
        with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
            for i in range(st_idx, ed_idx):
                feats = ds[i]
                example = to_tfexample(feats)
                tfrecord_writer.write(example.SerializeToString())
