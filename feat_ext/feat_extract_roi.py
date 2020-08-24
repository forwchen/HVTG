import os
import sys
import math
import ipdb
import h5py
import GPUtil
import random
import argparse
import importlib
import numpy as np

from functools import partial
from tensorpack import dataflow
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim
import nets_factory
import multiprocessing
import utils
import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="the network model to use")
parser.add_argument("frames", type=str, help="Frames in a list")
parser.add_argument("feat_db", type=str, help="HDF5 File storing the features")
parser.add_argument("port", type=int, help="ROI server port")
args = parser.parse_args()

###############################################################################
# Set up GPU and session
#########################
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxMemory = 0.001)
if len(deviceIDs) == 0:
    print('No available GPU.')
    sys.exit(0)
else:
    gpu_id = str(deviceIDs[0])

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print(gpu_id)

sess_cfg = tf.ConfigProto()
sess_cfg.gpu_options.allow_growth = True
sess = tf.Session(config=sess_cfg)
###############################################################################
# Set up dataset
########################
cnn_cfg = importlib.import_module('configs.'+args.model_name)

prep_img = partial(utils._prep_img, cfg=cnn_cfg)

nproc = 4

# a list of images
frames = [l.strip() for l in open(args.frames, 'r').readlines()]
frames = [tuple(('/'.join(l.split('/')[-2:]), l)) for l in frames]
ds = dataflow.DataFromList(frames, shuffle=False)

ds = dataflow.MultiProcessMapDataZMQ(ds, nr_proc=nproc, map_func=prep_img, buffer_size=1000, strict=True)

ds.reset_state()

def wrapper():
    for dp in ds.get_data():
        if dp[0].startswith('@invalid_img'):
            print(dp[0])
            continue
        yield tuple(dp)      # tf.data expects tuple input

tf_ds = tf.data.Dataset.from_generator(
    wrapper,
    output_types=(tf.string, tf.float32),
    output_shapes=(tf.TensorShape([]),
                   tf.TensorShape([None, None, 3])),
    )

tf_ds = tf_ds.repeat(1).batch(cnn_cfg.batch_size)
tf_ds = tf_ds.prefetch(buffer_size=1024)

keys, images = tf_ds.make_one_shot_iterator().get_next()

########################
### CNN model       ####
########################

network_fn = nets_factory.get_network_fn(
    args.model_name,
    num_classes=None,
    is_training=False)
logits, endpoints = network_fn(images)

for e in endpoints.keys():
    print(e, endpoints[e].shape, endpoints[e].op.name)

ext_layer = endpoints[cnn_cfg.layer]

ckpt = cnn_cfg.ckpt

variables_to_restore = slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, ckpt)

########################
### Start Extracting ###
########################

db_file = args.feat_db
if not db_file.endswith('.hdf5'):
    db_file += '.hdf5'

feat_db = h5py.File(db_file, 'w')

conn = rpyc.connect('localhost', 18870+args.port, config=rpyc.core.protocol.DEFAULT_CONFIG)

pbar = tqdm(total=ds.size(), ncols=64)
while True:
    try:
        feat, fids = sess.run([ext_layer, keys])
        pbar.update(len(fids))
        for k, o in zip(fids, feat):
            r = conn.root.roi(o, np.array(k.decode('ascii')))
            feat_db[k] = np.squeeze(np.asarray(r))
    except Exception as inst:
        print(inst)
        break

feat_db.close()
ipdb.set_trace()
pbar.close()



