import os
import sys
import json
import GPUtil
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from datetime import datetime
from collections import defaultdict

def set_gpu(args):
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        return
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = args.num_gpu, maxMemory = 0.001)
    if len(deviceIDs) < args.num_gpu:
        print('No enough GPU.')
        sys.exit(0)
    else:
        gpu_id = ','.join([str(d) for d in deviceIDs])
        print(gpu_id)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


def clip_gradient_norms(gradients_to_variables, max_norm):
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger

logger = get_logger()



def prepare_dirs(args):

    args.model_name = "{}_{}".format(args.dataset, args.model_name)
    if args.mode == 'train':
        if os.path.exists(os.path.join(args.log_dir,args.model_name)):
            if len(args.load_path) == 0:
                raise Exception(f"Model args.model_name already exits !! give a differnt name")

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)

    for path in [args.log_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_args(args):
    param_path = os.path.join(args.model_dir, "params-%s.json"%(get_time(),))

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)
    args_path = os.path.join(args.model_dir, "args-%s"%(get_time(),))
    with open(args_path, 'w') as fp:
        fp.write(' '.join(sys.argv))


def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))
