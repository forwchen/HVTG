import os
import argparse
from os.path import expanduser
home = expanduser("~")
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--net_type', type=str, default='base')
net_arg.add_argument('--cell_type', type=str, default='lstm', choices=['lstm', 'gru'])
net_arg.add_argument('--hid', type=int, default=256)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='charades')
data_arg.add_argument('--data_dir', type=str, default='./data_cha')
data_arg.add_argument('--feat_name', type=str, default='ir_roi')
data_arg.add_argument('--feat_dims', type=str, default='1536')
data_arg.add_argument('--max_sent_len', type=int, default=12)
data_arg.add_argument('--max_feat_len', type=int, default=128)
data_arg.add_argument('--train_set', type=str, default='train')
data_arg.add_argument('--test_set', type=str, default='test')
data_arg.add_argument('--n_train', type=int, default=12405)
data_arg.add_argument('--n_test', type=int, default=3720)
data_arg.add_argument('--use_tfr', type=str2bool, default=True)

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'infer'])
learn_arg.add_argument('--batch_size', type=int, default=32)
learn_arg.add_argument('--max_epoch', type=int, default=200)
learn_arg.add_argument('--optim', type=str, default='adam')
learn_arg.add_argument('--clip_grad', type=float, default=0.)
learn_arg.add_argument('--lr', type=float, default=0.0001)
learn_arg.add_argument('--lr_decay', type=str2bool, default=False)
learn_arg.add_argument('--decay_rate', type=float, default=0.96)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_name', type=str, default='')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=200)
misc_arg.add_argument('--save_epoch', type=int, default=1)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='./logs')
misc_arg.add_argument('--output_dir', type=str, default='/tmp')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--gpu_ids', type=str, default='')
misc_arg.add_argument('--random_seed', type=int, default=1234)
misc_arg.add_argument('--debug', type=str2bool, default=False)


# Model
model_arg = add_argument_group('Model')




def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed
