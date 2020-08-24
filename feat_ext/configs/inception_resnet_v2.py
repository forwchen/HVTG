import numpy as np

num_classes = 1001
label_offset = 1

no_crop = True

batch_size = 1 if no_crop else 40
ckpt = 'ckpts/inception_resnet_v2.ckpt'
color_mode = 'rgb'
base_size = 480
crop_size = -1


# mean and std in order of 'RGB'
mean_type = 'incep'

layer = 'Conv2d_7b_1x1'
#layer = 'global_pool'
