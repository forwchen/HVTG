import time
import os
import sys
sys.path.append('/home/forwchen/faster-rcnn.pytorch/lib')
import numpy as np
import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from tqdm import tqdm
import h5py
import pickle
import torch
from model.roi_layers import ROIAlign


roi_db = pickle.load(open(sys.argv[1], 'rb'))

roi_align = ROIAlign((1,1), 1, 2)

class ROIService(Service):
    def exposed_roi(self, feat_arr, key_arr):
        key = str(key_arr)
        vid = key.split('/')[0]
        fid = key.split('/')[1].split('.')[0]
        db_vid = roi_db[vid]
        rois = db_vid[fid]['box']
        feat = np.asarray(feat_arr)
        #print(rois.shape, flush=True)
        h,w = feat.shape[:2]
        rois[:,0] *= w
        rois[:,2] *= w
        rois[:,1] *= h
        rois[:,3] *= h
        rois = np.concatenate([np.zeros((len(rois), 1), dtype=np.float32),
                              rois], axis=1)
        roi_tensor = torch.tensor(rois).cuda()
        feat_tensor = torch.tensor(np.expand_dims(feat.transpose(2,0,1), 0)).cuda()
        #print(feat_tensor.size(), flush=True)
        aligned = roi_align(feat_tensor, roi_tensor)
        return aligned.data.cpu().numpy()


if __name__ == '__main__':

    s = ThreadedServer(ROIService, port=18870+int(sys.argv[2]), protocol_config = rpyc.core.protocol.DEFAULT_CONFIG)
    print('start serving')
    s.start()
