# HVTG
Code for ECCV 2020 paper "Hierarchical Visual-Textual Graph for Temporal Activity Localization via Language"

Requirements:
* Tensorflow `1.12 py3.6 cuda9.0_cudnn7.6.0` (for feature extraction and training)
* Pytorch `1.0.0 py3.6 cuda9.0.176_cudnn7.4.1_1` (for ROI Align module)
* Caffe `py2.7_cuda8.0` (for object ROI extraction)

### 0: Clone This Repo
Suppose we are in the HVTG/ folder.

### 1: Feature Extraction

#### 1.1 Extract video frames
Download videos from [Charades](http://ai2-website.s3.amazonaws.com/data/Charades_v1.zip) and [ActivityNet](http://activity-net.org/download.html).
It is simple to extract video frames with ffmpeg (We used version 4.1.3). Each video's frames should be put in a directory (e.g., ``$HOME/anet/frames/01vNlQLepsE``).
Make a list of all the directories, we can also split (using the split command) the list and do the following steps in parallel to speed things up.

#### 1.2 Setup object detection code
Download the object detection code and model from [this repository](https://github.com/peteanderson80/bottom-up-attention),
and compile and install the repo according to their guide.
Put the files in ``./tools`` into ``bottom-up-attention/tools``.

#### 1.3 ROI extraction
Go to bottom-up-attention/tools.
Run the script:
```
./roi_ext_per_vid.sh anet split_cha/cha_split_0 0
```
The parameters means dataset, path to frame directory list, and GPU id.

The extracted object ROIs will be stored in a directory, one pickled file per video.
We'll need to merge them in to a single dictionary keyed with each video's id.

#### 1.4 ROI feature extraction
Go to ./feat\_ext.
Download pretrained Inception-ResNet-v2 model [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained),
and unzip and put it into ./ckpts

We need the ROI Align code from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
Download and compile it, and put the library path in ``roi_align_rpc_server.py`` line 4.

Start the rpc server:
```
python roi_align_rpc_server.py rois_cha.pkl 1234
```

Start feature extraction: 
```
python feat_extract_roi.py inception_resnet_v2 all_frames_cha.txt roi_feat_cha.hdf5 1234
```
The parameters means network type, list of video frames (full paths), h5 DB to store output features, and RPC port number.

The frame list can be produced with command like: ``find $HOME/cha/frames -type f``.
The resulting features are stored in a hdf5 file, each key 'videoID/frameID' corresponds to object features for that frame. 
We need to merge the features of one video into a tensor shaped [# frames, # objects, # channels], and save as numpy file.

### 2: Train and Test
Go to ./hvtg. Download [processed annotations](https://drive.google.com/file/d/1lQgHcnM6-Bw7aEVyvaLXDS0_EQNokqF_/view?usp=sharing) and uncompress here.
Processed ActivityNet annotations can be downloaded [here](https://drive.google.com/file/d/1yn1PCVAFAxT15_KyH3s1jfS-lLv-g72e/view?usp=sharing).

#### 2.1 Convert features format
For the sake of I/O efficiency, we need to convert the training data into tfrecord format.
Place the numpy features in ``./data_cha/features/ir_roi`` and make a directory ``./data_cha/features/tfrecords_ir_roi``.

Run:
```
python make_tfrec.py --output_dir ./data_cha/features/tfrecords_ir_roi  --max_sent_len 12 --max_feat_len 128
```

#### 2.2 Train
Run: 
```
python main_mgpu.py --model_name hvtg --net_type hvtg --mode train --dataset charades --data_dir ./data_cha --max_sent_len 12  --max_feat_len 128 --use_tfr True --batch_size 32 --optim adam --lr 0.0001 --max_epoch 100 --log_dir ./logs --num_gpu 2
```

#### 2.3 Test
Select some model checkpoint in ``log_dir`` and run:
```
python main_mgpu.py --model_name hvtg --net_type hvtg --mode test --batch_size 16 --num_gpu 1 --load_path ./logs/charades_hvtg/model-xxxxx
```

``model-xxxxx`` refers to a model checkpoint stored at training step ``xxxxx``.
