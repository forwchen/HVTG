import numpy as np
import PIL
from PIL import Image
#from scipy.misc import imresize
import io


def _prep_img(dp, cfg=None):
    if type(dp[1]) is bytes:
        k, v = dp
        s = io.BytesIO(v)
        k = k.decode('ascii')
    elif type(dp[1]) is str:
        k, s = dp

    try:
        _img = Image.open(s)
    except:
        return '@invalid_img/'+k, np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.float32)

    if _img.mode != 'RGB':
        _img = _img.convert('RGB')
    #_img = np.array(_img) # shape is (height, width, channel)
    w,h = _img.size

    short_edge = min(w, h)
    if short_edge != cfg.base_size:
        scale = cfg.base_size * 1.0 / short_edge
        _img = _img.resize((int(w*scale), int(h*scale)),
                            PIL.Image.BILINEAR)

    _img = np.array(_img) # shape is (height, width, channel)

    if not cfg.no_crop:
        h_off = (_img.shape[0] - cfg.crop_size)//2
        w_off = (_img.shape[1] - cfg.crop_size)//2
        _img = _img[h_off:h_off+cfg.crop_size, w_off:w_off+cfg.crop_size]

    if cfg.color_mode == 'bgr':   # the color mode the network is expecting
        print('warning: the network is expecting BGR inputs')
    _img = _img.astype(np.float32)
    # input images should all be RGB
    if cfg.mean_type == 'vgg':
        _img = (_img - cfg.mean)/cfg.std
    elif cfg.mean_type == 'incep':
        _img = (_img / 255. - 0.5) * 2.0
    else:
        print('error: no mean_type specified')

    return k, _img
