from utils import *
import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)
from dataset import Dataset
from tqdm import tqdm
import ipdb
import importlib

logger = get_logger()


def combine_gradients(tower_grads):
    filtered_grads = [
        [x for x in grad_list if x[0] is not None] for grad_list in tower_grads
    ]
    final_grads = []
    for i in range(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0)
        final_grads.append((
            grad,
            filtered_grads[0][i][1],
        ))

    return final_grads


def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def recall_at_iou(iou, x1, x2):
    x1 = (x1[0], x1[1])
    x2 = (x2[0], x2[1])

    iou_this = calculate_IoU(x1, x2)
    if iou_this >= iou:
        return 1.
    else:
        return 0.


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        self.sess = tf.Session(config=config)

    def build_dataset_tfr(self, mode):
        tfr_root = os.path.join(self.args.data_dir, 'features','tfrecords_ir_roi', self.args.mode)
        tfrs = [os.path.join(tfr_root, l) for l in os.listdir(tfr_root)]
        tf_ds = tf.data.TFRecordDataset(tfrs, num_parallel_reads=4)

        feature_description = {
            'idx': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'vid': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'feats': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'sent': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'lbl': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'm': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        def _parse_function(example_proto):
            # Parse the input `tf.Example` proto using the dictionary above.
            raw = tf.io.parse_single_example(example_proto, feature_description)
            raw['feats'] = tf.reshape(tf.io.decode_raw(raw['feats'], tf.float32),
                                      [self.args.max_feat_len, 16, 1536])
            raw['sent'] = tf.reshape(tf.io.decode_raw(raw['sent'], tf.float32),
                                     [self.args.max_sent_len, 300])
            raw['lbl'] = tf.io.decode_raw(raw['lbl'], tf.float32)
            raw['m'] = tf.io.decode_raw(raw['m'], tf.float32)
            return raw
        batch_size = self.args.batch_size

        tf_ds = tf_ds.map(_parse_function)
        tf_ds = tf_ds.repeat()
        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        if mode == 'train':
            n_sample = self.args.n_train
        else:
            n_sample = self.args.n_test
        return tf_ds, n_sample


    def build_dataset(self, mode):
        self.ds = Dataset(self.args, mode)
        data_type, data_shape = self.ds.get_data_type_and_shape()
        print(data_shape)

        tf_ds = tf.data.Dataset.from_generator(
            self.ds.gen_data,
            output_types=data_type,
            output_shapes=data_shape)

        batch_size = self.args.batch_size

        tf_ds = tf_ds.repeat()
        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        #tf_ds = tf_ds.prefetch(buffer_size=128)

        return tf_ds, len(self.ds)

    def get_model(self):
        if self.args.net_type.startswith('base'):
            model = importlib.import_module('models.'+self.args.net_type).Att(self.args)
        elif self.args.net_type.startswith('dev'):
            model = importlib.import_module('models.'+self.args.net_type).Dev(self.args)

        return model

    def train(self):
        if self.args.use_tfr:
            ds, n_sample = self.build_dataset_tfr('train')
        else:
            ds, n_sample = self.build_dataset('train')
        raw_ = ds.make_one_shot_iterator().get_next()

        keys = ['feats', 'sent', 'lbl', 'm', 'vid', 'idx']
        if type(raw_) is dict:
            raw = [raw_[k] for k in keys]
        else:
            raw = raw_

        raw_split = []
        for i in range(len(raw)):
            raw_split.append(tf.split(raw[i], self.args.num_gpu, axis=0, num=self.args.num_gpu))

        global_step = tf.Variable(0, trainable=False, name='global_step')


        if self.args.lr_decay:
            print('lr decay')
            lr = tf.train.exponential_decay(self.args.lr, global_step,
                                            decay_steps=n_sample//self.args.batch_size,
                                            decay_rate=self.args.decay_rate)
        else:
            lr = self.args.lr


        if self.args.optim == 'adam':
            opt = tf.train.AdamOptimizer(lr)
        elif self.args.optim == 'momt':
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9)


        losses = []
        grads = []

        for i in range(self.args.num_gpu):
            with tf.device('/gpu:%d'%i):
                with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                    input_i = [rs[i] for rs in raw_split]
                    model = self.get_model()
                    out = model.build(input_i)
                    print(len(tf.trainable_variables()))

                    loss = out['all_loss']

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                if update_ops:
                    with tf.control_dependencies(update_ops):
                        barrier = tf.no_op(name="gradient_barrier")
                        with tf.control_dependencies([barrier]):
                            loss = tf.identity(loss)

                losses.append(loss)

                grad = opt.compute_gradients(
                    loss,
                    #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    colocate_gradients_with_ops=False)
                grads.append(grad)

        final_loss = tf.reduce_mean(tf.stack(losses))
        merged_gradients = combine_gradients(grads)

        if self.args.clip_grad > 0:
            with tf.name_scope("clip_grads"):
                merged_gradients = clip_gradient_norms(merged_gradients,
                                                   self.args.clip_grad)

        train_op = opt.apply_gradients(merged_gradients,
                                       global_step=global_step)

        for v in tf.trainable_variables():
            print(v.op.name)

        self.sess.run(tf.global_variables_initializer())
        gvars = [v for v in tf.global_variables() if 'Adam' not in v.op.name]
        saver = tf.train.Saver(gvars, max_to_keep=9999)
        if len(self.args.load_path) > 0:
            saver.restore(self.sess, self.args.load_path)

        summary_writer = tf.summary.FileWriter(self.args.model_dir, flush_secs=10)
        merged_summary_op = tf.summary.merge_all()

        tot_iter = 0.
        tot_loss = 0.
        for epoch in range(self.args.max_epoch):
            n_iter = n_sample // self.args.batch_size
            pbar = tqdm(total=n_iter, desc="train epoch %d"%epoch, ncols=80)
            for i in range(n_iter):
                r = self.sess.run([global_step, merged_summary_op, train_op, final_loss])
                tot_iter += 1
                tot_loss += r[3]

                pbar.set_description(f"train epoch {epoch} | loss: {tot_loss/tot_iter:5.3f}")
                pbar.update(1)
                summary_writer.add_summary(r[1], r[0])
            pbar.close()
            if epoch % self.args.save_epoch == 0:
                try:
                    saver.save(self.sess,
                               os.path.join(self.args.model_dir, 'model'),
                               global_step=global_step)
                except:
                    print('saving checkpoint failed')
                    pass

    def test(self):
        ds, n_sample = self.build_dataset('test')
        raw = ds.make_one_shot_iterator().get_next()

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            model = self.get_model()
            out = model.build(raw)
        out['idx'] = raw[-1]
        out['lbl'] = raw[2]

        gvars = tf.global_variables()
        saver = tf.train.Saver(gvars)
        saver.restore(self.sess, self.args.load_path)

        rcl = {'0.1': 0.,
               '0.3': 0.,
               '0.5': 0.,
               '0.7': 0.}
        tested = {}

        n_iter_test = n_sample // self.args.batch_size + 1
        for i in tqdm(range(n_iter_test), ncols=64):
            try:
                t = self.sess.run(out)
            except:
                break

            for ix, p, l in zip(t['idx'], t['preds'], t['lbl']):
                if ix in tested:
                    continue
                tested[ix] = 1

                for k in rcl.keys():
                    rcl[k] += recall_at_iou(float(k), p, l)
        for k in rcl.keys():
            rcl[k] /= len(tested)
        print(rcl)
        out_file = os.path.join(self.args.load_path+'.result')
        with open(out_file, 'w') as ff:
            #for k in rcl:
            #    ff.write('%s %f' % (k, rcl[k]))
            ff.write(json.dumps(rcl))


