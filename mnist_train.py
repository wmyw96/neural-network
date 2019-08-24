from utils import *
from dist_matching import *
import numpy as np
import datetime
import tensorflow as tf
import json, sys, os
from os import path
import time
import shutil
import matplotlib
import importlib
import argparse
from tqdm import tqdm


# Parse cmdline args
parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--x_dim', default=28*28, type=int)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--nclass', default=10, type=int)
parser.add_argument('--num_hidden', default='100,10', type=str)
parser.add_argument('--activation', default='tanh', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epoches', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch_id', default=10, type=int)
parser.add_argument('--mode', default='dist_train', type=str)
parser.add_argument('--save_weight_dir', default='saved_weights/mnist-100', type=str)
parser.add_argument('--load_weight_dir', default='saved_weights/mnist-100', type=str)

def get_network_params():
    num_hidden = args.num_hidden.split(',')
    num_hidden = [int(x) for x in num_hidden]
    act = None
    if args.activation == 'tanh':
        act = tf.nn.tanh
    else:
        raise NotImplemented
    activation = [act] * len(num_hidden) + [None]
    return num_hidden, activation

args = parser.parse_args()

# GPU settings
if args.gpu > -1:
    print("GPU COMPATIBLE RUN...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


def feed_forward(x, num_hidden, activation, is_training):
    depth = len(num_hidden)
    layers = [tf.identity(x)]
    for _ in range(depth):
        print(num_hidden[_])
        cur_layer = tf.layers.dense(layers[-1], num_hidden[_], name='dense_' + str(_), 
                                    activation=activation[_], 
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        layers.append(cur_layer)
    return layers[-1]


def show_variables(cl):
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))

def build_mnist_model(num_hidden, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    onehot_y = tf.one_hot(y, args.nclass)
    with tf.variable_scope('network'):
        out = feed_forward(x, num_hidden, activation, False)
    log_y = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_y, logits=out, dim=1)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), y), tf.float32))
    entropy_loss = tf.reduce_mean(log_y)

    all_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
    show_variables(all_weights)
    last_layer_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='network/dense_{}'.format(len(num_hidden) - 1))

    all_op = tf.train.AdamOptimizer(args.lr, beta1=0.5)
    all_grads = all_op.compute_gradients(loss=entropy_loss, var_list=all_weights)
    all_train_op = all_op.apply_gradients(grads_and_vars=all_grads)

    lst_op = tf.train.AdamOptimizer(args.lr, beta1=0.5)
    lst_grads = lst_op.compute_gradients(loss=entropy_loss, var_list=last_layer_weights)
    lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)

    weight_dict = {}
    for item in all_weights:
        weight_dict[item.name] = item
    print(weight_dict)

    ph = {
        'x': x,
        'y': y
    }
    ph['kernel_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/kernel:0'].get_shape())
    ph['bias_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/bias:0'].get_shape())

    targets = {
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            'acc_loss': acc,
            'entropy_loss': entropy_loss
        },
        'lst':{
            'weights': all_weights,
            'train': lst_train_op,
            'acc_loss': acc,
            'entropy_loss': entropy_loss        
        },
        'eval':{
            'weights': weight_dict,
            'acc_loss': acc,
            'entropy_loss': entropy_loss    
        },
        'assign_weights':{
            'weights': tf.assign(weight_dict['network/dense_0/kernel:0'], ph['kernel_l0']),
            'bias': tf.assign(weight_dict['network/dense_0/bias:0'], ph['bias_l0']),
        }
    }

    return ph, targets


num_hidden, activation = get_network_params()
ph, targets = build_mnist_model(num_hidden, activation)
ph_dist, targets_dist = build_distribution_matching(args.x_dim + 1)

if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
else:
    sess = tf.Session()


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_images = mnist.test.images # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

train_images = train_images.reshape(-1, args.x_dim)
train_labels = train_labels.reshape(-1)
test_images = test_images.reshape(-1, args.x_dim)
test_labels = test_labels.reshape(-1)

ndata_train = train_images.shape[0]
ndata_test = test_images.shape[0]

sess.run(tf.global_variables_initializer())


def load_weights(epoch_title):
    epoch_dir = os.path.join(args.load_weight_dir, epoch_title)
    kernel = np.load(os.path.join(epoch_dir, 'network_dense_0_kernel_0.npy'))
    kernel = np.transpose(kernel, [1, 0])
    bias = np.load(os.path.join(epoch_dir, 'network_dense_0_bias_0.npy'))
    bias = np.expand_dims(bias, 1)
    weights = np.concatenate([kernel, bias], 1)
    return weights

def get_weights(sess, ph, targets):
    kernel = sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])
    kernel = np.transpose(kernel, [1, 0])
    bias = sess.run(targets['eval']['weights']['network/dense_0/bias:0'])
    bias = np.expand_dims(bias, 1)
    weights = np.concatenate([kernel, bias], 1)
    return weights

def set_weights(val, sess, ph, targets):
    kernel = val[:, :args.x_dim]
    kernel = np.transpose(kernel, [1, 0])
    bias = val[:, args.x_dim]
    sess.run(targets['assign_weights']['weights'], feed_dict={ph['kernel_l0']: kernel})
    sess.run(targets['assign_weights']['bias'], feed_dict={ph['bias_l0']: bias})


if args.mode == 'train':

    epoch_dir = os.path.join(args.save_weight_dir, 'init')
    if not os.path.exists(epoch_dir):
        os.mkdir(epoch_dir)
    for item in targets['eval']['weights']:
        name = item
        name_saved = name.replace('/', '_', 3)
        name_saved = name_saved.replace(':', '_', 1)
        val = sess.run(targets['eval']['weights'][name])
        np.save(os.path.join(epoch_dir, name_saved + '.npy'), val)

    for epoch in range(args.num_epoches):
        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_images[batch_idx, :]
            batch_y = train_labels[batch_idx]
            fetch = sess.run(targets['all'], feed_dict={ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, train_info)

        test_info = {}
        for t in tqdm(range(ndata_test // args.batch_size)):
            batch_x = test_images[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = test_labels[t * args.batch_size: (t + 1) * args.batch_size]
            fetch = sess.run(targets['eval'], feed_dict={ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)

        print_log('Train', epoch, train_info)
        print_log('Test', epoch, test_info)

        # save weights
        epoch_dir = os.path.join(args.save_weight_dir, 'epoch{}'.format(epoch))
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        for item in targets['eval']['weights']:
            name = item
            name_saved = name.replace('/', '_', 3)
            name_saved = name_saved.replace(':', '_', 1)
            val = sess.run(targets['eval']['weights'][name])
            np.save(os.path.join(epoch_dir, name_saved + '.npy'), val)

elif args.mode == 'dist_train':
    init_weights = load_weights('init')
    trained_weights = load_weights('epoch{}'.format(args.epoch_id))

    cur_weights = get_weights(sess, ph, targets)

    for i in range(100):
        features = train_dist_matching(sess, init_weights, trained_weights, cur_weights, ph_dist, targets_dist)

        set_weights(features, sess, ph, targets)

        for epoch in range(20):
            cur_idx = np.random.permutation(ndata_train)
            train_info = {}
            for t in tqdm(range(ndata_train // args.batch_size)):
                batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
                batch_x = train_images[batch_idx, :]
                batch_y = train_labels[batch_idx]
                fetch = sess.run(targets['lst'], feed_dict={ph['x']: batch_x, ph['y']: batch_y})
                update_loss(fetch, train_info)

            test_info = {}
            for t in tqdm(range(ndata_test // args.batch_size)):
                batch_x = test_images[t * args.batch_size: (t + 1) * args.batch_size, :]
                batch_y = test_labels[t * args.batch_size: (t + 1) * args.batch_size]
                fetch = sess.run(targets['eval'], feed_dict={ph['x']: batch_x, ph['y']: batch_y})
                update_loss(fetch, test_info)

            print_log('Train', epoch, train_info)
            print_log('Test', epoch, test_info)

