import tensorflow as tf
import numpy as np
from utils import *


SEED=1234
BATCH_SIZE=100


class dataset(object):
    def __init__(self, inputs, labels, randomize=True):
        self.inputs = inputs
        self.labels = labels
        if len(self.labels.shape) == 1:
            self.labels = np.reshape(self.labels,
                                     [self.labels.shape[0], 1])
        assert len(self.inputs) == len(self.labels)

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = self.num_pairs
        #self.init_pointer()

    def len(self):
        return self.num_pairs

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]
            
    def next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


def gan_feed_forward(x, num_hidden, activation, is_training):
    depth = len(num_hidden)
    layers = [tf.identity(x)]
    for _ in range(depth):
        print(num_hidden[_])
        cur_layer = tf.layers.dense(layers[-1], num_hidden[_], name='dense_' + str(_), 
                                    activation=activation[_], 
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        layers.append(cur_layer)
    return layers[-1]


def build_distribution_matching(x_dim):
    init = tf.placeholder(dtype=tf.float32, shape=[None, x_dim])
    real_x = tf.placeholder(dtype=tf.float32, shape=[None, x_dim])

    batch_size = BATCH_SIZE
    stddev = 1.0 / np.sqrt(x_dim)

    epsilon = tf.random_normal([batch_size, x_dim * 2], 0.0, stddev, seed=1234)

    with tf.variable_scope('gen', reuse=False):
        fake_x = gan_feed_forward(epsilon, [1024, 1024, x_dim], [tf.nn.relu]*2+[None], False)

    real_x_concat = real_x #tf.concat([init, real_x], 1)
    with tf.variable_scope('disc', reuse=False):
        real_x_critic = gan_feed_forward(real_x_concat, [1024, 1024, 1], [tf.nn.relu]*2+[None], False)

    fake_x_concat = fake_x #tf.concat([init, fake_x], 1)
    with tf.variable_scope('disc', reuse=True):
        fake_x_critic = gan_feed_forward(fake_x_concat, [1024, 1024, 1], [tf.nn.relu]*2+[None], False)

    # wasserstein
    alpha = tf.random_uniform(tf.convert_to_tensor([batch_size, 1], dtype=tf.int64), 0, 1, seed=SEED)
    #inds = tf.range(batch_size)
    #inds = tf.squeeze(tf.random_shuffle(tf.expand_dims(tf.range(batch_size), 1)))
    #fake_x_ = fake_x_concat
    #real_x_ = tf.gather(real_x_concat, inds)

    hat_x_concat = fake_x * alpha + (1 - alpha) * real_x
    print(hat_x_concat.get_shape())

    with tf.variable_scope('disc', reuse=True):
        hat_x_critic = gan_feed_forward(hat_x_concat, [1024, 1024, 1], [tf.nn.relu]*2+[None], False)

    w_dist = tf.reduce_mean(real_x_critic) - tf.reduce_mean(fake_x_critic)
    gp_loss = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.gradients(hat_x_critic, hat_x_concat)[0] ** 2,
                               reduction_indices=[1])) - 1.) ** 2
    )

    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc')
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')

    d_loss = -w_dist + 10 * gp_loss
    disc_op = tf.train.AdamOptimizer(1e-4)
    disc_grads = disc_op.compute_gradients(loss=d_loss,
                                           var_list=d_vars)
    disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads)

    g_loss = w_dist
    gen_op = tf.train.AdamOptimizer(1e-4)
    gen_grads = gen_op.compute_gradients(loss=g_loss,
                                           var_list=g_vars)
    gen_train_op = gen_op.apply_gradients(grads_and_vars=gen_grads)

    ph = {
        'init': init,
        'real_x': real_x,
    }
    targets = {
        'gen':{
            'train': gen_train_op,
            'g_loss': g_loss,
        },
        'disc':{
            'train': disc_train_op,
            'd_loss': d_loss,
            'gp_loss': gp_loss,
            'w_dist_loss': w_dist,
        },
        'sample':{
            'fake_x': fake_x
        }
    }

    return ph, targets


def train_dist_matching(sess, init_weights_train, trained_weights_train, init_weights_test, ph, targets, niters=1000):
    log_interval = 100
    batch_size = BATCH_SIZE
    ndata_train = int(trained_weights_train.shape[0])
    print(ndata_train)
    logs = {}
    ncritic = 5

    cps = dataset(init_weights_train, trained_weights_train)
 
    for i in range(niters):
        cur_idx = np.random.permutation(ndata_train)
        for t in range(ndata_train // batch_size):
            for k in range(ncritic):
                batch_init, batch_x = cps.next_batch(batch_size)
                fetch = sess.run(targets['disc'], feed_dict={ph['init']: batch_init, ph['real_x']: batch_init})
                update_loss(fetch, logs)
            batch_init, batch_x = cps.next_batch(batch_size)
            fetch = sess.run(targets['gen'], feed_dict={ph['init']: batch_init, ph['real_x']: batch_init})
            update_loss(fetch, logs)
        if (i + 1) % log_interval == 0:
            print_log('Dist Train', (i + 1) // log_interval, logs)
            logs = {}
    outs = []
    for t in range(init_weights_test.shape[0] // batch_size):
        outs.append(sess.run(targets['sample']['fake_x'], feed_dict={ph['init']: init_weights_test[t * batch_size: (t + 1) * batch_size]}))
    return np.concatenate(outs, 0)




