import tensorflow as tf
import numpy as np
import os

# Hyper Parameters #

LR_A = 0.0001    # Learning rate for actor
LR_C = 0.002    # Learning rate for critic
GAMMA = 0.99     # Reward discount
TAU = 0.01      # Soft replacement
MEMORY_CAPACITY = 20000
BATCH_SIZE = 10#original 64


class DDPG(object):

    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.compat.v1.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            # Assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # Get parameters
        self.ae_params = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # Update the parameters of the target network
        # self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
        #                     for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        self.a_soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea)]
                               for ta, ea in zip(self.at_params, self.ae_params)]
        self.c_soft_replace = [[tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                               for tc, ec in zip(self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # In the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(q)  # Maximize Q(s,a)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())  # Initialize the variables

    # Choose action from the actor network
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    # Train the neural network
    def learn(self):
        # Soft target replacement
        self.sess.run(self.a_soft_replace)
        if self.c_replace_counter % 32 == 0:
            self.sess.run(self.c_soft_replace)
        self.c_replace_counter += 1

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    # Update the replay buffer
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:
            self.memory_full = True  # Start learn if the memory is full

    # Create actor network, output action a
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            a_n_l = 40
            a_l1 = tf.layers.dense(inputs=s, units=a_n_l, activation=tf.nn.relu, name='a_l1', trainable=trainable)
            a_l2 = tf.layers.dense(inputs=a_l1, units=a_n_l, activation=tf.nn.relu, name='a_l2', trainable=trainable)
            a = tf.layers.dense(inputs=a_l2, units=self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    # Create critic network, output state-action value Q(s,a)
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            c_n_l = 40
            w1_s = tf.get_variable('w1_s', [self.s_dim, c_n_l], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, c_n_l], trainable=trainable)
            b1 = tf.get_variable('b1', [1, c_n_l], trainable=trainable)
            c_l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            c_l2 = tf.layers.dense(inputs=c_l1, units=c_n_l, activation=tf.nn.relu, name='c_l2', trainable=trainable)
            c_l3 = tf.layers.dense(inputs=c_l2, units=c_n_l, activation=tf.nn.relu, name='c_l3', trainable=trainable)
            c_l4 = tf.layers.dense(inputs=c_l3, units=c_n_l, activation=tf.nn.tanh, name='c_l4', trainable=trainable)
            return tf.layers.dense(c_l4, 1, trainable=trainable)

    # Save model
    def save_model(self, model_directory):
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(model_directory, 'params'), write_meta_graph=False)

    # Restore model
    def restore_model(self, model_directory):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.join(model_directory, 'params'))

