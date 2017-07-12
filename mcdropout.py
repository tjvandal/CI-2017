import os, sys
import numpy as np
import tensorflow as tf

class MCDropout(object):
    def __init__(self, memory, hidden_layers=[64,64], learning_rate=0.001,
                dropout_rate=0.20, epochs=1000, mc_runs=20):
        self.memory = memory
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.graph = tf.Graph()
        self.epochs = epochs
        self.mc_runs = mc_runs

        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.memory))
        self.y = tf.placeholder(tf.float32, shape=(None,1))
        self.global_step = tf.Variable(1, dtype=tf.float32)
        with tf.device("/gpu:0"):
            h = self.x
            h = tf.nn.dropout(h, 1-self.dropout_rate) * (1 - self.dropout_rate)
            #h = tf.multiply(self.x, tf.random_normal(shape=(1, self.memory), mean=1.0, stddev=1.0,
            #                                         dtype=tf.float32))
            for h_layer in self.hidden_layers:
                h = tf.contrib.layers.fully_connected(h, num_outputs=h_layer, activation_fn=tf.nn.relu)
                h = tf.nn.dropout(h, 1-self.dropout_rate) * (1-self.dropout_rate)
                #h = tf.multiply(h, tf.random_normal(shape=(1, h_layer), mean=1.0, stddev=1.00,
                #                                    dtype=tf.float32))
            self.yhat = tf.contrib.layers.fully_connected(h, num_outputs=1, activation_fn=None)
            self.loss = 0.5*tf.reduce_mean(tf.square(self.y - self.yhat)) #tf.losses.mean_squared_error(self.y, self.yhat)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = opt.minimize(self.loss, global_step=self.global_step)

    def train(self, y, sess):
        x, y = _build_data(y, self.memory)
        feed_dict = {self.x: x, self.y: y[:,np.newaxis]}
        for epoch in range(self.epochs+1):
            res = sess.run([self.optimizer, self.loss, self.yhat], feed_dict=feed_dict)
            if epoch % 100 == 0:
                print "(Training) Interation: %i, Loss: %f, yhat mean: %f" % (epoch, res[1], np.mean(res[2]))

    def predict(self, init_state, steps, sess):
        y = init_state.tolist()
        y_samples = np.empty((steps, self.mc_runs))
        for step in range(steps):
            feed_dict = {self.x: np.array(y[-self.memory:])[np.newaxis,:]}
            y_samples[step, :] =  [sess.run(self.yhat, feed_dict=feed_dict)[0][0] for j in range(self.mc_runs)]
            y.append(y_samples[step].mean())
        return y_samples
