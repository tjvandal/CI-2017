import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Dense, Lambda, Wrapper
import tensorflow as tf
import numpy as np

class KerasMCDropout(Wrapper):
    def __init__(self, layer, dropout_prob=0.0, is_mc_dropout=True, alpha_divergence=False,
		**kwargs):
        super(KerasMCDropout, self).__init__(layer, **kwargs)
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p = dropout_prob
	self.alpha_divergence = alpha_divergence

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(KerasMCDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise regulariser / prior KL term
        input_dim = np.prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        
        # add l2_loss
        regularizer = K.sum(K.square(weight)) * self.p
        self.layer.add_loss(regularizer)
        
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def mc_dropout(self, x):
        retain_prob = 1. - self.p
        randomness = K.random_binomial(shape=K.shape(x), p=retain_prob)
        x *= randomness
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
	if self.alpha_divergence:
	    return self.layer.call(self.mc_dropout(inputs))
	else:
	    return self.layer.call(self.mc_dropout(inputs))


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        input_dim = np.prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        #kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) * (1. - self.p) 
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

        
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

class TFMCDropout(object):
    def __init__(self, hidden_layers=[64,64], learning_rate=0.01,
                dropout_rate=0.0, epochs=5000, mc_runs=20):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.graph = tf.Graph()
        self.epochs = epochs
        self.mc_runs = mc_runs
        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=(None,1))
        self.y = tf.placeholder(tf.float32, shape=(None,1))
        self.global_step = tf.Variable(1, dtype=tf.float32)
        with tf.device("/cpu:0"):
            h = self.x
            for h_layer in self.hidden_layers:
                h = tf.nn.dropout(h, 1-self.dropout_rate) * (1 - self.dropout_rate)
                h = tf.contrib.layers.fully_connected(h, num_outputs=h_layer, activation_fn=tf.nn.relu)
                
            h = tf.nn.dropout(h, 1-self.dropout_rate) * (1-self.dropout_rate)
            out = tf.contrib.layers.fully_connected(h, num_outputs=2, activation_fn=None)
            
            self.yhat = out[:, :1]
            log_var = out[:, 1:]
            precision = K.exp(-log_var)
            self.loss = tf.reduce_sum(precision * (self.y - self.yhat)**2. + log_var) # heteroskedastic
            #self.loss = 0.5*tf.reduce_mean(tf.square(self.y - self.yhat))  # homoskedastic
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = opt.minimize(self.loss, global_step=self.global_step)

    def train(self, x, y, sess):
        feed_dict = {self.x: x, self.y: y}
        for epoch in range(self.epochs+1):
            res = sess.run([self.optimizer, self.loss, self.yhat], feed_dict=feed_dict)
            if epoch % 100 == 0:
                print "(Training) Interation: %i, Loss: %f, yhat mean: %f" % (epoch, res[1], np.mean(res[2]))

    def predict(self, x, sess):
        feed_dict = {self.x: x}
        y_samples = np.array([sess.run(self.yhat, feed_dict=feed_dict) 
                        for _ in range(self.mc_runs)]).T
        return y_samples.squeeze()

def smooth(x,window_len=11,window='hanning'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y
