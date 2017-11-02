import numpy as np
import tensorflow as tf

def batch_norm(x, n_out, phase_train):
  
  with tf.variable_scope('bn'):
      beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
      gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
      ema = tf.train.ExponentialMovingAverage(decay=0.5)

      def mean_var_with_update():
          ema_apply_op = ema.apply([batch_mean, batch_var])
          with tf.control_dependencies([ema_apply_op]):
              return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(phase_train,
                          mean_var_with_update,
                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
      normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

  return normed    

def add_noise(images):
    noisy_images=images.copy()
    n_shape=noisy_images.shape
  
    probability_mask=np.random.uniform(0,1,(n_shape))
    for prob,i in np.nditer((probability_mask,noisy_images),op_flags=['readwrite']):
        if i[...]>0 and i[...]<0.1:
            noisy_images[...]=0
        elif i[...]>0.1 and i[...]<0.2:
            noisy_images[...]=1
                  
    return noisy_images
