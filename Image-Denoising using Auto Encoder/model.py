import tensorflow as tf
import numpy as np
import utils


class DAE(object):

    def __init__(self, config):

        # Model configuration.
        self.config = config    	

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.original_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.noisy_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.reconstructed_images = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Global step Tensor.
        self.global_step = None

        # A boolean indicating whether the current mode is 'training'.
        self.phase_train = True



    
  
    def build_inputs(self):
        original_images=tf.placeholder(tf.float32,[None,784])
        self.original_images = original_images
        noisy_images=tf.placeholder(tf.float32,[None,784])
        self.noisy_images = noisy_images
        phase_train=tf.placeholder(dtype=tf.bool)  
        self.phase_train = phase_train
              
        return original_images,noisy_images,phase_train
        


    def build_model(self):
        
                  
        def weight_var(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)


        def bias_var(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)


        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_two(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


        def conv_layer(input, shape):
          W = weight_var(shape)
          b = bias_var([shape[3]])
          return tf.nn.relu(conv2d(input, W) + b)


        x_image = tf.reshape(self.noisy_images, [-1, 28, 28, 1])
        conv1 = conv_layer(x_image, shape=[3, 3, 1, 32])
        conv1_pool = tf.nn.relu(max_pool_two(conv1))
        bn1=tf.nn.relu(tf.contrib.layers.batch_norm(conv1_pool))

        conv2 = conv_layer(conv1_pool, shape=[3, 3, 32, 64])
        conv2_pool = max_pool_two(conv2)
        bn1=tf.nn.relu(tf.contrib.layers.batch_norm(conv2_pool))


        conv_2_resize=tf.image.resize_images(conv2_pool,(14,14))
        conv_2_resize_conv=conv_layer(conv_2_resize,shape=[3, 3, 64, 64])

        conv_1_resize=tf.image.resize_images(conv_2_resize_conv,(28,28))
        conv_1_resize_conv=conv_layer(conv_1_resize,shape=[3, 3, 64, 64])
        images_original = tf.reshape(self.original_images, [-1, 28, 28, 1])
        
        #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) FOR ADAM OPTIMIZER
        #full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)  FOR DROPOUT
        x_reconstructed=tf.nn.relu(conv_1_resize_conv)        
        self.reconstructed_images = x_reconstructed

        # Compute losses.
        self.total_loss = tf.sqrt(tf.reduce_mean(tf.square(self.reconstructed_images - images_original)))

       

    def setup_global_step(self):
	    global_step = tf.Variable(
	    	initial_value=0,
	        name="global_step",
	        trainable=False,
	        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

	    self.global_step = global_step

    def build(self):
        self.build_inputs()
        self.build_model()
        self.setup_global_step()


