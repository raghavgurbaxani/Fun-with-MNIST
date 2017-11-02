import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data    #Data Load

#Define Data Directory
DATA_DIR='tmp/data'
MiniBatch_Size=50
Num_Steps=10000

#Function to define Weight variable
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Function to perform convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')
#Pooling Function
def max_pool(x):
    return tf.nn.max_pool(x,strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1] , padding='SAME')
#function for convolution layer
def conv_layer(input,shape):
    W=weight_variable(shape)
    b=bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input,W)+b)
#Function for fully connected layer
def full_layer(input,size):
    in_size=int(input.get_shape()[1])
    W=weight_variable([in_size,size])
    b=bias_variable([size])
    return tf.matmul(input,W)+b

X=tf.placeholder(tf.float32,shape=[None,784])   #input placeholder
Y=tf.placeholder(tf.float32,shape=[None,10])    #expected output placeholder

data=input_data.read_data_sets(DATA_DIR,one_hot=True)   #Read Data from directory

X_image=tf.reshape(X,shape=[-1, 28, 28, 1])     #reshaping to 28x28

conv_1=conv_layer(X_image, shape=[5, 5, 1, 32])     #sliding window of 5x5  and outputs a stack of 32 images
pool_1=max_pool(conv_1)
batch_norm_1=tf.nn.relu(tf.contrib.layers.batch_norm(pool_1))   #Batch Normalization

conv_2=conv_layer(batch_norm_1, shape=[5, 5, 32, 64])   #sliding window of 5x5  and outputs a stack of 32 images
pool_2=max_pool(conv_2)
batch_norm_2=tf.nn.relu(tf.contrib.layers.batch_norm(pool_2))   #Batch Normalization

X_flat=tf.reshape(batch_norm_2,shape=[-1,7*7*64])   #Flattening out the image

full_1=tf.nn.relu(full_layer(X_flat,1024))

dropout_probab=tf.placeholder(tf.float32)       #Placeholder for dropout
full_drop=tf.nn.dropout(full_1,keep_prob=dropout_probab)    #Perform dropout on fully connected layer

#SOFTMAX
Y_Conv=full_layer(full_drop,10)

#Loss Function
Loss_Function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y , logits= Y_Conv))
#Minimization 
Optimizer=tf.train.AdamOptimizer(1e-4).minimize(Loss_Function)


Correct_Mask=tf.equal( tf.argmax(Y,1) , tf.argmax(Y_Conv,1))
Accuracy=tf.reduce_mean(tf.cast(Correct_Mask,tf.float32))

#Start Session
with tf.Session() as session:
    #initialize all the variables
    session.run(tf.global_variables_initializer())
    #TRAIN
    print('TRAINING')
    for i in range(Num_Steps):
        batch=data.train.next_batch(MiniBatch_Size) #divide training data into batches
        
        if i % 100 == 0 :
            train_accuracy=session.run(Accuracy,feed_dict={X:batch[0] , Y:batch[1] , dropout_probab:1.0})
            
            print('Step {} ... Accuracy{}%  '.format(i,train_accuracy*100))
        
        session.run(Optimizer, feed_dict={X:batch[0] , Y:batch[1] , dropout_probab:1.0})
    #TEST
    x=data.test.images.reshape(10,1000,784)
    y=data.test.images.reshape(10,1000,784)
    print('TESTING')
    for j in range(10):
        test_accuracy=tf.reduce_mean(session.run(Accuracy,feed_dict={X:x[j] , Y:y[j] , dropout_probab:0.9}))
    print('Accuracy{}% '.format(test_accuracy*100))