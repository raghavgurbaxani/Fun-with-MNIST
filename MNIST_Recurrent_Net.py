import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data    #Data Load

#Define Data Directory
DATA_DIR='tmp/data'
Num_Steps=10000

#Function to define Weight variable
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

time_steps=28 
vector_size=28 
num_classes=10
batch_size=128
hidden_units=128

X=tf.placeholder(tf.float32,shape=[None,time_steps,vector_size])   #input placeholder
Y=tf.placeholder(tf.float32,shape=[None,num_classes])    #expected output placeholder

data=input_data.read_data_sets(DATA_DIR,one_hot=True)   #Read Data from directory

rnn_part=tf.contrib.rnn.BasicRNNCell(hidden_units)
output, _ =tf.nn.dynamic_rnn(rnn_part,X,dtype=tf.float32)

W=weight_variable([hidden_units,num_classes])
b=bias_variable([num_classes])

def get_linear_layer(vector):
    return tf.matmul(vector,W) + b

final_rnn_layer= output[:,-1,:]
final_output = get_linear_layer(final_rnn_layer)

#Loss Function
Loss_Function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y , logits= final_output))
#Minimization 
Optimizer=tf.train.AdamOptimizer(1e-4).minimize(Loss_Function)


Correct_Mask=tf.equal( tf.argmax(Y,1) , tf.argmax(final_output,1))
Accuracy=tf.reduce_mean(tf.cast(Correct_Mask,tf.float32))*100

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_data = data.test.images[:batch_size].reshape((-1,
                                            time_steps, vector_size))
test_label = data.test.labels[:batch_size]
  
for i in range(3001):

       batch_x, batch_y = data.train.next_batch(batch_size)
       batch_x = batch_x.reshape((batch_size, time_steps, vector_size))
       sess.run(Optimizer,feed_dict={X:batch_x,
                                      Y:batch_y})
       if i % 1000 == 0:
            acc = sess.run(Accuracy, feed_dict={X: batch_x,
                                                Y: batch_y})
            loss = sess.run(Loss_Function,feed_dict={X:batch_x,
                                                     Y:batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))   

print ("Testing Accuracy:", 
    sess.run(Accuracy, feed_dict={X: test_data, Y: test_label}))