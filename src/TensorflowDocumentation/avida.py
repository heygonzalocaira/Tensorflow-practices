import os
import tensorflow as tf
import cProfile

tf.executing_eagerly()
x = [[2.]]
m = tf.matmul(x,x)
#print("hello, {}".format(m))


# When you use tf.math you can cast python objects and numpy matrix to tf.tensor waaaaaaow! 
a = tf.constant([[1,2],[3,4]])

#print(a)

#broadcasting
b = tf.add(a,1)
#print(b)

#print(a*b)

import numpy as np

c = np.multiply(a,b)
#print(c)


#print(a.numpy())


# You can uuse tf.GradientTape to train and/or calculate gradients

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, w)
print(grad)