import os
import tensorflow as tf
import cProfile

tf.executing_eagerly()
x = [[2.]]
m = tf.matmul(x,x)
print("hello, {}".format(m))


# When you use tf.math you can cast python objects and numpy matrix to tf.tensor waaaaaaow! 
a = tf.constant([[1,2],[3,4]])

print(a)

#broadcasting
b = tf.add(a,1)
print(b)

print(a*b)

import numpy as np

c = np.multiply(a,b)
print(c)


print(a.numpy())