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



#################################################
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[...,tf.newaxis]/255,tf.float32),
    tf.cast(mnist_labels,tf.int64))
)
dataset = dataset.shuffle(1000).batch(32)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3], activation="relu",input_shape=(None,None,1)),
    tf.keras.layers.Conv2D(16,[3,3], activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

for images,labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True)

loss_history = []

