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


def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training = True)

        tf.debugging.assert_equal(logits.shape, (32,10))

        loss_value = loss_object(labels,logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        for (batch, (images,labels)) in enumerate(dataset):
            train_step(images,labels)
        print("Epoch {} finished".format(epoch))

#train(epochs=3)

import matplotlib.pyplot as plt

#plt.plot(loss_history)
#plt.xlabel("Batch #")
#plt.ylabel("loss [entropy]")
#plt.show()

###################################################
###################################################
###################################################
class Linear(tf.keras.Model):
  def __init__(self):
    super(Linear, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B


NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs  = training_inputs * 3 + 2 +noise

#The loss function to be optimized

def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))


steps = 300
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads,[model.W,model.B]))
    if i %20 == 0 :
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))


model.save_weights("weights")
status = model.load_weights("weights")

x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)

x.assign(2.)
checkpoint_path = "./ckpt/"
checkpoint.save(checkpoint_path)
