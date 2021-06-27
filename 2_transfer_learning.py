import tensorflow as tf
import matplotlib.pylab as plt

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


CLASSSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGES_RES = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSSIFIER_URL,input_shape=(IMAGES_RES,IMAGES_RES,3))
])

import numpy as np
import PIL.Image as Image

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

(train_examples, validation_examples), info = tfds.load(
    "cats_vs_dogs",
    with_info = True,
    as_supervised = True,
    split = ["train[:80%]","train[80%:]"]
)

num_examples = info.splits["train"].num_examples
num_examples = info.features["label"].num_classes

for i, example_image in enumerate(train_examples.take(3)):
    print("Image {} Shape : {}".format(i+1,example_image[0].shape))


def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)


image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

result_batch = model.predict(image_batch)

predicted_class_names = imagenet_labels[np.argmax(result_batch,axis= -1)]

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")