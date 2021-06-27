import io
import os
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

import te

model = "final_small.vec"


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    words, vecs = [], []
    for line in fin:
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        vecs.append(np.array(list(map(float, tokens[1:]))))
    return words, vecs


LOG_DIR = 'logs'

words, vecs = load_vectors(model)
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

#mnist = input_data.read_data_sets('MNIST_data')
#images = tf.Variable(mnist.test.images, name='images')

with open(metadata, 'w') as metadata_file:
    for row in words:
        metadata_file.write('%d\n' % row)

with tf.Session() as sess:
    saver = tf.train.Saver([vecs])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)