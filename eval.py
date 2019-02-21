# -*- coding: utf-8 -*-

import sys
import os
import time
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn


x_text = ['it is so bad.', 'nice buying experience.']
y_test = [0, 1]

# load the vocabulary
vocab_path = './data/vocab'
vocab_prosessor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_prosessor.transform(x_text)))

print('Evaluating ...\n')

# evaluating
checkpoint_file = './data/model-5500'
print('Loaded the latest checkpoint: <{}>\n'.format(checkpoint_file))

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # predict
        print(x_test.shape)
        predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

for i in range(len(x_text)):
    print('TEXT:       {}\nPREDICTION: {}\n'.format(x_text[i], predictions[i]))

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(predictions == y_test))
    print("\nTotal number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))
