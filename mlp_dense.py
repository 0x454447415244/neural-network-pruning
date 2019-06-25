# Code adapted from Fashion MNIST repo
# https://github.com/zalandoresearch/fashion-mnist

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './data/fashion'


def mlp_model_fn(features, labels, mode):

    # Input tensor shape: [batch_size, 28 x 28]
    x = features["x"]

    # Below are 4 hidden layers with 1000, 1000, 500 and 200 units respectively
    # ReLU activation function is used for each layer

    h1 = tf.contrib.layers.fully_connected(x, num_outputs=1000, biases_initializer=None, activation_fn=tf.nn.relu, scope='h1')

    h2 = tf.contrib.layers.fully_connected(h1, num_outputs=1000, biases_initializer=None, activation_fn=tf.nn.relu, scope='h2')

    h3 = tf.contrib.layers.fully_connected(h2, num_outputs=500, biases_initializer=None, activation_fn=tf.nn.relu, scope='h3')

    h4 = tf.contrib.layers.fully_connected(h3, num_outputs=200, biases_initializer=None, activation_fn=tf.nn.relu, scope='h4')

    # Logits layer
    # Input Tensor Shape: [batch_size, 200]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.contrib.layers.fully_connected(h4, num_outputs=10, biases_initializer=None, activation_fn=None)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=mlp_model_fn, model_dir="./logs/")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=400,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for j in range(100):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=2000)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
