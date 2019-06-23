# Code adapted from Fashion MNIST repo
# https://github.com/zalandoresearch/fashion-mnist

# Code to update tf.estimator parameters adapted from:
# https://github.com/tensorflow/tensorflow/issues/16646

# Code to perform weight and neuron pruning adapted from:
# https://github.com/for-ai/TD

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    weight_names = ['h1/weights:0', 'h2/weights:0', 'h3/weights:0', 'h4/weights:0']

    accuracy_wp = [] #Accuracy for weight pruning
    accuracy_np = [] #Accuracy for neuron pruning

    #Percentage of sparcity
    sparsity = np.array([0, 25, 50, 60, 70, 80, 90, 95, 97, 99]) / 100.0

    #Testing weight pruning...

    for k in sparsity:

        #Loading saved dense model
        session = tf.Session()
        checkpoint = tf.train.get_checkpoint_state('./logs/')
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(session, checkpoint.model_checkpoint_path)

        #Gathering list of weights to prune
        w_vars = [
            [v for v in tf.trainable_variables() if v.name == node_name][0] for node_name in weight_names
        ]

        #Get actual values for the weights
        w_vals = [session.run(wm) for wm in w_vars]

        w_vals_pruned = []

        #Perform weight pruning for each layer
        for vals in w_vals:

            n1, n2 = np.shape(vals)

            #Reshape to 1D tensor
            vals = vals.reshape([-1])

            #Get absolute value for each weight
            abs_vals = np.abs(vals)

            #Get pruning threshold
            index = int(k * abs_vals.shape[0])
            mid = np.sort(abs_vals)[index:index + 1]

            #Remove weights below threshold
            mask = abs_vals >= mid
            pruned_vals = mask * vals

            #Reshape weight matrix back to 2D
            pruned_vals = pruned_vals.reshape([n1, n2])

            w_vals_pruned.append(pruned_vals)

        #Update the weights for each layer
        for i, v in zip(range(0, len(w_vars)), w_vars):
            session.run(v.assign(w_vals_pruned[i]))

        #Save the new model in a temporary directory
        reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
        global_step = reader.get_tensor('global_step')
        saver.save(session, './logs_tmp/model.ckpt', global_step=global_step)

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=mlp_model_fn, model_dir='./logs_tmp/')

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        accuracy_wp.append(eval_results['accuracy']*100.0)

    #Testing neuron pruning...

    for k in sparsity:

        #Loading saved dense model
        session = tf.Session()
        checkpoint = tf.train.get_checkpoint_state('./logs/')
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(session, checkpoint.model_checkpoint_path)

        #Gathering list of weights to prune
        w_vars = [
            [v for v in tf.trainable_variables() if v.name == node_name][0] for node_name in weight_names
        ]

        #Get actual values for the weights
        w_vals = [session.run(wm) for wm in w_vars]

        w_vals_pruned = []

        #Perform weight pruning for each layer
        for vals in w_vals:

            #Get the L2-norm for each column (neuron)
            norm = np.linalg.norm(vals, axis=0)

            #Get pruning threshold
            idx = int(k * norm.shape[0])
            mid = np.sort(norm, axis=0)[idx]

            #Remove weights below threshold
            mask = norm >= mid
            pruned_vals = mask * vals

            w_vals_pruned.append(pruned_vals)

        #Update the weights for each layer
        for i, v in zip(range(0, len(w_vars)), w_vars):
            session.run(v.assign(w_vals_pruned[i]))

        #Save the new model in a temporary directory
        reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
        global_step = reader.get_tensor('global_step')
        saver.save(session, './logs_tmp/model.ckpt', global_step=global_step)

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=mlp_model_fn, model_dir='./logs_tmp/')

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        accuracy_np.append(eval_results['accuracy']*100.0)

    #Plot the resutls
    plt.xticks(np.arange(0.0,  100.0, step=10.0))
    plt.yticks(np.arange(0.0, 100.0, step=5.0))
    plt.grid(color='k', linestyle='-', linewidth=1)
    plt.xlim(0, 100.0)
    #Plot weight pruning accuracy in green
    plt.plot(sparsity*100.0, accuracy_wp, 'g')
    #Plot neuron pruning accuracy in red
    plt.plot(sparsity*100.0, accuracy_np, 'r')
    plt.xlabel('Sparsity %')
    plt.ylabel('Model Accuracy')
    plt.show()

if __name__ == "__main__":
    tf.app.run()
