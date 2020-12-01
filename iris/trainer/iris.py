#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ## TensorFlow programming
# 
# This tutorial is structured like many TensorFlow programs:
# 
# 1. Import and parse the dataset.
# 2. Create the model.
# 3. Train the model.
# 4. Evaluate the model's effectiveness.
# 5. Use the trained model to make predictions.

# ## Setup program

# ### Configure imports
# 
# Import TensorFlow and the other required Python modules.

import os
import argparse
import tensorflow as tf


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):

    # ## Import and parse the training dataset
    # 
    # Download the dataset file and convert it into a structure that can be used by this Python program.
    # 
    # ### Download the dataset
    # 
    # Download the training dataset file using the `tf.keras.utils.get_file` function. This returns the file path of the downloaded file:

    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                               origin=train_dataset_url)


    # column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    feature_names = column_names[:-1]
    label_name = column_names[-1]


    # Each label is associated with string name (for example, "setosa"), but machine learning typically relies on numeric values. The label numbers are mapped to a named representation, such as:
    # 
    # * `0`: Iris setosa
    # * `1`: Iris versicolor
    # * `2`: Iris virginica
    # 

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


    # ### Create a `tf.data.Dataset`
    # 
    # TensorFlow's [Dataset API](../../guide/data.ipynb) handles many common cases for loading data into a model. This is a high-level API for reading data and transforming it into a form used for training.
    # 
    # 

    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)


    # The `make_csv_dataset` function returns a `tf.data.Dataset` of `(features, label)` pairs, where `features` is a dictionary: `{'feature_name': value}`


    # To simplify the model building step, create a function to repackage the features dictionary into a single array with shape: `(batch_size, num_features)`.
    # 
    # This function uses the `tf.stack` method which takes values from a list of tensors and creates a combined tensor at the specified dimension:

    def pack_features_vector(features, labels):
      """Pack the features into a single array."""
      features = tf.stack(list(features.values()), axis=1)
      return features, labels


    # Then use the `tf.data.Dataset#map` method to pack the `features` of each `(features,label)` pair into the training dataset:

    train_dataset = train_dataset.map(pack_features_vector)


    # ## Import and parse the test dataset

    test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

    test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                      origin=test_url)

    test_dataset = tf.data.experimental.make_csv_dataset(
        test_fp,
        batch_size,
        column_names=column_names,
        label_name='species',
        num_epochs=1,
        shuffle=False)

    test_dataset = test_dataset.map(pack_features_vector)


    # ### Create a model using Keras
    # 
    # The TensorFlow `tf.keras` API is the preferred way to create models and layers. This makes it easy to build models and experiment while Keras handles the complexity of connecting everything together.
    # 
    # The `tf.keras.Sequential` model is a linear stack of layers. Its constructor takes a list of layer instances, in this case, one input layer with 4 nodes, two `tf.keras.layers.Dense` layers with 10 nodes each, and an output layer with 3 nodes representing our label predictions.

    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(4,)),
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(3)
    ])


    # The *[activation function](https://developers.google.com/machine-learning/crash-course/glossary#activation_function)* determines the output shape of each node in the layer. These non-linearities are important—without them the model would be equivalent to a single layer. There are many `tf.keras.activations`, but [ReLU](https://developers.google.com/machine-learning/crash-course/glossary#ReLU) is common for hidden layers.
    # 
    # The ideal number of hidden layers and neurons depends on the problem and the dataset. Like many aspects of machine learning, picking the best shape of the neural network requires a mixture of knowledge and experimentation. As a rule of thumb, increasing the number of hidden layers and neurons typically creates a more powerful model, which requires more data to train effectively.


    # ## Train the model
    # 
    # Both training and evaluation stages need to calculate the model's *[loss](https://developers.google.com/machine-learning/crash-course/glossary#loss)*. This measures how off a model's predictions are from the desired label, in other words, how bad the model is performing. We want to minimize, or optimize, this value.
    # 
    # Our model will calculate its loss using the `tf.keras.losses.SparseCategoricalCrossentropy` function which takes the model's class probability predictions and the desired label, and returns the average loss across the examples.

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_dataset,
              epochs=200)

    loss, accuracy = model.evaluate(test_dataset)
    print("Accuracy", accuracy)


    # ## Use the trained model to make predictions
    # 
    # We've trained a model and "proven" that it's good—but not perfect—at classifying Iris species. Now let's use the trained model to make some predictions on [unlabeled examples](https://developers.google.com/machine-learning/glossary/#unlabeled_example); that is, on examples that contain features but not a label.
    # 
    # In real-life, the unlabeled examples could come from lots of different sources including apps, CSV files, and data feeds. For now, we're going to manually provide three unlabeled examples to predict their labels. Recall, the label numbers are mapped to a named representation as:
    # 
    # * `0`: Iris setosa
    # * `1`: Iris versicolor
    # * `2`: Iris virginica

    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])

    predictions = model(predict_dataset)

    for i, logits in enumerate(predictions):
      class_idx = tf.argmax(logits).numpy()
      p = tf.nn.softmax(logits)[class_idx]
      name = class_names[class_idx]
      print("Example {} prediction: {} {} ({:4.1f}%)".format(i, predictions[i], name, 100*p))


    # ## Save the model

    model.save(args.job_dir)


if __name__ == '__main__':
    args = get_args()

    # Set verbosity
    tf.compat.v1.logging.set_verbosity('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train_and_evaluate(args)
