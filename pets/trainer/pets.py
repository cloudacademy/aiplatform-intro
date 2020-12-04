#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

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


# This tutorial demonstrates how to classify structured data (e.g. tabular data in a CSV). We will use [Keras](https://www.tensorflow.org/guide/keras) to define the model, and [feature columns](https://www.tensorflow.org/guide/feature_columns) as a bridge to map from columns in a CSV to features used to train the model. This tutorial contains complete code to:
# 
# * Load a CSV file using [Pandas](https://pandas.pydata.org/).
# * Build an input pipeline to batch and shuffle the rows using [tf.data](https://www.tensorflow.org/guide/datasets).
# * Map from columns in the CSV to features used to train the model using feature columns.
# * Build, train, and evaluate a model using Keras.
# 
# ## The Dataset
# 
# We will use a simplified version of the PetFinder [dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). There are several thousand rows in the CSV. Each row describes a pet, and each column describes an attribute. We will use this information to predict the speed at which the pet will be adopted.
# 
# Following is a description of this dataset. Notice there are both numeric and categorical columns. There is a free text column which we will not use in this tutorial.
# 
# Column        | Description                       | Feature Type  | Data Type
# --------------|-----------------------------------|---------------|---------
# Type          | Type of animal (Dog, Cat)         | Categorical   | string
# Age           | Age of the pet                    | Numerical     | integer
# Breed1        | Primary breed of the pet          | Categorical   | string
# Gender        | Gender of the pet                 | Numerical     | integer
# Color1        | Color 1 of pet                    | Categorical   | string
# Color2        | Color 2 of pet                    | Categorical   | string
# MaturitySize  | Size at maturity                  | Categorical   | string
# FurLength     | Fur length                        | Categorical   | string
# Vaccinated    | Pet has been vaccinated           | Categorical   | string
# Sterilized    | Pet has been sterilized           | Categorical   | string
# Health        | Health Condition                  | Categorical   | string
# Fee           | Adoption Fee                      | Numerical     | integer
# Description   | Profile write-up for this pet     | Text          | string
# PhotoAmt      | Total uploaded photos for this pet | Numerical    | integer
# AdoptionSpeed | Speed of adoption                 | Classification | integer

# ## Import TensorFlow and other libraries

import argparse
import os
import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


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

     # ## Use Pandas to create a dataframe
    # 
    # [Pandas](https://pandas.pydata.org/) is a Python library with many helpful utilities for loading and working with structured data. We will use Pandas to download the dataset from a URL, and load it into a dataframe.

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='.')
    dataframe = pd.read_csv(csv_file)


    # ## Create target variable
    # 
    # The task in the original dataset is to predict the speed at which a pet will be adopted (e.g., in the first week, the first month, the first three months, and so on). Let's simplify this for our tutorial. Here, we will transform this into a binary classification problem, and simply predict whether the pet was adopted, or not.
    # 
    # After modifying the label column, 0 will indicate the pet was not adopted, and 1 will indicate it was.

    # In the original dataset "4" indicates the pet was not adopted.
    dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

    # Drop unused columns.
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])


    # ## Split the dataframe into train, validation, and test
    # 
    # The dataset we downloaded was a single CSV file. We will split this into train, validation, and test sets.

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')


    # ## Create an input pipeline using tf.data
    # 
    # Next, we will wrap the dataframes with [tf.data](https://www.tensorflow.org/guide/datasets). This will enable us  to use feature columns as a bridge to map from the columns in the Pandas dataframe to features used to train the model. If we were working with a very large CSV file (so large that it does not fit into memory), we would use tf.data to read it from disk directly. That is not covered in this tutorial.

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
      dataframe = dataframe.copy()
      labels = dataframe.pop('target')
      ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
      if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
      ds = ds.batch(batch_size)
      return ds


    train_ds = df_to_dataset(train)
    val_ds = df_to_dataset(val, shuffle=False)
    test_ds = df_to_dataset(test, shuffle=False)


    # ## Choose which columns to use
    # The goal of this tutorial is to show you the complete code (e.g. mechanics) needed to work with feature columns. We have selected a few columns to train our model below arbitrarily.
    # 
    # Key point: If your aim is to build an accurate model, try a larger dataset of your own, and think carefully about which features are the most meaningful to include, and how they should be represented.

    feature_columns = []

    # ### Numeric columns
    # A [numeric column](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column) is the simplest type of column. It is used to represent real valued features. When using this column, your model will receive the column value from the dataframe unchanged.

    for header in ['PhotoAmt', 'Fee', 'Age']:
      feature_columns.append(feature_column.numeric_column(header))


    # In the PetFinder dataset, most columns from the dataframe are categorical.

    # ### Indicator columns
    # In this dataset, Type is represented as a string (e.g. 'Dog', or 'Cat'). We cannot feed strings directly to a model. Instead, we must first map them to numeric values. The categorical vocabulary columns provide a way to represent strings as a one-hot vector (much like you have seen above with age buckets). The vocabulary can be passed as a list using [categorical_column_with_vocabulary_list](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list), or loaded from a file using [categorical_column_with_vocabulary_file](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file).

    indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                              'FurLength', 'Vaccinated', 'Sterilized', 'Health']
    for col_name in indicator_column_names:
      categorical_column = feature_column.categorical_column_with_vocabulary_list(
          col_name, dataframe[col_name].unique())
      indicator_column = feature_column.indicator_column(categorical_column)
      feature_columns.append(indicator_column)


    # ### Embedding columns
    # Suppose instead of having just a few possible strings, we have thousands (or more) values per category. For a number of reasons, as the number of categories grow large, it becomes infeasible to train a neural network using one-hot encodings. We can use an embedding column to overcome this limitation. Instead of representing the data as a one-hot vector of many dimensions, an [embedding column](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column) represents that data as a lower-dimensional, dense vector in which each cell can contain any number, not just 0 or 1. The size of the embedding (8, in the example below) is a parameter that must be tuned.
    # 
    # Key point: using an embedding column is best when a categorical column has many possible values. We are using one here for demonstration purposes, so you have a complete example you can modify for a different dataset in the future.

    breed1 = feature_column.categorical_column_with_vocabulary_list(
          'Breed1', dataframe.Breed1.unique())
    breed1_embedding = feature_column.embedding_column(breed1, dimension=8)
    feature_columns.append(breed1_embedding)

    # ### Hashed feature columns
    # 
    # Another way to represent a categorical column with a large number of values is to use a [categorical_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket). This feature column calculates a hash value of the input, then selects one of the `hash_bucket_size` buckets to encode a string. When using this column, you do not need to provide the vocabulary, and you can choose to make the number of hash_buckets significantly smaller than the number of actual categories to save space.
    # 
    # Key point: An important downside of this technique is that there may be collisions in which different strings are mapped to the same bucket. In practice, this can work well for some datasets regardless.


    # ### Bucketized columns
    # Often, you don't want to feed a number directly into the model, but instead split its value into different categories based on numerical ranges. Consider raw data that represents a pet's age. Instead of representing age as a numeric column, we could split the age into several buckets using a [bucketized column](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column). Notice the one-hot values below describe which age range each row matches.

    age = feature_column.numeric_column('Age')
    age_buckets = feature_column.bucketized_column(age, boundaries=[6, 12, 18, 24])
    feature_columns.append(age_buckets)


    # ### Crossed feature columns
    # Combining features into a single feature, better known as [feature crosses](https://developers.google.com/machine-learning/glossary/#feature_cross), enables a model to learn separate weights for each combination of features. Here, we will create a new feature that is the cross of Age and Type. Note that `crossed_column` does not build the full table of all possible combinations (which could be very large). Instead, it is backed by a `hashed_column`, so you can choose how large the table is.


    animal_type = feature_column.categorical_column_with_vocabulary_list(
          'Type', ['Cat', 'Dog'])
    age_type_feature = feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=5)
    feature_columns.append(feature_column.indicator_column(age_type_feature))


    # ### Create a feature layer
    # Now that we have defined our feature columns, we will use a [DenseFeatures](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/DenseFeatures) layer to input them to our Keras model.

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


    # Create an input pipeline.

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


    # ## Create, compile, and train the model

    model = tf.keras.Sequential([
      feature_layer,
      #layers.Dense(128, activation='relu'),
      #layers.Dense(128, activation='relu'),
      layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)


    # ## Save the model

    model.save(args.job_dir)


if __name__ == '__main__':
    args = get_args()

    # Set verbosity
    tf.compat.v1.logging.set_verbosity('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train_and_evaluate(args)


# Key point: You will typically see best results with deep learning with much larger and more complex datasets. When working with a small dataset like this one, we recommend using a decision tree or random forest as a strong baseline. The goal of this tutorial is not to train an accurate model, but to demonstrate the mechanics of working with structured data, so you have code to use as a starting point when working with your own datasets in the future.

