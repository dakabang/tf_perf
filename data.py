#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

import random
from random import seed
from random import randint
import os
import uuid

def gen_random_tfrd(items, num_samples, filename="data.tfrd"):
    assert(len(items) > 0)
    assert(isinstance(items[0], tuple))
    if os.path.exists(filename):
        os.unlink(filename)
    writer = tf.io.TFRecordWriter(filename)
    for _ in range(num_samples):
        feature = dict()
        for item in items:
            name = item[0]
            dtype = item[1]
            shape = item[2]
            length = item[3]
            if dtype == tf.int64:
                feature.update({name: _int64_feature([randint(0, 10000) for _ in range(length)])})
            elif dtype == tf.float32:
                feature.update({name: _float_feature([random.random() for _ in range(length)])})
        feature.update({"logid": _bytes_feature((str(uuid.uuid4()).encode()))})
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())
    print ("write tfrd %s done!" % filename)
    return filename

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def gen_parse_input_fn(items):
    assert(len(items) > 0)
    features = dict()
    for item in items:
        name = item[0]
        dtype = item[1]
        shape = item[2]
        length = item[3]
        features.update({name: tf.io.FixedLenFeature(shape, dtype)})
    def parse_tfrd_fn(example_proto):
        example = tf.io.parse_example(example_proto, features=features)
        return {k:v for k,v in example.items() if k!='label'}, example['label']
    return parse_tfrd_fn


def serving_input_fn(items, batch_size):
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    assert(len(items) > 0)
    features = dict()
    for item in items:
        name = item[0]
        dtype = item[1]
        shape = item[2]
        length = item[3]
        features.update({name: tf.compat.v1.placeholder(dtype, shape, name)})
    print ("serving input %s" % features)
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features, batch_size)
