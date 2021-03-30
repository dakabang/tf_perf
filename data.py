#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

import random
from random import seed
from random import randint
import os

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
            else:
                raise ValueError("invalid dtype %s" % dtype)
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())
    print ("write tfrd %s done!" % filename)
    return filename

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def gen_parse_input_fn(items):
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
