#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import random 
from random import seed
from random import randint
import os
import uuid

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


def generate_data(feature_dict, sample_num, data_path,
                  max_file_size, overwrite, gen_logid):
    if os.path.exists(data_path):
        assert(os.path.isdir(data_path))
        return
    else:
        os.makedirs(data_path)
    file_prefix = "part-"
    file_index = 0
    files = []
    tfrd_path = os.path.join(data_path, (file_prefix + str(file_index).zfill(5)))
    tfrd_writer = tf.io.TFRecordWriter(tfrd_path)
    files.append(tfrd_path)
    file_size = 0
    for _ in range(sample_num):
        features = dict()
        for feakey, feainfo in feature_dict.items():
            dtype = feainfo[0]
            shape = feainfo[1]
            length = feainfo[2]
            if dtype == tf.int64:
                features.update({feakey: _int64_feature([randint(0, 1000000) for _ in range(length)])})
            elif dtype == tf.float32:
                features.update({feakey: _float_feature([random.random() for _ in range(length)])})
            else:
                raise ValueError("invalid dtype %s" % dtype)
        if gen_logid:
            features.update({"logid": _bytes_feature(str(uuid.uuid4()).encode())})
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        buf = example_proto.SerializeToString()
        tfrd_writer.write(buf)
        file_size += len(buf)
        if file_size > max_file_size:
            file_index += 1
            file_size = 0
            tfrd_path = os.path.join(data_path, (file_prefix + str(file_index).zfill(5)))
            tfrd_writer = tf.io.TFRecordWriter(tfrd_path)
            files.append(tfrd_path)
    return files


def generate_parse_tfrd_fn(feature_dict, label_dict):
    print (feature_dict)
    print (label_dict)
    features = dict()
    for feakey, feainfo in feature_dict.items():
        dtype = feainfo[0]
        shape = feainfo[1]
        length = feainfo[2]
        features.update({feakey: tf.io.FixedLenFeature(shape, dtype)})
    features.update({"logid": tf.io.FixedLenFeature([], tf.string)})
    for labelkey, labelinfo in label_dict.items():
        dtype = labelinfo[0]
        shape = labelinfo[1]
        length = labelinfo[2]
        features.update({labelkey: tf.io.FixedLenFeature(shape, dtype)})
    def parse_tfrd_fn(example_proto):
        example = tf.io.parse_example(example_proto, features=features)
        return {k:v for k,v in example.items() if k not in label_dict.keys()}, \
                {k:v for k,v in example.items() if k in label_dict.keys()}
    return parse_tfrd_fn


def serving_input_fn(feature_dict, batch_size):
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    features = dict()
    for feakey, feainfo in feature_dict.items():
        dtype = feainfo[0]
        shape = feainfo[1]
        length = feainfo[2]
        features.update({feakey: tf.compat.v1.placeholder(dtype, shape, feakey)})
    features.update({"logid": tf.compat.v1.placeholder(tf.string, [], "logid")})
    print ("serving input %s" % features)
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features, batch_size)
