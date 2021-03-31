#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

def parse_tfrd_fn(example_proto):
    features = {
        "value_fea": tf.io.FixedLenFeature([1000], dtype=tf.float32),
        "id_fea": tf.io.FixedLenFeature([100], dtype=tf.int64),
        "labels": tf.io.FixedLenFeature([1], dtype=tf.float32)
    }
    return tf.io.parse_example(example_proto, features)

def input_fn(batch_size):
    dataset = tf.data.TFRecordDataset(["./data.tfrd"])
    dataset = dataset.map(parse_tfrd_fn).prefetch(batch_size).batch(batch_size)
    return dataset

def model_fn(features, labels, mode, params):
    print (features)
    exit(0)


if __name__ == "__main__":
    es = tf.estimator.Estimator(model_fn=model_fn)
    es.train(input_fn=lambda: input_fn(10))
