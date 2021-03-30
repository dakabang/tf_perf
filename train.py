#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import data
import importlib

def train(model_name, batch_size, model_dir="./model"):
    # generate random dataset
    items = [
        ("value_fea", tf.float32, [1000], 1000),
        ("id_fea", tf.int64, [100], 100),
        ("label", tf.float32, [], 1)
    ]
    batch_size = 10
    filepath = data.gen_random_tfrd(items, 1000)
    parse_input_fn = data.gen_parse_input_fn(items)

    model = importlib.import_module(model_name)
    trainer = tf.estimator.Estimator(
        model_dir=model_dir, 
        model_fn=model.model_fn, 
        params={'batch_size': batch_size})
    print ('train model from %s' % filepath)
    def input_fn():
        dataset = tf.data.TFRecordDataset([filepath])
        return dataset.prefetch(batch_size).batch(batch_size).map(parse_input_fn)
    trainer.train(input_fn)

if __name__ == "__main__":
    train("dnn_model", batch_size=10, model_dir="dnn_model")
