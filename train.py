#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import data
import importlib
import argparse

def run(model_name, mode, ckpt_path, batch_size, model_dir):
    # generate random dataset
    items = [
        ("value_fea", tf.float32, [1000], 1000),
        ("id_fea", tf.int64, [100], 100),
        ("label", tf.float32, [], 1)
    ]
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
    if mode == "train":
        trainer.train(input_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='model selected to run')
    parser.add_argument('mode', type=str, help='run mode: train/eval/predict')
    parser.add_argument('ckpt_path', type=str, help='checkpoint save path')
    parser.add_argument('model_path', type=str, help='save model path(for serving)')
    parser.add_argument('batch_size', type=int, help='batch size')
    args = parser.parse_args()
    run(
        model_name=args.model,
        mode=args.mode,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size, 
        model_dir=args.model_path)
