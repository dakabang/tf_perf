#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import data
import importlib
import argparse
import os

def run(model_name, 
        mode, 
        input_dict,
        label_dict,
        data_path,
        ckpt_path, 
        batch_size, 
        model_dir="",
        warm_start_from=None):
    # generate random dataset
    items = input_dict
    items.extend(label_dict)
    filepath = data_path
    if not os.path.exists(data_path):
        filepath = data.gen_random_tfrd(
            items, 1000, filename="data.tfrd" if not data_path else data_path)
    parse_input_fn = data.gen_parse_input_fn(items)
    model = importlib.import_module(model_name)
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_checkpoints_secs=None
    )
    trainer = tf.estimator.Estimator(
        model_dir=ckpt_path,
        model_fn=model.model_fn,
        params={'batch_size': batch_size},
        config=run_config,
        warm_start_from=warm_start_from)
    print ('%s model from %s' % (mode, filepath))
    def input_fn():
        dataset = tf.data.TFRecordDataset([filepath])
        return dataset.map(parse_input_fn).prefetch(batch_size).batch(batch_size)
    ops = mode.split("_")
    for op in ops:
        if mode == "train":
            trainer.train(input_fn)
        elif mode == "eval":
            trainer.eval(input_fn)
        elif mode == "predict":
            predictions = trainer.predict(input_fn)
            for x in predictions:
                print (x)
                break
        else:
            raise ValueError("invalid mode %s" % mode)
    # Save model for serving
    if model_dir:
        print ("export model for serving to %s..." % model_dir)
        serving_input_dict = []
        for x in input_dict:
            serving_input_dict.append((x[0], x[1], [1] + x[2], x[3]))
        trainer.export_saved_model(
            model_dir, 
            data.serving_input_fn(serving_input_dict, batch_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model selected to run')
    parser.add_argument('--mode', type=str, help='run mode: train/eval/predict')
    parser.add_argument('--data_path', type=str, default="", help='run mode: train/eval/predict')
    parser.add_argument('--ckpt_path', type=str, help='checkpoint save path')
    parser.add_argument('--model_path', type=str, help='save model path(for serving)')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--warm_start_from', type=str, help='filepath to a checkpoint or SavedModel to warm-start from')
    args = parser.parse_args()
    input_dict = [
        ("value_fea", tf.float32, [1000], 1000),
        ("id_fea", tf.int64, [100], 100),
    ]
    label_dict = [("label", tf.float32, [], 1)]
    run(
        model_name=args.model_name,
        mode=args.mode,
        input_dict=input_dict,
        label_dict=label_dict,
        data_path=args.data_path,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size, 
        model_dir=args.model_path,
        warm_start_from=args.warm_start_from)

