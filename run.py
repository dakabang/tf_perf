#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import data
import importlib
import argparse
import os
import json

from multiprocessing import Process


def distributed_run(model_name, mode, input_dict, label_dict, 
                    data_path, ckpt_path, batch_size, shuffle_size, 
                    model_dir, node_num, port, warm_start_from=None):
    # Generate random data if needed
    if not os.path.exists(data_path):
        data_path = data.gen_random_tfrd(
            items, 1000, filename="data.tfrd" if not data_path else data_path)
    if node_num > 1:
        # distributed mode, use tf.estimator.train_and_evaluate
        # ${node_num} worker, ${node_num} ps servers
        cluster_config = dict()
        cluster_config['chief'] = ['localhost:%d' % port]
        cluster_config.update({'worker': ['localhost:%d' % (port + i + 1) for i in range(node_num - 1)]})
        cluster_config.update({'ps': ['localhost:%d' % (port + i + node_num) for i in range(node_num)]})
        print ('cluster_config: ', cluster_config)
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": cluster_config,
            "task": {"type": "chief", "index": 0}
        })
        chief_process = Process(target=run, args=(model_name, mode, input_dict, 
                                                  label_dict, data_path, ckpt_path, 
                                                  batch_size, shuffle_size, 
                                                  model_dir, warm_start_from, node_num, ))
        chief_process.start()
        # Start worker process
        worker_process = []
        for i in range(node_num - 1):
            os.environ['TF_CONFIG'] = json.dumps({
                "cluster": cluster_config,
                "task": {"type": "worker", "index": i}
            })
            p = Process(target=run, args=(model_name, mode, input_dict, 
                                          label_dict, data_path, ckpt_path, 
                                          batch_size, shuffle_size, 
                                          model_dir, warm_start_from, node_num, ))
            p.start()
            worker_process.append(p)
        # Start ps process
        ps_process = []
        for i in range(node_num):
            os.environ['TF_CONFIG'] = json.dumps({
                "cluster": cluster_config,
                "task": {"type": "ps", "index": i}
            })
            p = Process(target=run, args=(model_name, mode, input_dict, 
                                          label_dict, data_path, ckpt_path, 
                                          batch_size, shuffle_size, 
                                          model_dir, warm_start_from, node_num, ))
            p.start()
            ps_process.append(p)
        # Join all process for gracefully exit
        for p in worker_process:
            p.join()
        for p in ps_process:
            p.join()
        chief_process.join()
    else:
        run(model_name, mode, input_dict, 
            label_dict, data_path, ckpt_path, 
            batch_size, shuffle_size, 
            model_dir, warm_start_from, node_num)


def run(model_name, mode, input_dict,
        label_dict, data_path, ckpt_path, batch_size, 
        shuffle_size, model_dir, warm_start_from, ps_num):
    print ("TF_CONFIG ", os.environ['TF_CONFIG'])
    # generate random dataset
    items = input_dict
    items.extend(label_dict)
    parse_input_fn = data.gen_parse_input_fn(items)
    model = importlib.import_module(model_name)
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_checkpoints_secs=None
    )
    trainer = tf.estimator.Estimator(
        model_dir=ckpt_path,
        model_fn=model.model_fn,
        params={'batch_size': batch_size, 'ps_num': ps_num},
        config=run_config,
        warm_start_from=warm_start_from)
    print ('%s model from %s' % (mode, data_path))
    def input_fn():
        dataset = tf.data.TFRecordDataset([data_path])
        dataset = dataset.map(parse_input_fn).prefetch(batch_size).batch(batch_size)
        if shuffle_size != 0:
            dataset = dataset.shuffle(shuffle_size)
        return dataset
    ops = mode.split("_")
    if 'train' in ops and 'eval' in ops:
        # For test purpose, we don't shard dataset only when we need it
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn())
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn())
        tf.estimator.train_and_evaluate(trainer, train_spec, eval_spec)
    else:
        for operation in ops:
            if operation == "train":
                trainer.train(input_fn)
            elif operation == "eval":
                trainer.eval(input_fn)
            elif operation == "predict":
                predictions = trainer.predict(input_fn)
                count = 0
                np.set_printoptions(threshold=1000)
                for x in predictions:
                    print (x["logid"], x["logits"])
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
    parser.add_argument('--shuffle_size', type=int, default=0, help='shuffle size')
    parser.add_argument('--warm_start_from', type=str, help='filepath to a checkpoint or SavedModel to warm-start from')
    parser.add_argument('--node_num', type=int, default=1, help='number of parameter server and workers')
    parser.add_argument('--port', type=int, default=7553, help='port of worker/ps when using distributed traning')
    args = parser.parse_args()
    input_dict = [
        ("logid", tf.string, [1], 1),
        ("value_fea", tf.float32, [1000], 1000),
        ("id_fea", tf.int64, [100], 100),
    ]
    label_dict = [("label", tf.float32, [], 1)]
    distributed_run(
        model_name=args.model_name,
        mode=args.mode,
        input_dict=input_dict,
        label_dict=label_dict,
        data_path=args.data_path,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size, 
        shuffle_size=args.shuffle_size,
        model_dir=args.model_path,
        warm_start_from=args.warm_start_from,
        node_num=args.node_num, port=args.port)

