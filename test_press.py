#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tfperf
import argparse
import yaml

if __name__ == "__main__":
    tfperf.run_press(data_path="./data/20210405/",
                     url="localhost:8500",
                     model_name="dnn_model",
                     model_signature_name="outputs",
                     batch_size=10,
                     feature_dict={
                       "value_fea": (tf.float32, [1000], 1000),
                       "id_fea": (tf.int64, [100], 100),
                     },
                     label_dict={
                       "label": (tf.float32, [], 1),
                     },
                     parallel=16,
                     duration=30)
