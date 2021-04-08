#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tfperf
import argparse
import yaml

if __name__ == "__main__":
    tfperf.run(model_name="models.dnn_model", 
               feature_dict={
                   "value_fea": (tf.float32, [1000], 1000),
                   "id_fea": (tf.int64, [100], 100),
               },
               label_dict={
                   "label": (tf.float32, [], 1),
               },
               data_and_mode=[
                   ("./data/20210405", "train_eval"),
               ],
               batch_size=10,
               ckpt_path="ckpt",
               saved_model_path="saved_model",
               autogen_data=True,
               autogen_data_num=10000,
               max_file_size=1024 * 1024 * 10,
               local_distributed=True,
               cluster_config={"chief_num": 1, "worker_num": 2, "ps_num": 3},
               profile=True)
