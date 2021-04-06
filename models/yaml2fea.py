#!/usr/bin/env python
# coding=utf-8

import yaml
import os
import tensorflow as tf

def parse(feature_yaml):
    with open(feature_yaml, 'r') as io:
        try:
            config = yaml.safe_load(io)
        except Exception as ex:
            raise ex
    label_dict = []
    feature_dict = []
    for k,v in config.items():
        if k == "labels":
            for label_name in v:
                label_dict.append((label_name, tf.float32, [], 1))
        else:
            assert(isinstance(v, dict))
            for featype, feakeys in v.items():
                if featype not in ['dense', 'sparse', 'dense_seq', 'sparse_seq']:
                    raise ValueError("invalid feature type %s" % featype)
                if featype == 'dense':
                    for name in feakeys:
                        feature_dict.append((name, tf.float32, [], 1))
                elif featype == 'dense_seq':
                    raise ValueError("feature type %s not supported yet" % featype)
                elif featype == 'sparse':
                    for name in feakeys:
                        feature_dict.append((name, tf.int64, [], 1))
                elif featype == 'sparse_seq':
                    for name in feakeys:
                        feature_dict.append((name, tf.int64, [50], 50))
    return feature_dict, label_dict
