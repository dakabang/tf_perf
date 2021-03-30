#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from tensorflow.keras.layers import Dense

import mlp

def model_fn(features, labels, mode, params):
    batch_size = params['batch_size']
    value_fea = features['value_fea']
    id_fea = features['id_fea']
    value_fea_len = value_fea.shape[1]
    id_fea_len = id_fea.shape[1]
    dynamic_embeddings = tfra.dynamic_embedding.get_variable(
        name="dynamic_embeddings",
        dim=32,
        initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    id_fea = tf.reshape(id_fea, [-1])
    id_fea_val, id_fea_idx = tf.unique(id_fea)
    embs = tfra.dynamic_embedding.embedding_lookup(
        params=dynamic_embeddings,
        ids=id_fea_val,
        name="embs")
    embs = tf.gather(embs, id_fea_idx)
    value_fea = tf.reshape(value_fea, [-1, value_fea_len])
    embs = tf.reshape(embs, [-1, id_fea_len * 32])
    inputs = tf.concat([value_fea, embs], axis=-1)
    # three layer mlp
    out = mlp.multilayer_perception(inputs, [256, 64, 1])
    logits = tf.reshape(out, [batch_size])
    loss = tf.math.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    opt = de.DynamicEmbeddingOptimizer(tf.compat.v1.train.AdamOptimizer())
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = opt.minimize(loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    else:
        raise ValueError("unsupported mode: %s" % mode)
