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
embedding_dim = 32

def model_fn(features, labels, mode, params):
    print ("features %s" % features)
    value_fea = features['value_fea']
    id_fea = features['id_fea']
    id_fea_len = id_fea.shape[1]
    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    if is_training:
        initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    else:
        initializer=tf.keras.initializers.Zeros()
    if mode == tf.estimator.ModeKeys.PREDICT:
        tfra.dynamic_embedding.enable_inference_mode()
    dynamic_embeddings = tfra.dynamic_embedding.get_variable(
        name="dynamic_embeddings",
        dim=embedding_dim,
        initializer=initializer,
        trainable=is_training)
    id_fea_shape = id_fea.shape
    id_fea = tf.reshape(id_fea, [-1])
    id_fea_val, id_fea_idx = tf.unique(id_fea)
    raw_embs = tfra.dynamic_embedding.embedding_lookup(
        params=dynamic_embeddings,
        ids=id_fea_val,
        name="embs")
    embs = tf.gather(raw_embs, id_fea_idx)
    embs = tf.reshape(embs, [-1, id_fea_len * embedding_dim])
    inputs = tf.concat([value_fea, embs], axis=-1)
    #inputs = value_fea
    # three layer mlp
    out, weight_and_bias = mlp.multilayer_perception(inputs, [256, 64, 1])
    logits = tf.reshape(out, [-1])
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_global_step()
        loss = tf.math.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        opt = de.DynamicEmbeddingOptimizer(tf.compat.v1.train.AdamOptimizer())
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.math.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    elif mode == tf.estimator.ModeKeys.PREDICT:
        pass
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = opt.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        outputs = {
            "outputs": tf.estimator.export.PredictOutput(outputs=logits)
        }
        predictions = {
            "logits": logits,
            "embs": embs,
            "raw_embs": tf.reshape(tf.slice(raw_embs, [0, 0], [100, 32]), [5, -1])
            #"weight": tf.reshape(weight_and_bias[0]["weight_0"].read_value(), [5, -1])
        }
        for k,v in features.items():
            predictions.update({k: v})
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=outputs)
