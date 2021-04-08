#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from tensorflow.keras.layers import Dense

from models import mlp
embedding_dim = 32

def model_fn(features, labels, mode, params):
    print ("features %s labels %s" % (features, labels))
    value_fea = features['value_fea']
    id_fea = features['id_fea']
    id_fea_len = id_fea.shape[1]
    batch_size = params["batch_size"]
    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    devices = None
    if is_training:
        devices = ["/job:ps/replica:0/task:{}/CPU:0".format(i) for i in range(params['ps_num'])]
        initializer=tf.keras.initializers.RandomNormal(0.0, 0.1)
    else:
        devices = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for i in range(params['ps_num'])]
        initializer=tf.keras.initializers.Zeros()
    if mode == tf.estimator.ModeKeys.PREDICT:
        tfra.dynamic_embedding.enable_inference_mode()
    dynamic_embeddings = tfra.dynamic_embedding.get_variable(
        name="dynamic_embeddings",
        dim=embedding_dim,
        devices=devices,
        initializer=initializer,
        trainable=is_training,
        init_size=8192)
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
    inputs = tf.compat.v1.layers.batch_normalization(inputs, training=is_training)
    print ("inputs shape %s" % inputs)
    # three layer mlp
    out, inners = mlp.multilayer_perception(inputs, [1024, 512, 256, 64, 1])
    logits = tf.reshape(out, [-1])
    tf.compat.v1.summary.histogram("logits", logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_global_step()
        loss = tf.math.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['label'], logits=logits))
        opt = de.DynamicEmbeddingOptimizer(tf.compat.v1.train.AdamOptimizer(beta1=0.9, beta2=0.999))
        tf.compat.v1.summary.scalar("loss", loss)
        tf.compat.v1.summary.scalar("global_step", global_step)
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.math.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['label'], logits=logits))
        tf.compat.v1.summary.scalar("loss", loss)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        pass
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        train_op = opt.minimize(loss, global_step=global_step)
        train_op = tf.group([train_op, update_ops])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        outputs = {
            "outputs": tf.estimator.export.PredictOutput(outputs=logits)
        }
        predictions = {
            "logid": features["logid"],
            "logits": logits,
        }
        for k,v in features.items():
            predictions.update({k: v})
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=outputs)
