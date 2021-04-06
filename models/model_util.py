#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

class MultiTaskType(object):
    SB = "sb"
    MMOE = "mmoe"
    NONE = "none"
    ESMM_SB = "esmm_sb"
    ESMM_MMOE = "esmm_mmoe"
    _MTL_COLLECTIONS = [
        SB, MMOE, ESMM_SB, ESMM_MMOE
    ]

def default_partition_fn(keys, shard_num):
    return tf.cast(keys % shard_num, dtype=tf.int32)


def dense_pooling(inputs, combiner="sum"):
    if combiner not in ("mean", "sum"):
        raise ValueError("combiner must be one of 'mean'' or 'sum'")

    if combiner == "sum":
        embeddings = tf.reduce_sum(inputs, axis=0)
    elif combiner == "mean":
        embeddings = tf.reduce_mean(inputs, axis=0)
    else:
        assert False, "Unrecognized combiner"
    return embeddings


def dense_to_sparse(tensor, eos_token=0):
    indices = tf.where(
        tf.not_equal(tensor, tf.constant(eos_token, tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.dtypes.int64)
    outputs = tf.SparseTensor(indices, values, shape)
    return outputs


def get_learning_rate(lr, use_decay):
    if use_decay:
        def_rate = tf.constant(0.00001, dtype=tf.float32)
        learning_rate = tf.compat.v1.train.exponential_decay(lr, tf.compat.v1.train.get_global_step(), 50000, 0.98,
                                                             staircase=True)
        decay_learning_rate = tf.cond(tf.less(def_rate, learning_rate), lambda: learning_rate, lambda: def_rate)
        return decay_learning_rate
    else:
        return tf.constant(lr, dtype=tf.float32)

def get_auc(labels, preds, name="auc"):
    auc_metric = tf.keras.metrics.AUC(name=name)
    auc_metric.update_state(y_true=labels, y_pred=preds)
    auc = auc_metric.result()
    auc_metric.reset_states()
    return auc

def fully_connected_with_bn_ahead(inputs, num_outputs, l2_reg, scope, activation_fn, train_phase, use_bn=False):
    if use_bn:
        inputs = tf.compat.v1.layers.batch_normalization(inputs, training=train_phase)
    net = tf.compat.v1.layers.dense(inputs=inputs,
                                    units=num_outputs,
                                    activation=activation_fn,
                                    use_bias=True,
                                    kernel_initializer=None,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                    name="mlp_{}".format(scope),
                                    trainable=train_phase,
                                    reuse=tf.compat.v1.AUTO_REUSE
                                    )
    return net


def experts_layer(shared_bottom, params, is_training, scope=None, use_bn=False):
    # expert
    expert_layers = params["parameters"]["expert_layers"]
    expert_dim = params["parameters"]["expert_dim"]
    l2_reg = params["parameters"]["l2_reg"]

    expert_embs = []
    n = len(expert_layers)  # expert_num
    E = expert_dim

    for i, layer in enumerate(expert_layers):
        expert_emb = build_fc_net(shared_bottom,
                                  layer,
                                  l2_reg,
                                  is_training,
                                  scope='expert_{}'.format(i),
                                  use_bn=use_bn)
        expert_embs.append(expert_emb)

    experts = tf.reshape(tf.concat(expert_embs, axis=1),
                         shape=(-1, n, E))  # (B, n, E)

    # gate = build_fc_net(shared_bottom,
    #                     [n],
    #                     l2_reg,
    #                     is_training,
    #                     activate=tf.nn.softmax,
    #                     scope='gate_{}'.format(scope))  # (B, n)

    if use_bn:
        experts = tf.compat.v1.layers.batch_normalization(experts, training=is_training)
    gate = tf.compat.v1.layers.dense(inputs=shared_bottom,
                                     units=n,
                                     activation=tf.nn.softmax,
                                     use_bias=True,
                                     kernel_initializer=None,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                     name="gate_{}".format(scope),
                                     trainable=is_training
                                     )   # (B, n)

    gate = tf.expand_dims(gate, axis=2)  # (B, n, 1)
    expert_out = tf.reduce_sum(
        gate * experts,  # (B, n, E)
        axis=1
    )  # (B, E)

    return expert_out  # (B, E)


def build_fc_net(inputs, hidden_units, l2_reg, is_training, activate=tf.nn.relu, scope=None, use_bn=False):
    for idx, units in enumerate(hidden_units):
        if use_bn:
            inputs = tf.compat.v1.layers.batch_normalization(inputs, training=is_training)
        inputs = tf.compat.v1.layers.dense(inputs=inputs,
                                           units=units,
                                           activation=activate,
                                           use_bias=True,
                                           kernel_initializer=None,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                           name="mlp_{}_{}".format(scope, idx),
                                           trainable=is_training,
                                           reuse=tf.compat.v1.AUTO_REUSE
                                           )
    return inputs

def build_fc_nets(inputs, hidden_units, l2_reg, is_training, activates=[], scope=None, use_bn=False):
    assert len(hidden_units) == len(activates)
    for idx, units in enumerate(hidden_units):
        if use_bn:
            inputs = tf.compat.v1.layers.batch_normalization(inputs, training=is_training)
        inputs = tf.compat.v1.layers.dense(inputs=inputs,
                                           units=units,
                                           activation=activates[idx],
                                           use_bias=True,
                                           kernel_initializer=None,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                           name="mlp_{}_{}".format(scope, idx),
                                           trainable=is_training,
                                           reuse=tf.compat.v1.AUTO_REUSE
                                           )
    return inputs

def safe_log_sigmoid(logits):
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)
    return tf.negative(relu_logits - logits + tf.math.log1p(tf.exp(neg_abs_logits)))

def get_sample_logits(logits, sample_rate, sample_bias=0.0):
    return tf.cond(tf.less(sample_bias, 1e-6),
	lambda : tf.add(logits, tf.negative(tf.math.log(sample_rate))),
	lambda : tf.add(safe_log_sigmoid(logits), tf.negative(tf.math.log(sample_rate))),
    )


if __name__ == '__main__':
    pass
