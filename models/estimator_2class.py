#!/usr/bin/env python
# coding=utf-8

# !/usr/bin/env python
# encoding=utf-8
import tensorflow as tf
from absl import logging
import math

from util.model_util import fully_connected_with_bn_ahead, get_learning_rate, get_sample_logits

SPARSE_MASK = ['sparse_'+str(i) for i in list(range(101, 119)) + list(range(746, 754)) + list(range(201, 220)) + list(range(622, 637)) + list(range(1501, 1509))]
SPARSE_SEQ_MASK = []
DENSE_MASK = ['dense_'+str(i) for i in [205, 207] + list(range(1401, 1410)) + list(range(1601, 1610))]
DENSE_SEQ_MASK = []
goods_id_feat = 'sparse_601'
INIT_SIZE = math.pow(2, 20) #1048576

input_dict = {

}

def model_fn(features, labels, mode, params):
    logging.info('mode: %s, labels: %s, params: %s, features: %s', mode, labels, params, features)
    if params["args"].get("addon_embedding"):
        import tensorflow_recommenders_addons as tfra
        import tensorflow_recommenders_addons.dynamic_embedding as dynamic_embedding
    else:
        import tensorflow.dynamic_embedding as dynamic_embedding

    logging.info("------ build hyper parameters -------")
    embedding_size = params["parameters"]["embedding_size"]
    learning_rate = params["parameters"]["learning_rate"]
    use_bn = params["parameters"]["use_bn"]

    feat = params['features']
    sparse_feat_list = list(set(feat["sparse"]) - set(SPARSE_MASK)) if 'sparse' in feat else []
    sparse_seq_feat_list = list(set(feat["sparse_seq"]) - set(SPARSE_SEQ_MASK)) if 'sparse_seq' in feat else []
    sparse_seq_feat_list = []
    #dense_feat_list = list(set(feat["dense"]) - set(DENSE_MASK)) if 'dense' in feat else []
    # hashtable v1/v2 image均无法同时关bn和mask dense_feat
    dense_feat_list = []
    dense_seq_feat_list = list(set(feat["dense_seq"]) - set(DENSE_SEQ_MASK)) if 'dense_seq' in feat else []

    sparse_feat_num = len(sparse_feat_list)
    sparse_seq_num = len(sparse_seq_feat_list)
    dense_feat_num = len(dense_feat_list)
    dense_seq_feat_num = len(dense_seq_feat_list)

    all_features = (sparse_feat_list +
                    sparse_seq_feat_list +
                    dense_feat_list +
                    dense_seq_feat_list)

    batch_size = tf.shape(features[goods_id_feat])[0]
    logging.info("------ show batch_size: {} -------".format(batch_size))

    level_0_feats = list(set(params.get('level_0_feat_list')) & set(all_features))
    logging.info('level_0_feats: {}'.format(level_0_feats))
    new_features = dict()
    if params["args"].get("level_flag") and params["args"].get("job_type") == "export":
        for feature_name in features:
            if feature_name in level_0_feats:
                new_features[feature_name] = tf.reshape(tf.tile(tf.reshape(features[feature_name], [1, -1]), [batch_size, 1]), [batch_size, -1])
            else:
                new_features[feature_name] = features[feature_name]
        features = new_features

    l2_reg = params["parameters"]["l2_reg"]
    is_export = params["args"]["is_export"]
    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    has_label = True if 'is_imp' in features else False
    logging.info("is_training: {}, has_label: {}".format(is_training, has_label))

    logging.info("------ build embedding -------")
    # def partition_fn(keys, shard_num=params["parameters"]["ps_nums"]):
    #     return tf.cast(keys % shard_num, dtype=tf.int32)
    if is_training:
        devices_info = ["/job:ps/replica:0/task:{}/CPU:0".format(i) for i in range(params["parameters"]["ps_nums"])]
        initializer = tf.compat.v1.truncated_normal_initializer(0.0, 1e-2)
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for i in range(params["parameters"]["ps_nums"])]
        initializer = tf.compat.v1.zeros_initializer()

    logging.info("------ dynamic_embedding devices_info is {}-------".format(devices_info))
    if mode == tf.estimator.ModeKeys.PREDICT:
        dynamic_embedding.enable_inference_mode()
    deep_dynamic_variables = dynamic_embedding.get_variable(
        name="deep_dynamic_embeddings",
        devices=devices_info,
        initializer=initializer,
        # partitioner=partition_fn,
        dim=embedding_size,
        trainable=is_training,
        #init_size=INIT_SIZE
    )

    sparse_feat = None
    sparse_unique_ids = None
    if sparse_feat_num > 0:
        logging.info("------ build sparse feature -------")
        id_list = sorted(sparse_feat_list)

        ft_sparse_idx = tf.concat([tf.reshape(features[str(i)], [-1, 1]) for i in id_list], axis=1)

        sparse_unique_ids, sparse_unique_idx = tf.unique(tf.reshape(ft_sparse_idx, [-1]))

        sparse_weights = dynamic_embedding.embedding_lookup(
            params=deep_dynamic_variables,
            ids=sparse_unique_ids,
            name="deep_sparse_weights"
        )
        if params["args"].get("zero_padding"):
            sparse_weights = tf.reshape(sparse_weights, [-1, embedding_size])
            sparse_weights = tf.where(
                tf.not_equal(tf.expand_dims(sparse_unique_ids, axis=1)
                             , tf.zeros_like(tf.expand_dims(sparse_unique_ids, axis=1)))
                , sparse_weights, tf.zeros_like(sparse_weights))

        sparse_weights = tf.gather(sparse_weights, sparse_unique_idx)
        sparse_feat = tf.reshape(sparse_weights, shape=[batch_size, sparse_feat_num * embedding_size])

    sparse_seq_feat = None
    sparse_seq_unique_ids = None
    if sparse_seq_num > 0:
        logging.info("---- build sparse seq feature ---")
        if params["args"].get("merge_sparse_seq"):
            sparse_seq_name_list = sorted(sparse_seq_feat_list) #[B, s1], [B, s2], ... [B, sn]
            ft_sparse_seq_ids = tf.concat(
                [
                    tf.reshape(features[str(i)], [batch_size, -1])
                    for i in sparse_seq_name_list
                ],
                axis=1
            ) #[B, [s1, s2, ...sn]] => [B, per_seq_len*seq_num]

            sparse_seq_unique_ids, sparse_seq_unique_idx = tf.unique(
                tf.reshape(ft_sparse_seq_ids, [-1])
            ) #[u], [B*per_seq_len*seq_num]

            sparse_seq_weights = dynamic_embedding.embedding_lookup(
                    params=deep_dynamic_variables,
                    ids=sparse_seq_unique_ids,
                    name="deep_sparse_seq_weights"
            ) #[u, e]

            deep_embed_seq = tf.where(
                tf.not_equal(
                    tf.expand_dims(sparse_seq_unique_ids, axis=1),
                    tf.zeros_like(
                        tf.expand_dims(sparse_seq_unique_ids, axis=1)
                    )
                ),
                sparse_seq_weights,
                tf.zeros_like(sparse_seq_weights)
            ) #[u, e]

            deep_embedding_seq = tf.reshape(
                tf.gather(deep_embed_seq, sparse_seq_unique_idx), #[B*per_seq_len*seq_num, e]
                shape=[batch_size, sparse_seq_num, -1, embedding_size]
            ) #[B, seq_num, per_seq_len, e]
            if params["parameters"]["combiner"] == "sum":
                tmp_feat = tf.reduce_sum(deep_embedding_seq, axis=2)
            else:
                tmp_feat = tf.reduce_mean(deep_embedding_seq, axis=2)
            sparse_seq_feat = tf.reshape(tmp_feat, [batch_size, sparse_seq_num*embedding_size]) #[B, seq_num*e]
        else:
            sparse_seq_feats = []
            sparse_ids = []
            for sparse_seq_name in sparse_seq_feat_list:
                sp_ids = features[sparse_seq_name]
                if params["args"].get("zero_padding2"):
                    sparse_seq_unique_ids, sparse_seq_unique_idx, _ = tf.unique_with_counts(
                        tf.reshape(sp_ids, [-1]))

                    deep_sparse_seq_weights = tf.reshape(dynamic_embedding.embedding_lookup(
                        params=deep_dynamic_variables,
                        ids=sparse_seq_unique_ids,
                        name="deep_sparse_weights_{}".format(sparse_seq_name)
                    ), [-1, embedding_size])

                    deep_embed_seq = tf.where(
                        tf.not_equal(tf.expand_dims(sparse_seq_unique_ids, axis=1)
                                     , tf.zeros_like(tf.expand_dims(sparse_seq_unique_ids, axis=1)))
                        , deep_sparse_seq_weights, tf.zeros_like(deep_sparse_seq_weights))

                    deep_embedding_seq = tf.reshape(
                        tf.gather(deep_embed_seq, sparse_seq_unique_idx),
                        shape=[batch_size, -1, embedding_size]
                    )

                    if params["parameters"]["combiner"] == "sum":
                        tmp_feat = tf.reduce_sum(deep_embedding_seq, axis=1)
                    else:
                        tmp_feat = tf.reduce_mean(deep_embedding_seq, axis=1)
                    sparse_ids.append(sparse_seq_unique_ids)
                    sparse_seq_feats.append(tf.reshape(tmp_feat, [batch_size, embedding_size]))
                else:
                    tmp_feat = dynamic_embedding.safe_embedding_lookup_sparse(
                        embedding_weights=deep_dynamic_variables,
                        sparse_ids=sp_ids,
                        combiner=params["parameters"]["combiner"],
                        name="safe_embedding_lookup_sparse"
                    )
                    temp_uni_id, _, _ = tf.unique_with_counts(tf.reshape(sp_ids.values, [-1]))
                    sparse_ids.append(temp_uni_id)
                    sparse_seq_feats.append(tf.reshape(tmp_feat, [batch_size, embedding_size]))

            sparse_seq_feat = tf.concat(sparse_seq_feats, axis=1)
            sparse_seq_unique_ids, _ = tf.unique(tf.concat(sparse_ids, axis=0))

    dense_feat = None
    if dense_feat_num > 0:
        logging.info("------ build dense feature -------")
        den_id_list = sorted(dense_feat_list)
        dense_feat_base = tf.concat([tf.reshape(features[str(i)], [-1, 1]) for i in den_id_list], axis=1)

        #deep_dense_w1 = tf.compat.v1.get_variable('deep_dense_w1',
        #                                          tf.TensorShape([dense_feat_num]),
        #                                          initializer=tf.compat.v1.truncated_normal_initializer(
        #                                              2.0 / math.sqrt(dense_feat_num)),
        #                                          dtype=tf.float32)
        #deep_dense_w2 = tf.compat.v1.get_variable('deep_dense_w2',
        #                                          tf.TensorShape([dense_feat_num]),
        #                                          initializer=tf.compat.v1.truncated_normal_initializer(
        #                                              2.0 / math.sqrt(dense_feat_num)),
        #                                          dtype=tf.float32)

        #w1 = tf.tile(tf.expand_dims(deep_dense_w1, axis=0), [tf.shape(dense_feat_base)[0], 1])
        #dense_input_1 = tf.multiply(dense_feat_base, w1)
        #dense_feat = dense_input_1
        dense_feat = dense_feat_base

    dense_seq_feat = None
    if dense_seq_feat_num > 0:
        logging.info("------ build dense seq feature -------")
        den_seq_id_list = sorted(dense_seq_feat_list)
        dense_seq_feat = tf.concat([tf.reshape(features[str(i[0])], [-1, i[1]]) for i in den_seq_id_list], axis=1)

    logging.info("------ join all feature -------")
    fc_inputs = tf.concat([x for x in [sparse_feat, sparse_seq_feat, dense_feat, dense_seq_feat] if x is not None],
                          axis=1)

    logging.info("---- tracy debug input is ----")
    logging.info(sparse_feat)
    logging.info(sparse_seq_feat)
    logging.info(dense_feat)
    logging.info(dense_seq_feat)
    logging.info(fc_inputs)

    logging.info("------ join fc -------")
    for idx, units in enumerate(params["parameters"]["hidden_units"]):
        fc_inputs = fully_connected_with_bn_ahead(inputs=fc_inputs,
                                                  num_outputs=units,
                                                  l2_reg=l2_reg,
                                                  scope="out_mlp_{}".format(idx),
                                                  activation_fn=tf.nn.relu,
                                                  train_phase=is_training,
                                                  use_bn=use_bn)
    y_deep_ctr = fully_connected_with_bn_ahead(
        inputs=fc_inputs,
        num_outputs=1,
        activation_fn=tf.identity,
        l2_reg=l2_reg,
        scope="ctr_mlp",
        train_phase=is_training,
        use_bn=use_bn
    )

    logging.info("------ build ctr out -------")
    sample_rate = params["args"]["sample_rate"]
    logit = tf.reshape(y_deep_ctr, shape=[-1], name="logit")
    sample_logit = get_sample_logits(logit, sample_rate)
    pred_ctr = tf.nn.sigmoid(logit, name="pred_ctr")
    sample_pred_ctr = tf.nn.sigmoid(sample_logit, name="sample_pred_ctr")

    logging.info("------ build predictions -------")
    preds = {
        'p_ctr': tf.reshape(pred_ctr, shape=[-1, 1]),
    }

    logging.info("---- deep_dynamic_variables.size ----")
    logging.info(deep_dynamic_variables.size())
    size = tf.identity(deep_dynamic_variables.size(), name="size")

    label_col = "is_clk"
    if params["args"].get("set_train_labels"):
        label_col = params["args"]["set_train_labels"]["1"]

    logging.info("------ build labels, label_col: {} -------".format(label_col))
    if has_label:
        labels_ctr = tf.reshape(features["is_clk"], shape=[-1], name="labels_ctr")

    if mode == tf.estimator.ModeKeys.PREDICT:
        logging.info("---- build tf-serving predict ----")
        pred_cvr = tf.fill(tf.shape(pred_ctr), 1.0)
        preds.update({
            'labels_cart': tf.reshape(pred_cvr, shape=[-1, 1]),
            'p_car': tf.reshape(features["dense_1608"], shape=[-1, 1]),
            'labels_cvr': tf.reshape(pred_cvr, shape=[-1, 1]),
            'p_cvr': tf.reshape(pred_cvr, shape=[-1, 1]),
        })
        if 'logid' in features:
            preds.update({
                'logid': tf.reshape(features["logid"], shape=[-1, 1])
                })
        if has_label:
            logging.info("------ build offline label -------")
            preds["labels_ctr"] = tf.reshape(labels_ctr, shape=[-1, 1])
        export_outputs = {
            "predict_export_outputs": tf.estimator.export.PredictOutput(outputs=preds)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=preds, export_outputs=export_outputs)

    logging.info("----all vars:-----" + str(tf.compat.v1.global_variables()))
    for var in tf.compat.v1.trainable_variables():
        logging.info("----trainable------" + str(var))

    logging.info("------ build metric -------")
    #loss = tf.reduce_mean(
    #    tf.compat.v1.losses.log_loss(labels=labels_ctr, predictions=sample_pred_ctr),
    #    name="loss")
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_ctr,logits=sample_logit), name="loss")
    ctr_auc = tf.compat.v1.metrics.auc(labels=labels_ctr, predictions=sample_pred_ctr, name="ctr_auc")

    label_ctr_avg  = tf.reduce_mean(labels_ctr, name="label_ctr_avg")
    real_pred_ctr_avg  = tf.reduce_mean(pred_ctr, name="real_pred_ctr_avg")
    sample_pred_ctr_avg  = tf.reduce_mean(sample_pred_ctr, name="pred_ctr_avg")
    sample_pred_bias_avg = tf.add(sample_pred_ctr_avg, tf.negative(label_ctr_avg), name="pred_bias_avg")
    tf.compat.v1.summary.histogram('labels_ctr', labels_ctr)
    tf.compat.v1.summary.histogram('pred_ctr', sample_pred_ctr)
    tf.compat.v1.summary.histogram('real_pred_ctr', pred_ctr)

    tf.compat.v1.summary.scalar('label_ctr_avg', label_ctr_avg)
    tf.compat.v1.summary.scalar('pred_ctr_avg', sample_pred_ctr_avg)
    tf.compat.v1.summary.scalar('real_pred_ctr_avg', real_pred_ctr_avg)
    tf.compat.v1.summary.scalar('pred_bias_avg', sample_pred_bias_avg)
    tf.compat.v1.summary.scalar('loss', loss)
    tf.compat.v1.summary.scalar('ctr_auc', ctr_auc[1])

    logging.info("------ compute l2 reg -------")
    if params["parameters"]["use_l2"]:
        all_unique_ids, _ = tf.unique(
            tf.concat([x for x in [sparse_unique_ids, sparse_seq_unique_ids] if x is not None], axis=0))

        all_unique_ids_w = dynamic_embedding.embedding_lookup(deep_dynamic_variables,
                                                                 all_unique_ids,
                                                                 name="unique_ids_weights",
                                                                 return_trainable=False)
        embed_loss = l2_reg * tf.nn.l2_loss(
            tf.reshape(all_unique_ids_w, shape=[-1, embedding_size])) + tf.reduce_sum(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

        tf.compat.v1.summary.scalar('embed_loss', embed_loss)
        loss = loss + embed_loss

    loss = tf.identity(loss, name="total_loss")
    tf.compat.v1.summary.scalar('total_loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        logging.info("------ EVAL -------")
        eval_metric_ops = {
            "ctr_auc_eval": ctr_auc,
        }
        if has_label:
            logging.info("------ build offline label -------")
            preds["labels_ctr"] = tf.reshape(labels_ctr, shape=[-1, 1])
        export_outputs = {
            "predict_export_outputs": tf.estimator.export.PredictOutput(outputs=preds)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs)

    logging.info("---- Learning rate ----")
    lr = get_learning_rate(params["parameters"]["learning_rate"], params["parameters"]["use_decay"])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_global_step()
        logging.info("------ TRAIN -------")
        optimizer_type = params["parameters"].get('optimizer', 'Adam')
        if optimizer_type == 'Sgd':
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr)
        elif optimizer_type == 'Adagrad':
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=lr)
        elif optimizer_type == 'Rmsprop':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr)
        elif optimizer_type == 'Ftrl':
            optimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate=lr)
        elif optimizer_type == 'Momentum':
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        else:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        if params["args"].get("addon_embedding"):
            optimizer = dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=global_step)

        # fix tf2 batch_normalization bug
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        logging.info('train ops: {}, update ops: {}'.format(str(train_op), str(update_ops)))
        train_op = tf.group([train_op, update_ops])

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=preds,
                                          loss=loss,
                                          train_op=train_op)
