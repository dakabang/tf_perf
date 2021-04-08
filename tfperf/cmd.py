#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import importlib
from multiprocessing import Process, Queue
import os
import json
import time
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

from tfperf import dataset

def run_local_or_distributed(model_name, data_and_mode, 
                             batch_size, ckpt_path, saved_model_path, 
                             feature_dict, 
                             label_dict, autogen_data, 
                             autogen_data_num, max_file_size, 
                             warm_start_from=None,
                             local_distributed=False,
                             cluster_config=None,
                             profile=False):
    # Generate data if needed
    if not isinstance(data_and_mode, list):
        raise TypeError(
            "data_and_mode must be list, given: {}".format(type(data_and_mode)))
    all_features = {**feature_dict, **label_dict}
    if autogen_data:
        gened_data = set()
        for (data_path, _) in data_and_mode:
            tfrds = dataset.generate_data(
                all_features, autogen_data_num, data_path,
                max_file_size, overwrite=True, gen_logid=True)
    else:
        for (data_path, _) in data_and_mode:
            if not os.path.exists(data_path):
                raise Exception("data_path %s not exists" % data_path)
    model = importlib.import_module(model_name)
    if not hasattr(model, "parse_tfrd_fn"):
        parse_tfrd_fn = dataset.generate_parse_tfrd_fn(feature_dict, label_dict)
    else:
        parse_tfrd_fn = model.parse_tfrd_fn
    if not hasattr(model, "model_fn"):
        raise Exception("model %s don't have model_fn function!" % model)

    def input_fn(data_path, shuffle=True, compress_type=None, interleave=False, **kwargs):
        dataset = tf.data.Dataset.list_files(os.path.join(data_path, "part-*"), shuffle=False)
        dataset = dataset.apply(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=compress_type))
        if hasattr(model, "dataset_etl"):
            dataset = model.dataset_etl(dataset)
        else:
            dataset = dataset.prefetch(batch_size).batch(batch_size)
        return dataset.map(parse_tfrd_fn, num_parallel_calls=8)

    def input_fn_shard(data_path, node_num, node_index, 
                       shuffle=True, compress_type=None, interleave=False, **kwargs):
        # Shuffle must be false when run in distributed mode
        dataset = tf.data.Dataset.list_files(os.path.join(data_path, "part-*"), shuffle=False)
        dataset = dataset.shard(int(node_num), int(node_index))
        dataset = dataset.apply(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=compress_type))
        if hasattr(model, "dataset_etl"):
            dataset = model.dataset_etl(dataset)
        else:
            dataset = dataset.prefetch(batch_size).batch(batch_size)
        return dataset.map(parse_tfrd_fn, num_parallel_calls=8)

    # Run params passed to model_fn
    run_params={
        "features": feature_dict, 
        "labels": label_dict, 
        "batch_size": batch_size,
        "ps_num": 0 if 'ps_num' not in cluster_config else cluster_config['ps_num']
    }
    def run_distributed_local(**kwargs):
        run_local(model, data_and_mode, input_fn_shard, 
                  run_params, ckpt_path, saved_model_path, 
                  warm_start_from, profile, **kwargs)
    if local_run_distributed and cluster_config:
        results = local_run_distributed(cluster_config, run_distributed_local)
        logging.info("local distributed result: %s", results)
    elif cluster_config:
        os.environ['TF_CONFIG'] = json.dumps(cluster_config)
        p = Process(
            target= \
            lambda: run_local(model, data_and_mode, input_fn_shard, \
                    run_params, ckpt_path, saved_model_path, warm_start_from, profle), args=())
        p.start()
        p.join()
    else:
        run_local(model, data_and_mode, 
                  input_fn, run_params, ckpt_path, saved_model_path, warm_start_from, profile)


def local_run_distributed(cluster_config, callback, port=7553):
    assert ('chief_num' in cluster_config)
    assert(cluster_config['chief_num'] == 1)
    assert ('worker_num' in cluster_config)
    assert ('ps_num' in cluster_config)
    all_worker_num = cluster_config['chief_num'] + cluster_config['worker_num']
    tf_cluster = dict()
    tf_cluster['chief'] = ['localhost:%d' % port]
    tf_cluster.update(
        {"worker": ['localhost:%d' % (port + i + 1) for i in range(cluster_config['worker_num'])]})
    tf_cluster.update(
        {"ps": ['localhost:%d' % (port + i + all_worker_num) for i in range(cluster_config['ps_num'])]})
    # Start chief process
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': tf_cluster,
        'task': {'type': 'chief', "index": 0},
    })
    chief_q = Queue()
    chief_process = Process(target=callback, 
                            kwargs={'node_num': str(all_worker_num), 
                                    'node_index': '0',
                                    'return_value_queue': chief_q})
    chief_process.start()
    # Start worker process
    worker_process = []
    for i in range(cluster_config['worker_num']):
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': tf_cluster,
            'task': {'type': 'worker', "index": i},
        })
        # retrieve result
        q = Queue()
        p = Process(target=callback, 
                    kwargs={'node_num': str(all_worker_num), 
                            'node_index': str(i + 1),
                            'return_value_queue': q})
        p.start()
        worker_process.append((p, q))
    # Start ps process
    ps_process = []
    for i in range(cluster_config['ps_num']):
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': tf_cluster,
            'task': {'type': 'ps', "index": i},
        })
        p = Process(target=callback, args=())
        p.start()
        ps_process.append(p)
    results = dict()
    results[chief_process.pid] = json.loads(chief_q.get())
    for p,q in worker_process:
        p.join()
        results[p.pid] = json.loads(q.get())
    # Manually stop ps servers
    for p in ps_process:
        p.terminate()
    return json.dumps(results)


class GlobalStepHook(tf.estimator.SessionRunHook):
    def __init__(self):
        self._global_step_tensor = None
        self.value = None

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_global_step()

    def after_run(self, run_context, run_values):
        self.value = run_context.session.run(self._global_step_tensor)

    def __str__(self):
        return str(self.value)


def run_local(model, data_and_mode, input_fn, run_params, 
              ckpt_path, saved_model_path, warm_start_from, profile, **kwargs):
    tf_config = json.loads(os.environ['TF_CONFIG'])
    is_chief = tf_config['task']['type'] == 'chief'
    role = tf_config['task']['type'] + '_' + str(tf_config['task']['index'])
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        save_summary_steps=10,
    )
    features = run_params['features']
    batch_size = run_params['batch_size']
    estimator = tf.estimator.Estimator(
        model_dir=ckpt_path,
        model_fn=model.model_fn,
        config=run_config,
        params=run_params,
        warm_start_from=warm_start_from)
    # Distributed mode, should use train_and_evaluate
    key_results = []
    for (data_path, mode) in data_and_mode:
        begin_time = time.time() * 1000
        data_key_result = dict()
        data_key_result['data_path'] = data_path
        data_key_result['mode'] = mode
        def data_input_fn():
            return input_fn(data_path, **kwargs)
        if profile:
            data_key_result['profile_log_path'] = 'logdir_' + role
            tf.profiler.experimental.start('logdir_' + role)
        train_global_step_hook = GlobalStepHook()
        eval_global_step_hook = GlobalStepHook()
        train_spec = tf.estimator.TrainSpec(input_fn=data_input_fn, hooks=[train_global_step_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=data_input_fn, hooks=[eval_global_step_hook])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        train_end_time = time.time() * 1000
        data_key_result['run_duration'] = str(train_end_time - begin_time)
        data_key_result['train_steps'] = str(train_global_step_hook.value)
        data_key_result['eval_steps'] = str(eval_global_step_hook.value)
        #data_key_result['train_rate'] = "%d steps/s" % \
        #        (data_key_result['train_steps'] + data_key_result['eval_steps']) / (data_key_result['run_duration'] / 1000)
        if profile:
            tf.profiler.experimental.stop()
        if is_chief and saved_model_path:
            print ("export model for serving to %s..." % saved_model_path)
            serving_input_dict = dict()
            for feakey, feainfo in features.items():
                serving_input_dict.update({feakey: (feainfo[0], [1] + feainfo[1], feainfo[2])})
            estimator.export_saved_model(
                saved_model_path,
                dataset.serving_input_fn(serving_input_dict, batch_size))
            save_model_end_time = time.time() * 1000
            data_key_result['export_path'] = saved_model_path
            data_key_result['export_duration'] = str(save_model_end_time - train_end_time)
        key_results.append(data_key_result)
    if kwargs.get('return_value_queue', None):
        kwargs['return_value_queue'].put(json.dumps(key_results))
    else:
        return json.dumps(key_results)


def fea2req(model_name, model_signature_name, features):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = model_signature_name
    for k,v in features.items():
        request.inputs[k].CopyFrom(tf.make_tensor_proto(v))
    return request


def run_press_parallel(data_path, url, model_name, model_signature_name, 
                       batch_size, feature_dict, label_dict, parallel, duration):
    press_process = []
    for i in range(parallel):
        q = Queue()
        p = Process(target=run_press_internal, 
                    args=(data_path, url, model_name, model_signature_name, 
                          batch_size, feature_dict, label_dict, duration, q, ))
        p.start()
        press_process.append((p, q))
    results = dict()
    for p,q in press_process:
        p.join()
        results[p.pid] = json.loads(q.get())
    overall_qps = 0
    overall_elapse = 0
    for pid, result in results.items():
        overall_qps += result["average_qps"]
        overall_elapse += result["elapse"]
    average_elapse = overall_elapse / parallel
    logging.info("Press test duration=%dsec average qps=%d average elapse=%dms", 
                 duration, overall_qps, average_elapse)


def run_press_internal(data_path, url, model_name, model_signature_name, 
              batch_size, feature_dict, label_dict, duration, q):
    channel = grpc.insecure_channel(url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    parse_tfrd_fn = dataset.generate_parse_tfrd_fn(feature_dict, label_dict)
    def input_fn(data_path, shuffle=True, compress_type=None, interleave=False, **kwargs):
        dataset = tf.data.Dataset.list_files(os.path.join(data_path, "part-*"), shuffle=False)
        dataset = dataset.apply(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=compress_type))
        dataset = dataset.prefetch(batch_size).batch(batch_size)
        return dataset.map(parse_tfrd_fn, num_parallel_calls=8)
    req_count = 0
    req_dura = 0
    start_time = time.time()
    press_data_iter = iter(input_fn(data_path).repeat(-1))
    while True:
        features = press_data_iter.get_next()
        # pass in feature, discard label
        req = fea2req(model_name, model_signature_name, features[0])
        call_start_time = time.time() * 1000
        result_future = stub.Predict.future(req, 5.0) # timeout 5 seconds
        resp = result_future.result().outputs
        call_end_time = time.time() * 1000
        req_dura += (call_end_time - call_start_time)
        req_count += 1
        if time.time() - start_time > duration:
            break
        if req_count % 100 == 0 and req_count != 0:
            logging.info("Press %s count %d average duration %d ms", url, req_count, req_dura / req_count)
    press_dura = time.time() - start_time
    q.put(json.dumps({
        "average_qps": req_count / press_dura,
        "duration": press_dura,
        "elapse": req_dura / req_count}))

