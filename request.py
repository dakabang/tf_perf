#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import data
import argparse

model_name = "tf_perf"
model_signature_name = "outputs"

def fea2req(features):
    print (features)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = model_signature_name
    for k,v in features.items():
        request.inputs[k].CopyFrom(tf.make_tensor_proto(v))
    return request


def run(filename, url, batch_size=1, is_shuffle=False):
    channel = grpc.insecure_channel(url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    items = [
        ("value_fea", tf.float32, [1000], 1000),
        ("id_fea", tf.int64, [100], 100),
        ("label", tf.float32, [], 1)
    ]
    parse_input_fn = data.gen_parse_input_fn(items)
    tfrd_data = tf.data.TFRecordDataset([filename])
    tfrd_data = tfrd_data.map(parse_input_fn)
    if batch_size > 1:
        tfrd_data = tfrd_data.batch(batch_size)
    for features in tfrd_data:
        # pass in feature, discard label
        req = fea2req(features[0])
        result_future = stub.Predict.future(req, 5.0)
        resp = result_future.result().outputs
        for k,v in resp.items():
            print ("k %s v %s" % (k, v))
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='tfrd data path')
    parser.add_argument('--url_tf', type=str, help='tf-serving url')
    parser.add_argument('--batch_size', type=int, help='batch size')
    args = parser.parse_args()
    run(args.data_path, args.url_tf, args.batch_size)
