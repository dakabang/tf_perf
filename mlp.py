#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

def multilayer_perception(input, hiddens):
    assert (isinstance(hiddens, list))
    weights = dict()
    bias = dict()
    n_input = input.shape[-1]
    for index, hidden in enumerate(hiddens):
        if index == 0:
            weights.update({"weight_%d" % index: tf.Variable(tf.random.normal([n_input, hidden]))})
        else:
            weights.update({"weight_%d" % index: tf.Variable(tf.random.normal([hiddens[index - 1], hidden]))})
        bias.update({"bias_%d" % index: tf.Variable(tf.random.normal([hidden]))})
    inners = []
    inners.append(input)
    prev = input
    for i in range(len(hiddens)):
        #out = tf.add(tf.matmul(prev, weights["weight_%d" % i]), bias["bias_%d" % i])
        out = tf.add(tf.matmul(prev, weights["weight_%d" % i]), bias["bias_%d" % i])
        inners.append(out)
        prev = out
    return out, inners
