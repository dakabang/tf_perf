#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np

np.set_printoptions(precision=8)

input = tf.Variable(tf.random.normal([1, 32]))
w0 = tf.Variable(tf.random.normal([32, 1]))
b0 = tf.Variable(tf.random.normal([1]))
result = tf.add(tf.matmul(input, w0), b0)
input_2 = tf.Variable(tf.random.normal([1, 32]))
input_3 = tf.concat([input, input_2], axis=0)
result_3 = tf.add(tf.matmul(input_3, w0), b0)

print (result)
print (result_3)
