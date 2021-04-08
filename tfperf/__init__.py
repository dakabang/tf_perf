#!/usr/bin/env python
# coding=utf-8

from tfperf.cmd import run_local_or_distributed
from tfperf.cmd import run_press_parallel

# Train mode, disable eager mode
def run(**kwargs):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    run_local_or_distributed(**kwargs)

def run_press(**kwargs):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    run_press_parallel(**kwargs)
