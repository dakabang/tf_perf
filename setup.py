#!/usr/bin/env python
# coding=utf-8
from setuptools import setup

setup(name='tfperf',
  version='1.0',
  description='Tensorflow Test Kit',
  author='ethan',
  email='ethan01.zhan@vipshop.com',
  packages=['tfperf'],
  install_requires = [
      "tensorflow==2.4.1"
  ],
)
