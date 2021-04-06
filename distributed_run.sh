#!/bin/bash
# export TF_HASHTABLE_INIT_SIZE=100000
# python run.py --model_name=models.dnn_model --mode=train_eval --data_path=data.tfrd --ckpt_path=ckpt --batch_size=10 --node_num=3
python run.py --model_name=models.estimator_2class --mode=train_eval --data_path=data.tfrd --ckpt_path=ckpt --batch_size=10 --node_num=2
