#!/bin/bash
config_path="models/model/rfcn_resnet101_dota.config"
train_path="models/model/rfcn-train"
CUDA_VISIBLE_DEVICES=0,1 python train.py --logtostderr --pipeline_config_path=${config_path} --train_dir=${train_path}
