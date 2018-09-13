#! /usr/bin/env bash

nohup python train_nets.py --net_depth 50 --eval_db_path ../../dataset/faces_ms1m_112x112 --tfrecords_file_path tfrecords/  --pretrained_model output/ckpt/InsightFace_iter_1200000.ckpt > train.out &
