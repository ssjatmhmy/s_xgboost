#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
#python mapfeat.py
# split train and test
#python mknfold.py agaricus.txt 1

cd ../../
make
cd demo/binary_classification

# training and output the models
../../xgboost mushroom.conf
# output prediction task=pred 
../../xgboost mushroom.conf task=pred model_in=0002.model

rm agaricus.txt.test.buffer agaricus.txt.train.buffer
