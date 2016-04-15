make
cd demo/binary_classification
../../xgboost mushroom.conf
rm agaricus.txt.test.buffer agaricus.txt.train.buffer
