make
cd demo/binary_classification
../../xgboost mushroom.conf
../../xgboost mushroom.conf task=pred model_in=0002.model
rm agaricus.txt.test.buffer agaricus.txt.train.buffer
