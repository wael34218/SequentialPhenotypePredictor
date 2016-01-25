#!/bin/bash
# example running script for basicMF

# make buffer, transform text format to binary format
../../lib/svdfeature/tools/make_feature_buffer ../../Data/svd/mimic_train_d_584_0 ua.base.buffer
../../lib/svdfeature/tools/make_feature_buffer ../../Data/svd/mimic_test_d_584_0 ua.test.buffer


# training for 40 rounds
../../lib/svdfeature/svd_feature svd_predict.conf num_round=10 model_out_folder='d_584'
# write out prediction from 0040.model
../../lib/svdfeature/svd_feature_infer svd_predict.conf pred=10 model_out_folder='d_584'
