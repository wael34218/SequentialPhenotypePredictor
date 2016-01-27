#!/bin/bash
# example running script for basicMF

selected_diags=("d_584" "d_428" "d_272" "d_403" "d_427")
rounds=40

for diag in "${selected_diags[@]}"
do
    mkdir -p $diag
    for i in {0..9}
    do
        # make buffer, transform text format to binary format
        ../../lib/svdfeature/tools/make_feature_buffer ../../Data/svd_balanced/mimic_train_$diag\_$i ua.base.buffer
        ../../lib/svdfeature/tools/make_feature_buffer ../../Data/svd_balanced/mimic_test_$diag\_$i ua.test.buffer

        # training for 40 rounds
        ../../lib/svdfeature/svd_feature svd_predict.conf num_round=$rounds model_out_folder=$diag
        # write out prediction from 0040.model
        ../../lib/svdfeature/svd_feature_infer svd_predict.conf pred=$rounds model_out_folder=$diag name_pred=predictions/pred_$diag\_$i
    done
done

