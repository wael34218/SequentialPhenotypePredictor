#!/bin/bash
# example running script for basicMF

selected_diags=("d_244" "d_327" "d_493" "d_250" "d_403" "d_274" "d_427" "d_428" "d_496" "d_272" "d_414" "d_530" "d_585" "d_401" "d_412" "d_584" "d_486" "d_285.9" "d_511" "d_410" "d_038" "d_458" "d_997" "d_785.5" "d_287.5" "d_518" "d_507" "d_276" "d_285.1" "d_995" "d_599" "d_774" "d_311" "d_416" "d_424" "d_305")
rounds=100

for diag in "${selected_diags[@]}"
do
    mkdir -p $diag
    echo "============ Diagnosis $diag  ============================================"
    for f in `seq 0 50 100`; do
        for r in `seq .005 .01 .095`; do
            echo "============ regularizer $r ============"
    	    for rl in `seq .005 .01 .095`; do
                echo "============ latent reg $rl ============"
    	        for i in {0..9}
    	        do
    	            ../../lib/svdfeature/tools/make_feature_buffer ../../Data/svd_balanced/mimic_train_$diag\_$i ua.base.buffer
    	            ../../lib/svdfeature/tools/make_feature_buffer ../../Data/svd_balanced/mimic_test_$diag\_$i ua.test.buffer
    
    	            ../../lib/svdfeature/svd_feature svd_predict.conf num_round=$rounds model_out_folder=$diag wd_item=$r wd_user=$r wd_global=$r wd_item_bias=$rl wd_user_bias=$rl num_factor=$f
    	            ../../lib/svdfeature/svd_feature_infer svd_predict.conf pred=$rounds model_out_folder=$diag name_pred=predictions/pred_$diag\_$r\_$rl\_$i\_$f
    	        done
            done
        done
    done
done

