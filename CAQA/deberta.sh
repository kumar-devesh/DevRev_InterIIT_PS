CUDA_VISIBLE_DEVICES=0 python3 src/bert_squad_trans.py \
    --bert_model microsoft/deberta-v3-base \
    --train_batch_size 8 \
    --do_lower_case \
    --adaptation_method=smooth_can \
    --num_train_epochs=2.0 \
    --train_squad \
   	--version_2_with_negative \
    --do_train \
    --beta=0.001 \
    --sigma=0.001 \
    --do_predict \
    --predict_squad \
    --mrqa_train_file=generated_T5base_squadv2.json \
    --debug True \
    --predict_file data/squad/dev-v2.0.json
    #--train_both \ #use mrqa+squad for training    

#python3 src/evaluation_v2.py --data_file data/squad/dev-v2.0.json
#echo "running da without generated target domain data"
#cat eval_squadv2.json
#echo "\n"