python3 src/bert_squad_trans.py \
    --do_lower_case \
    --adaptation_method=smooth_can \
    --num_train_epochs=2.0 \
   	--version_2_with_negative \
    --do_train \
    --beta=0.001 \
    --sigma=0.001 \
    --do_predict \
    --predict_squad \
    --mrqa_train_file=data/synthetic/generated_T5small_squadv2.json \
    --debug False \
    --predict_file data/squad/dev-v2.0.json \
   	--train_both
   	#--train_squad
   	#use mrqa+squad for training    

python3 src/evaluation_v2.py \
    --data_file data/squad/dev-v2.0.json \
    --pred_file ./experiments/train_squad_generated_t5small_squadv2_smooth_can_predict_squadv2/squad_predictions.json \
    --na-prob-file ./experiments/train_squad_generated_t5small_squadv2_smooth_can_predict_squadv2/squad_null_odds.json

echo " ****** running da using generated target domain data ****** "
cat eval_squadv2.json
echo "\n"
