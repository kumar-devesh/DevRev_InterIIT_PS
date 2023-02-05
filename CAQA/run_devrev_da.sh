#train using mrqa+squad
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 src/bert_squad_trans.py \
    --bert_model devpranjal/deberta-v3-base-devrev-data \
    --do_lower_case \
    --train_batch_size 8 \
    --adaptation_method=smooth_can \
    --num_train_epochs=2.0 \
    --version_2_with_negative \
    --do_train \
    --beta=0.001 \
    --sigma=0.001 \
    --predict_squad \
    --mrqa_train_file data/synthetic/finetune_data_devrev.json \
    --debug False \
    --predict_file data/squad/dev-v2.0.json \
    --train_both \
    --learning_rate 0.01 \
    --warmup_proportion 0.2 \
