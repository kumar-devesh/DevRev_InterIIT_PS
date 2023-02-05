python3 src/run_source.py \
	--debug False \
	--bert_model bert-base-uncased \
	--do_train \
	--do_predict \
	--num_train_epochs 1.0 \
	--version_2_with_negatives True \
	--train_file data/squad/train-v2.0.json \
	--predict_file data/squad/dev-v2.0.json \
	--output_dir squad \
	--output_model_file squad_finetuned_model.bin \
	--logger_path squad 
	
python3 src/evaluation_v2.py \
	--data_file './data/squad/dev-v2.0.json' \
	--pred_file './squad/predictions.json' \
  	--out-file 'eval_squadv2_source.json' \
  	--na-prob-file './squad/null_odds.json'

cat eval_squadv2_source.json
