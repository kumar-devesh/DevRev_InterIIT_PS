python3 src/run_qada.py \
	--debug False \
	--do_predict \
	--version_2_with_negative True \
	--bert_model bert-base-uncased \
	--num_train_epochs 4.0 \
	--train_batch_size 12 \
	--output_dir squad2target \
	--output_model_file squad_adapted_model.bin \
	--input_dir squad \
    --input_model_file squad_finetuned_model.bin.final \
	--output_prediction \
	--logger_path squad2target \
	--dirichlet_ratio 0.0 \
	--cutoff_ratio 0.0 \
	--lambda_c 0.0 \
	--do_adaptation 

python3 src/evaluation_v2.py \
	--data_file './data/squad/dev-v2.0.json' \
	--pred_file './squad2target/predictions.json' \
  	--out-file 'eval_squadv2_adaptation.json' \
  	--na-prob-file './squad2target/null_odds.json'

cat eval_squadv2_adaptation.json