#!/bin/bash

# Parameters ------------------------------------------------------
DATA_DIR=../data/val
inference_model=./models/multiwoz-convbert-dg
inference_logs_filename=logs.json
inference_diag_acts_filename=dialogue_acts.json

OUT_DIR=./DSTC10_DST
mkdir -p $OUT_DIR
# First we would like to obtain predicted slots for system utterances
OUT_FILENAME=pred_res.sys.json

python run_dst.py \
--data_dir=${DATA_DIR} \
--dataset_config=dstc10-dst.json \
--inference_logs_filename ${inference_logs_filename} \
--inference_diag_acts_filename ${inference_diag_acts_filename} \
--model_type="bert" \
--model_name_or_path=${inference_model} \
--inference_model_dir=${inference_model} \
--do_lower_case \
--learning_rate=1e-4 \
--num_train_epochs=50 \
--max_seq_length=180 \
--per_gpu_train_batch_size=64 \
--per_gpu_eval_batch_size=1 \
--output_dir=${OUT_DIR} \
--output_filename=${OUT_FILENAME} \
--save_epochs=20 \
--logging_steps=10 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-6 \
--label_value_repetitions \
--swap_utterances \
--append_history \
--use_history_labels \
--delexicalize_sys_utts \
--class_aux_feats_inform \
--class_aux_feats_ds \
--for_sys_utt \
--seed 42 \
--do_infer \
--predict_type=infer \
  2>&1 | tee ${OUT_DIR}/sys.log

# Then we convert the predicted slots for system utterances into dialogue acts
python ./transform-system-dialogue-acts.py ${OUT_DIR}/${OUT_FILENAME} ${DATA_DIR}/${inference_diag_acts_filename}

# Now we predict slots for the user utterances
OUT_FILENAME=pred_res.usr.json

python run_dst.py \
--data_dir=${DATA_DIR} \
--dataset_config=dstc10-dst.json \
--inference_logs_filename ${inference_logs_filename} \
--inference_diag_acts_filename ${inference_diag_acts_filename} \
--model_type="bert" \
--model_name_or_path=${inference_model} \
--inference_model_dir=${inference_model} \
--do_lower_case \
--learning_rate=1e-4 \
--num_train_epochs=50 \
--max_seq_length=180 \
--per_gpu_train_batch_size=64 \
--per_gpu_eval_batch_size=1 \
--output_dir=${OUT_DIR} \
--output_filename=${OUT_FILENAME} \
--save_epochs=20 \
--logging_steps=10 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-6 \
--label_value_repetitions \
--swap_utterances \
--append_history \
--use_history_labels \
--delexicalize_sys_utts \
--class_aux_feats_inform \
--class_aux_feats_ds \
--seed 42 \
--do_infer \
--predict_type=infer \
  2>&1 | tee ${OUT_DIR}/usr.log

# Finally we transform the predicted slots into the output format we want
FINAL_RES_FILENAME=DST_preds.json
python dstc10_dst_finalize.py ${OUT_DIR} ${OUT_FILENAME} ${FINAL_RES_FILENAME}