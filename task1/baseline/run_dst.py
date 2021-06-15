# coding=utf-8
#
# Original Copyright 2020 Heinrich Heine University Duesseldorf. Licensed under the Apache License, Version 2.0 (the "License").
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import logging
import os
import random
import json
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from trippy.modeling_bert_dst import (BertForDST)
from data_processors import DSTC10Processor
from utils import convert_examples_to_features_inference
from trippy.tensorlistdataset import (TensorListDataset)

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForDST, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        
def to_list(tensor):
    return tensor.detach().cpu().tolist()


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def inference(args, model, tokenizer, processor, prefix=""):
    dataset, features = load_and_cache_examples_inference(args, model, tokenizer, processor, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset) # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in model.slot_list}
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = batch_to_device(batch, args.device)

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[-1]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            inputs = {'input_ids':       batch[0],
                      'input_mask':      batch[1],
                      'segment_ids':     batch[2],
                      'start_pos':       None,
                      'end_pos':         None,
                      'inform_slot_id':  batch[3],
                      'refer_id':        None,
                      'diag_state':      diag_state,
                      'class_label_id':  None}
            unique_ids = [features[i.item()].guid for i in batch[-1]]
            input_ids_unmasked = [features[i.item()].input_ids_unmasked for i in batch[-1]]
            inform = [features[i.item()].inform for i in batch[-1]]
            outputs = model(**inputs)

            # Update dialog state for next turn.
            for slot in model.slot_list:
                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        preds, ds = predict_and_format_inference(model, tokenizer, outputs[2], outputs[3], outputs[4], outputs[5],
                                                      unique_ids, input_ids_unmasked, inform, prefix, ds)
        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist] # Flatten list

    # Write final predictions (for evaluation with external tool)
    output_prediction_file = os.path.join(args.output_dir, args.output_filename)
    with open(output_prediction_file, "w") as f:
        json.dump(all_preds, f, indent=2)


def predict_and_format_inference(model, tokenizer, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                       per_slot_refer_logits, ids, input_ids_unmasked, inform, prefix, ds):
    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: 'none' for slot in model.slot_list}

        prediction = {}
        prediction_addendum = {}
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            prediction['guid'] = ids[i].split("-")
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['refer_prediction_%s' % slot] = refer_prediction

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])
                dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = '§§' + inform[i][slot]
            # Referral case is handled below

            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in model.class_types and class_prediction == model.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                dialog_state[slot] = dialog_state[model.slot_list[refer_prediction - 1]]
                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]  # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)

    return prediction_list, dialog_state


def load_and_cache_examples_inference(args, model, tokenizer, processor, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_file = os.path.join(args.output_dir, 'cached_{}_features'.format(
        'sys' if args.for_sys_utt else 'usr'))
    if os.path.exists(cached_file) and not args.overwrite_cache: # and not output_examples:
        logger.info("Loading features from cached file %s", cached_file)
        features = torch.load(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        processor_args = {'append_history': args.append_history,
                          'use_history_labels': args.use_history_labels,
                          'swap_utterances': args.swap_utterances,
                          'label_value_repetitions': args.label_value_repetitions,
                          'delexicalize_sys_utts': args.delexicalize_sys_utts,
                          'for_sys_utt': args.for_sys_utt}
        examples = processor.get_test_examples(os.path.join(args.data_dir, args.inference_logs_filename),
                                               os.path.join(args.data_dir, args.inference_diag_acts_filename),
                                               processor_args)
        features = convert_examples_to_features_inference(examples=examples,
                                                slot_list=model.slot_list,
                                                class_types=model.class_types,
                                                model_type=args.model_type,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                slot_value_dropout=(0.0 if evaluate else args.svd))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_file)
            torch.save(features, cached_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    f_inform_slot_ids = [f.inform_slot for f in features]
    all_inform_slot_ids = {}
    for s in model.slot_list:
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_inform_slot_ids,
                                all_example_index)

    return dataset, features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    parser.add_argument("--dataset_config", default=None, type=str, required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--predict_type", default=None, type=str, required=True,
                        help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--output_filename", default=None, type=str, required=True,
                        help="The output filename that predictions will be written into.")
    parser.add_argument("--inference_model_dir", default=None, type=str,
                        help="In inference, output_dir is used to specify whether to save predictions, which can "
                             "be different from model dir to resume from.")
    parser.add_argument("--inference_logs_filename", default=None, type=str,
                        help="In inference, filename of the dialogue logs.")
    parser.add_argument("--inference_diag_acts_filename", default=None, type=str,
                        help="In inference, filename of the inferred dialogue acts for system utts.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--do_infer", action='store_true',
                        help="Whether to run inference on the <predict_type> set without evaluation.")
    parser.add_argument("--for_sys_utt", action='store_true',
                        help="Whether this is for slot tagging for system utterances.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", default=0.3, type=float,
                        help="Dropout rate for BERT representations.")
    parser.add_argument("--heads_dropout", default=0.0, type=float,
                        help="Dropout rate for classification heads.")
    parser.add_argument("--class_loss_ratio", default=0.8, type=float,
                        help="The ratio applied on class loss in total loss calculation. "
                             "Should be a value in [0.0, 1.0]. "
                             "The ratio applied on token loss is (1-class_loss_ratio)/2. "
                             "The ratio applied on refer loss is (1-class_loss_ratio)/2.")
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")

    parser.add_argument("--append_history", action='store_true',
                        help="Whether or not to append the dialog history to each turn.")
    parser.add_argument("--use_history_labels", action='store_true',
                        help="Whether or not to label the history as well.")
    parser.add_argument("--swap_utterances", action='store_true',
                        help="Whether or not to swap the turn utterances (default: sys|usr, swapped: usr|sys).")
    parser.add_argument("--label_value_repetitions", action='store_true',
                        help="Whether or not to label values that have been mentioned before.")
    parser.add_argument("--delexicalize_sys_utts", action='store_true',
                        help="Whether or not to delexicalize the system utterances.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument("--svd", default=0.0, type=float,
                        help="Slot value dropout ratio (default: 0.0)")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps. Overwritten by --save_epochs.")
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    assert(args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0)
    assert(args.svd >= 0.0 and args.svd <= 1.0)
    assert(args.class_aux_feats_ds is False or args.per_gpu_eval_batch_size == 1)
    assert(not args.class_aux_feats_inform or args.per_gpu_eval_batch_size == 1)
    assert(not args.class_aux_feats_ds or args.per_gpu_eval_batch_size == 1)

    processor = DSTC10Processor(args.dataset_config)
    dst_slot_list = processor.slot_list
    dst_class_types = processor.class_types
    dst_class_labels = len(dst_class_types)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # Add DST specific parameters to config
    config.dst_dropout_rate = args.dropout_rate
    config.dst_heads_dropout_rate = args.heads_dropout
    config.dst_class_loss_ratio = args.class_loss_ratio
    config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
    config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
    config.dst_class_aux_feats_inform = args.class_aux_feats_inform
    config.dst_class_aux_feats_ds = args.class_aux_feats_ds
    config.dst_slot_list = dst_slot_list
    config.dst_class_types = dst_class_types
    config.dst_class_labels = dst_class_labels

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    logger.info("Updated model config: %s" % config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Inference without checking performance
    if args.do_infer and args.local_rank in [-1, 0]:
        model = model_class.from_pretrained(args.inference_model_dir)
        model.to(args.device)

        # Evaluate
        inference(args, model, tokenizer, processor, prefix='final')


if __name__ == "__main__":
    main()
