# Original Copyright 2020 Heinrich Heine University Duesseldorf. Licensed under the Apache License, Version 2.0 (the "License").
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import json
from trippy.utils_dst import (DSTExample)
from trippy.dataset_multiwoz21 import load_acts, delex_utt, tokenize, normalize_label

def create_examples_inference(input_file, acts_file, set_type, slot_list,
                    label_maps={},
                    append_history=False,
                    use_history_labels=False,
                    swap_utterances=False,
                    label_value_repetitions=False,
                    delexicalize_sys_utts=False,
                    analyze=False,
                    for_sys_utt=False):
    """Read a DST json file into a list of DSTExample."""
    if not for_sys_utt:
        sys_inform_dict = load_acts(acts_file)
        # load_acts() automatically adds .json to the dialog_id
        # We remove it here
        sys_inform_dict = {
            (key[0].replace(".json", ""), key[1], key[2]): value
            for key, value in sys_inform_dict.items()
        }
    else:
        sys_inform_dict = {}

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    global LABEL_MAPS
    LABEL_MAPS = label_maps

    examples = []
    for dialog_id, utterances in enumerate(input_data):

        # First system utterance is empty, since multiwoz starts with user input
        utt_tok_list = [[]]

        # Collect all utterances and their metadata
        usr_sys_switch = True
        turn_itr = 0
        for utt in utterances:
            # Assert that system and user utterances alternate
            is_sys_utt = utt['speaker'] == 'S'

            if usr_sys_switch == is_sys_utt:
                print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            if is_sys_utt:
                turn_itr += 1

            # Delexicalize sys utterance
            if delexicalize_sys_utts and is_sys_utt:
                inform_dict = {slot: 'none' for slot in slot_list}
                for slot in slot_list:
                    if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                        inform_dict[slot] = sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]
                utt_tok_list.append(delex_utt(utt['text'], inform_dict))  # normalize utterances
            else:
                utt_tok_list.append(tokenize(utt['text']))  # normalize utterances

        # Form proper (usr, sys) turns
        turn_itr = 0
        sys_utt_tok = []
        usr_utt_tok = []
        hst_utt_tok = []
        start_idx = 2 if for_sys_utt else 1
        end_idx = len(utt_tok_list)
        for i in range(start_idx, end_idx, 2):
            inform_dict = {}
            inform_slot_dict = {}

            # Collect turn data
            if append_history:
                if swap_utterances:
                    hst_utt_tok = usr_utt_tok + sys_utt_tok + hst_utt_tok
                else:
                    hst_utt_tok = sys_utt_tok + usr_utt_tok + hst_utt_tok
            sys_utt_tok = utt_tok_list[i - 1]
            usr_utt_tok = utt_tok_list[i]

            guid = '%s-%s-%s' % (set_type, str(dialog_id), str(turn_itr))

            if analyze:
                print("%15s %2s %s ||| %s" % (dialog_id, turn_itr, ' '.join(sys_utt_tok), ' '.join(usr_utt_tok)))
                print("%15s %2s [" % (dialog_id, turn_itr), end='')

            for slot in slot_list:

                # Get dialog act annotations
                inform_label = list(['none'])
                inform_slot_dict[slot] = 0
                if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                    inform_label = list([normalize_label(slot, i) for i in sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]])
                    inform_slot_dict[slot] = 1
                elif (str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1]) in sys_inform_dict:
                    inform_label = list([normalize_label(slot, i) for i in sys_inform_dict[(str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1])]])
                    inform_slot_dict[slot] = 1

                inform_dict[slot] = inform_label[0]

            if analyze:
                print("]")

            if swap_utterances:
                txt_a = usr_utt_tok
                txt_b = sys_utt_tok
            else:
                txt_a = sys_utt_tok
                txt_b = usr_utt_tok
            examples.append(DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                history=hst_utt_tok,
                inform_label=inform_dict,
                inform_slot_label=inform_slot_dict
            ))

            turn_itr += 1

        if analyze:
            print("----------------------------------------------------------------------")

    return examples


class DataProcessor(object):
    def __init__(self, dataset_config):
        with open(dataset_config, "r", encoding='utf-8') as f:
            raw_config = json.load(f)
        self.class_types = raw_config['class_types']
        self.slot_list = raw_config['slots']
        self.label_maps = raw_config['label_maps']

    def get_train_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_test_examples(self, data_dir, **args):
        raise NotImplementedError()


class DSTC10Processor(DataProcessor):
    def get_test_examples(self, logs_path, diag_acts_path, args):
        return create_examples_inference(logs_path,
                                        diag_acts_path,
                                        'test', self.slot_list, self.label_maps, **args)
