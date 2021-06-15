'''
This file aims to convert the raw format of DST predictions to DSTC10 format
'''
import os, sys
import json
from collections import defaultdict
from mosestokenizer import MosesTokenizer, MosesDetokenizer

data_dir = sys.argv[1]
src_res_filename = sys.argv[2]
tgt_res_filename = sys.argv[3]

detokenizer = MosesDetokenizer('en')
slots_eval = [
    "taxi-leaveAt",
    "taxi-destination",
    "taxi-departure",
    "taxi-arriveBy",
    "restaurant-book_people",
    "restaurant-book_day",
    "restaurant-book_time",
    "restaurant-food",
    "restaurant-pricerange",
    "restaurant-name",
    "restaurant-area",
    "hotel-book_people",
    "hotel-book_day",
    "hotel-book_stay",
    "hotel-name",
    "hotel-area",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-internet",
    "hotel-type",
    "attraction-type",
    "attraction-name",
    "attraction-area",
    "train-book_people",
    "train-leaveAt",
    "train-destination",
    "train-day",
    "train-arriveBy",
    "train-departure"
  ]

# read dialogues

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()  # retain local pointer to value
        return value

def load_pred(input_file):
    with open(input_file) as f:
        preds = json.load(f)
    p_dict = defaultdict(list)
    curr_dial_id = 0
    cumulative_slots = {}
    for entry in preds:
        if int(entry['guid'][1]) != curr_dial_id:
            cumulative_slots = {}
            curr_dial_id = int(entry['guid'][1])

        for slot in slots_eval:
            # detokenization
            slot_val = entry['slot_prediction_{}'.format(slot)].strip('\u00a7\u00a7')
            if slot_val != 'none':
                slot_val = detokenizer(detokenizer(slot_val.lower().split()).split()).replace(' - ', '-')
                cumulative_slots[slot] = slot_val

        assert len(p_dict[int(entry['guid'][1])]) == int(entry['guid'][2])
        p_dict[int(entry['guid'][1])].append(cumulative_slots)

    return p_dict

usr_preds = load_pred(os.path.join(data_dir, src_res_filename))

out = []
for diag_id in range(len(usr_preds)):
    usr_pred = usr_preds[diag_id]
    tmp_res = Vividict()

    for slot, val in usr_pred[-1].items():
        slot_domain, slot_type = slot.split('-')
        if len(slot_type.split('_')) == 2:
            slot_type_prefix, slot_type_content = slot_type.split('_')
        else:
            slot_type_prefix = 'semi'
            slot_type_content = slot_type
        if isinstance(tmp_res[slot_domain][slot_type_prefix][slot_type_content], dict):
            tmp_res[slot_domain][slot_type_prefix][slot_type_content] = [val]
        else:
            tmp_res[slot_domain][slot_type_prefix][slot_type_content].append(val)
    out.append(tmp_res)

json.dump(out, open(os.path.join(data_dir, tgt_res_filename), 'w'))