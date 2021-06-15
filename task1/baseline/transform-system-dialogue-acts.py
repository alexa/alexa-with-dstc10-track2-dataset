'''
This file aims to convert the DST prediction results for the system utterances into dialogue acts
so that they can be consumed by DST model for the user utterances
'''
from mosestokenizer import MosesTokenizer, MosesDetokenizer
import sys
import json
from collections import defaultdict

in_file_path = sys.argv[1]
out_file_path = sys.argv[2]

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

def load_pred(input_file):
    with open(input_file) as f:
        preds = json.load(f)
    p_dict = {}
    curr_dial_id = 0
    cumulative_slots = {}
    for entry in preds:
        modified_slots = {}
        if int(entry['guid'][1]) != curr_dial_id:
            cumulative_slots = {}
            curr_dial_id = int(entry['guid'][1])

        for slot in slots_eval:
            # detokenization
            slot_val = entry['slot_prediction_{}'.format(slot)]
            if slot_val != 'none':
                slot_val = detokenizer(detokenizer(slot_val.lower().split()).split()).replace(' - ', '-')
                if slot in cumulative_slots and slot_val == cumulative_slots[slot]:
                    continue
                cumulative_slots[slot] = slot_val
                modified_slots[slot] = slot_val

        for slot, slot_val in modified_slots.items():
            key = (entry['guid'][1], str(int(entry['guid'][2]) + 1), slot)
            p_dict[key] = slot_val

    return p_dict


sys_inform_pd = load_pred(in_file_path)
# sys_inform_pd_docIDs = set(map(lambda x: x[0], sys_inform_pd.keys()))


## I want to write the predicted dialogue acts into file
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()  # retain local pointer to value
        return value

dialogue_acts_out = Vividict()
for key, val in sys_inform_pd.items():
    if isinstance(dialogue_acts_out[key[0]][key[1]][key[-1].split('-')[0].capitalize() + '-Inform'], dict):
        dialogue_acts_out[key[0]][key[1]][key[-1].split('-')[0].capitalize() + '-Inform'] = [[key[-1].split('-')[-1], val]]
    else:
        dialogue_acts_out[key[0]][key[1]][key[-1].split('-')[0].capitalize() + '-Inform'].append([key[-1].split('-')[-1], val])

json.dump(dialogue_acts_out, open(out_file_path, 'w'))

