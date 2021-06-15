# DSTC10 Track 2 - Task 1 Dataset

This directory contains the official validation dataset for [DSTC10 Track 2 - Task 1](../README.md).
The evaluation dataset will be available later.

## Data

We are releasing the validation data with the following files:

* Validation set:
  * [logs.json](val/logs.json): validation instances
  * [labels.json](val/labels.json): ground-truths for validation instances

Participants will develop systems to take *logs.json* as an input and generates the outputs following the **same format** as *labels.json*.

## JSON Data Formats

### Log Objects

The *logs.json* file includes the list of instances each of which is a partial conversation from the beginning to the target user turn.
Each instance is a list of the following turn objects:

* speaker: the speaker of the turn (string: "U" for user turn/"S" for system turn)
* text: utterance text (string)
* nbest 
  [
    * hyp: ASR hypothesis (string)
    * score: language model score (float)
  ]

### Label Objects

The *labels.json* file provides the ground-truth DST label of each instance in *logs.json* in the same order to each other.
The label objects have the following format which is compatible with [MultiWOZ 2.x](https://github.com/budzianowski/multiwoz):

* domain\_name (string: hotel/restaurant/attraction)
  * slot\_type (string: semi/book)
    * slot_name (string)
    [
      * value (string)
    ]
    
