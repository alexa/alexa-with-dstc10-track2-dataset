# DSTC10 Track 2 - Task 2 Dataset

This directory contains the official validation dataset for [DSTC10 Track 2 - Task 2](../README.md).
The evaluation dataset will be available later.

## Data

We are releasing the validation data with the following files:

* Validation set:
  * [logs.json](val/logs.json): validation instances
  * [labels.json](val/labels.json): ground-truths for validation instances
* Test set:
  * [logs.json](test/logs.json): test instances

The ground-truth labels for Knowledge Selection task refer to the knowledge snippets in [knowledge.json](knowledge.json).

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

The *labels.json* file provides the ground-truth labels and human responses for the final turn of each instance in *logs.json*.
It includes the list of the following objects in the same order as the input instances:

* target: whether the turn is knowledge-seeking or not (boolean: true/false)
* knowledge: [
  * domain: the domain identifier referring to a relevant knowledge snippet in *knowledge.json* (string)
  * entity\_id: the entity identifier referring to a relevant knowledge snippet in *knowledge.json* (integer for entity-specific knowledge or string "*" for domain-wide knowledge)
  * doc\_id: the document identifier referring to a relevant knowledge snippet in *knowledge.json* (integer)
  ]
* response: knowledge-grounded system response (string)

NOTE: *knowledge* and *response* exist only for the target instances with *true* for the *target* value.

### Knowledge Objects

The *knowledge.json* contains the unstructured knowledge sources to be selected and grounded in the tasks.
It includes the domain-wide or entity-specific knowledge snippets in the following format:

* domain\_id: domain identifier (string: "hotel", "restaurant", "train", "taxi", etc.)
  * entity\_id: entity identifier (integer or string: "*" for domain-wide knowledge)
      * name: entity name (string; only exists for entity-specific knowledge)
      * docs
          * doc\_id: document identifier (integer)
            * title: document title (string)
            * body: document body (string)
