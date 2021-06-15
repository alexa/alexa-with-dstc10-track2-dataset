[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DSTC10 Track 2 - Task 2: Task-oriented Conversational Modeling with Unstructured Knowledge Access

This repository contains the data, scripts and baseline codes for [DSTC10](https://dstc10.dstc.community/) Track 2 - Task 2.

Following the [DSTC9 Track 1](https://github.com/alexa/alexa-with-dstc9-track1-dataset), this challenge task aims to support frictionless task-oriented conversations, where the dialogue flow does not break when users have requests that are out of the scope of APIs/DB but potentially are already available in external knowledge sources.
Track participants will develop dialogue systems to understand relevant domain knowledge, and generate system responses with the relevant selected knowledge.

**Organizers:** 
Seokhwan Kim, Yang Liu, Di Jin, Alexandros Papangelis, Behnam Hedayatnia, Karthik Gopalakrishnan, Dilek Hakkani-Tur

## Important Links
* [Track Proposal](https://drive.google.com/file/d/1JMK6EdD_QY2bR49wHhCaiFLPnGj-9Ztd/view)
* [Challenge Registration](https://forms.gle/Qigb3N3hGqpEgsuW8)
* [Data Formats](data/README.md)
* [Baseline Details](baseline/README.md)

## Sub-tasks

This challenge task decouples between turns that could be handled by the existing task-oriented conversational models with no extra knowledge and turns that require external knowledge resources to be answered by the dialogue system.
We focus on the turns that require knowledge access as the evaluation target in this track by the following three tasks:

| Task #1 | Knowledge-seeking Turn Detection                                                                                                      |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To decide whether to continue the existing scenario or trigger the knowledge access branch for a given utterance and dialogue history |
| Input   | Current user utterance, Dialogue context, Knowledge snippets                                                                          |
| Output  | Binary class (requires knowledge access or not)                                                                                       |

| Task #2 | Knowledge Selection                                                                                                                   |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To select proper knowledge sources from the domain knowledge-base given a dialogue state at each turn with knowledge access           |
| Input   | Current user utterance, Dialogue context, Knowledge snippets                                                                          |
| Output  | Ranked list of top-k knowledge candidates                                                                                             |

| Task #3 | Knowledge-grounded Response Generation                                                                                                |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To take a triple of input utterance, dialog context, and the selected knowledge snippets and generate a system response               |
| Input   | Current user utterance, Dialogue context, and Selected knowledge snippets                                                             |
| Output  | Generated system response                                                                                                             |

Participants will develop systems to generate the outputs for each task.
They can leverage the annotations and the ground-truth responses available in the [DSTC9 Track 1 datasets](https://github.com/alexa/alexa-with-dstc9-track1-dataset/tree/master/data) and the [DSTC10 validation dataset](data/README.md).

In the test phase, participants will be given a set of unlabeled test instances.
And they will submit **up to 5** system outputs for **all three tasks**.

**NOTE**: For someone who is interested in only one or two of the tasks, we recommend to use [baseline systems](baseline/README.md) for the remaining tasks to complete the system outputs.

## Evaluation

Each submission will be evaluated in the following task-specific automated metrics first:

| Task                                   | Automated Metrics          |
|----------------------------------------|----------------------------|
| Knowledge-seeking Turn Detection       | Precision/Recall/F-measure |
| Knowledge Selection                    | Recall@1, Recall@5, MRR@5  |
| Knowledge-grounded Response Generation | BLEU, ROUGE, METEOR        |

To consider the dependencies between the tasks, the scores for knowledge selection and knowledge-grounded response generation are weighted by knowledge-seeking turn detection performances. Please find more details from [scores.py](scripts/scores.py).

The final ranking will be based on **human evaluation results** only for selected systems according to automated evaluation scores.
It will address the following aspects: grammatical/semantical correctness, naturalness, appropriateness, informativeness and relevance to given knowledge.

## Data

In this challenge task, participants will use both [DSTC9 Track 1 datasets](https://github.com/alexa/alexa-with-dstc9-track1-dataset/tree/master/data) and [DSTC10 validation dataset](data/README.md). While the DSTC9 datasets include written conversations or manual transcription of spoken conversations, the DSTC10 validation dataset contains the n-best ASR outputs for spoken conversations only.
In this task, we use the knowledge snippets in [knowledge.json](data/knowledge.json) which is the same as DSTC9 Track 1.

In the test phase, participants will be evaluated on the results generated by their models for the unlabeled test set also with the n-best ASR outputs for spoken conversations.
The test set will be on the same domains, entities and locales as the validation set.

Data and system output format details can be found from [data/README.md](data/README.md).

## Timeline

Training data release: June 14th, 2021

Test data release: September 13th, 2021

Submission of final results (any tracks): September 21st, 2021

Final result announcement: October 1st - October 8th, 2021

* Validation data released: Jun 14, 2021
* Test data released: Sep 13, 2021
* Entry submission deadline: Sep 21, 2021
* Objective evaluation completed: Sep 28, 2021
* Human evaluation completed: Oct 8, 2021

## Rules

* Participation is welcome from any team (academic, corporate, non profit, government).
* The identity of participants will NOT be published or made public. In written results, teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the workshop chosen for communicating results.
* Participants may identify their own team label (e.g. team5), in publications or presentations, if they desire, but may not identify the identities of other teams.
* Participants are allowed to use any external datasets, resources or pre-trained models.
* All the submitted system outputs with both automatic and human evaluation results will be released to public after the evaluation period.

## Contact

### Join the DSTC mailing list to get the latest updates about DSTC10
* To join the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/join

* To post a message: send your message to list@dstc.community

* To leave the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/unsubscribe

### For specific enquiries about DSTC10 Track2

Please feel free to contact: seokhwk (at) amazon (dot) com
