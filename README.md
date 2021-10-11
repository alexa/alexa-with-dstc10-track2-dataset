[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DSTC10 Track 2 - Knowledge-grounded Task-oriented Dialogue Modeling on Spoken Conversations

This repository contains the data, scripts and baseline codes for [DSTC10](https://dstc10.dstc.community/) Track 2.

This challenge track aims to benchmark the robustness of the conversational models against the gaps between written and spoken conversations.
Specifically, it includes two target tasks: 1) [multi-domain dialogue state tracking](task1/README.md) and 2) [task-oriented conversational modeling with unstructured knowledge access](task2/README.md).
For both tasks, participants will develop models using any existing public data and submit the model outputs on the unlabeled test data set with the ASR outputs.

**Organizers:** 
Seokhwan Kim, Yang Liu, Di Jin, Alexandros Papangelis, Behnam Hedayatnia, Karthik Gopalakrishnan, Dilek Hakkani-Tur

## News
* **October 11, 2021** - The ground-truth labels/responses of the test data are released at [Task 1 Labels](task1/data/test/labels.json), [Task 2 Labels](task2/data/test/labels.json)  
* **October 11, 2021** - All the submitted entries by the participants are released at [Task 1 Submissions](task1/results), [Task 2 Submissions](task2/results)
* **October 8, 2021** - The human evaluation scores for Task 2 are now available: [Task 2 Results](https://docs.google.com/spreadsheets/d/19XIj4m9z7_-uSBP8slTEm113VLrWCL5vXwKCL8EFs2U/edit?usp=sharing)
* **September 27, 2021** - The objective evaluation results are now available: [Task 1 Results](https://docs.google.com/spreadsheets/d/1SyOGA_WbfWmcSExFzrmVqKpu5lJS-4qLrHTgzHAIziU/edit?usp=sharing), [Task 2 Results](https://docs.google.com/spreadsheets/d/19XIj4m9z7_-uSBP8slTEm113VLrWCL5vXwKCL8EFs2U/edit?usp=sharing)
* **September 13, 2021** - The evaluation data is released for [Task 1](task1/data/test/logs.json) and [Task 2](task2/data/test/logs.json). Please find the participation details for [Task 1](task1/README.md#participation) and [Task 2](task2/README.md#participation).
* **August 25, 2021** - [Frequently Asked Questions](FAQ.md) added.
* **August 12, 2021** - Patched data/code released for task 1. Please find the details from [release notes](https://github.com/alexa/alexa-with-dstc10-track2-dataset/blob/main/release-notes.md#2021-08-10) and  **update your local branch**.


## Important Links
* [Track Proposal](https://drive.google.com/file/d/1JMK6EdD_QY2bR49wHhCaiFLPnGj-9Ztd/view)
* [Challenge Registration](https://forms.gle/Qigb3N3hGqpEgsuW8)
* [Task 1 Details](task1/README.md)
* [Task 2 Details](task2/README.md)
* Objective Evaluation Results
  * [Task 1 Results](https://docs.google.com/spreadsheets/d/1SyOGA_WbfWmcSExFzrmVqKpu5lJS-4qLrHTgzHAIziU/edit?usp=sharing)
  * [Task 2 Results](https://docs.google.com/spreadsheets/d/19XIj4m9z7_-uSBP8slTEm113VLrWCL5vXwKCL8EFs2U/edit?usp=sharing)
* Human Evaluation Results
  * [Task 2 Results](https://docs.google.com/spreadsheets/d/19XIj4m9z7_-uSBP8slTEm113VLrWCL5vXwKCL8EFs2U/edit?usp=sharing)
* Submitted Entries
  * [Task 1 Submissions](task1/results)
  * [Task 2 Submissions](task2/results) (including the human evaluation scores for each finalist)
* Ground-truth Labels
  * [Task 1 Labels](task1/data/test/labels.json)
  * [Task 2 Labels](task2/data/test/labels.json)  

If you want to publish experimental results with this dataset or use the baseline models, please cite [this article](https://arxiv.org/abs/2109.13489):
```
@misc{kim2021how,
      title={"How robust r u?": Evaluating Task-Oriented Dialogue Systems on Spoken Conversations}, 
      author={Seokhwan Kim and Yang Liu and Di Jin and Alexandros Papangelis and Karthik Gopalakrishnan and Behnam Hedayatnia and Dilek Hakkani-Tur},
      year={2021},
      eprint={2109.13489},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Timeline
* Validation data released: Jun 14, 2021
* Test data released: Sep 13, 2021
* Entry submission deadline: Sep 21, 2021
* Objective evaluation completed: Sep 28, 2021
* Human evaluation completed: Oct 8, 2021

## Rules
* Participation is welcome from any team (academic, corporate, non profit, government).
* Each team can participate in either or both sub-tracks by submitting up to 5 entries for each track.
* The identity of participants will NOT be published or made public. In written results, teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the workshop chosen for communicating results.
* Participants may identify their own team label (e.g. team5), in publications or presentations, if they desire, but may not identify the identities of other teams.
* Participants are allowed to use any external datasets, resources or pre-trained models which are publicly available.
* Participants are NOT allowed to do any manual examination or modification of the test data.
* All the submitted system outputs with the evaluation results will be released to public after the evaluation period.

## Contact

### Join the DSTC mailing list to get the latest updates about DSTC10
* To join the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/join

* To post a message: send your message to list@dstc.community

* To leave the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/unsubscribe

### For specific enquiries about DSTC10 Track2

Please feel free to contact: seokhwk (at) amazon (dot) com
