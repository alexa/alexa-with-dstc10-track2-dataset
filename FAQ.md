# DSTC10 Track 2 - Frequently Asked Questions

## Participation

Q: Do I need to complete any sign-up or registration process to participate this challenge track?

A: We strongly recommend you to complete the [challenge registration](https://forms.gle/Qigb3N3hGqpEgsuW8) by Sep 13. You will be asked to provide more details about your submission in the evaluation phase.

---

Q: I finished my registration, but haven't been contacted by the track organizers yet. What should I do to participate in the track?

A: You don't need any confirmation from us. You can start visiting [our repository](https://github.com/alexa/alexa-with-dstc10-track2-dataset). All the data, baseline and evaluation scripts are publicly available there.

---

Q: Do I have to participate in both tasks or can I participate in either of them?

A: You can participate in either or both sub-tasks.


## Task 1: Multi-domain Dialogue State Tracking

Q: Will the DB contents and DST schema remain consistent between the validation and test sets?

A: Yes, both validation and test data sets are collected with the same [db.json](task1/data/db.json) and [output_schema.json](task1/data/output_schema.json).

---

Q: The evaluation script outputs multiple scores in different metrics. Which metric will be used for the final evaluation?

A: Joint Goal Accuracy (JGA) will be used as the final evaluation metric for the official ranking. In the case of ties, we will also look at the other metrics. The detailed tie-breaking criteria will be announced later.

---

Q: Will there be any human evaluation conducted for this task?

A: No, only the automatic evaluation by [scores.py](task1/scripts/scores.py) will be done for each submitted entry.


## Task 2: Task-oriented Conversational Modeling with Unstructured Knowledge Access

Q: Will the knowledge snippets remain consistent between the validation and test sets?

A: Yes, both validation and test data sets are collected with the same [knowledge.json](task2/data/knowledge.json).

---

Q: Will the domain API/DBs remain consistent for the validation and test sets?

A: Yes, both validation and test data sets are collected with the same [db.json](task1/data/db.json).

---

Q: Should I work on all three subtasks or can I participate in only one or two of them?

A: You will be asked to submit the full outputs across all three subtasks in the pipeline. If you're interested in just one or two subtasks, we recommend you use the baseline model for the other tasks, so that you can make valid submissions.

---

Q: What is the final criterion of team ranking? Will we have rankings for each subtask or only one ranking as a final for all subtasks?

A: The official ranking will be based on human evaluation for the end-to-end performances. The human evaluation will be done only for selected systems according to automated evaluation scores.

---

Q: Will all the submissions be included in the final ranking by human evaluation?

A: No, the human evaluation will be done only for selected systems according to automated evaluation scores. The detailed selection criteria will be announced later with the automated evaluation results.


## Rules

Q: Will I be allowed to manually annotate or preprocess the evaluation data?

A: No, any manual task involved in processing the evaluation resources will be considered as a violation of the challenge policy. Please DO NOT touch the files in `data/test` during the evaluation phase.

---

Q: Will I be allowed to manually examine my system outputs on the evaluation data to select the best model/configuration?

A: No, any manual examination on the evaluation data will be also considered as a violation of the challenge policy. Please freeze your models/systems by the end of the development phase only with the validation dataset and just submit the system outputs on the unseen evaluation data.

---

Q: Can I fine-tune or re-train my models on the unlabeled evaluation data in an unsupervised way?

A: No, you're not allowed to fine-tune or re-train your systems based on any evaluation data. Please freeze your models/systems only with the validation dataset and just submit the system outputs on the unseen evaluation data.

---

Q: Can I fine-tune or re-train my models with the [knowledge.json](task2/data/knowledge.json) or [db.json](task1/data/db.json)?

A: Yes, you are allowed to use any data/resources released in the development period (such as the validation data, [knowledge.json](task2/data/knowledge.json) or [db.json](task1/data/db.json)).

