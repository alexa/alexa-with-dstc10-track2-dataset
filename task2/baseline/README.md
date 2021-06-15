[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Baselines for DSTC10 Track 2 - Task 2

In this task, we take the following two publicly available systems as the official baselines:

* [DSTC9 Track 1 Baseline](https://github.com/alexa/alexa-with-dstc9-track1-dataset/tree/master/baseline)
* [Knover (DSTC9 Track 1 Winner)](https://github.com/PaddlePaddle/Knover/blob/develop/projects/DSTC9-Track1/README.md)

If you want to publish experimental results with these baselines, please cite the following articles:
```
@inproceedings{kim-etal-2020-beyond,
    title = "Beyond Domain {API}s: Task-oriented Conversational Modeling with Unstructured Knowledge Access",
    author = "Kim, Seokhwan and Eric, Mihail and Gopalakrishnan, Karthik  and Hedayatnia, Behnam and Liu, Yang  and Hakkani-Tur, Dilek",
    booktitle = "Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    pages = "278--289"
}

@article{DBLP:journals/corr/abs-2102-02096,
  author    = {Huang He and Hua Lu and Siqi Bao and Fan Wang and Hua Wu and Zhengyu Niu and Haifeng Wang},
  title     = {Learning to Select External Knowledge with Multi-Scale Negative Sampling},
  journal   = {CoRR},
  volume    = {abs/2102.02096},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.02096},
  archivePrefix = {arXiv},
  eprint    = {2102.02096},
  timestamp = {Tue, 16 Feb 2021 16:58:52 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2102-02096.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

The remainder of this document describes how to run each baseline system on the DSTC10 validation dataset.

## [DSTC9 Track 1 Baseline](https://github.com/alexa/alexa-with-dstc9-track1-dataset/tree/master/baseline)

* Clone the DSTC9 Track 1 repository into your working directory.
``` shell
$ git clone https://github.com/alexa/alexa-with-dstc9-track1-dataset.git
$ cd alexa-with-dstc9-track1-dataset
```

* Install the required python packages.
``` shell
$ pip3 install -r requirements.txt
$ python3 -m nltk.downloader 'punkt'
$ python3 -m nltk.downloader 'wordnet'
```

* Train the baseline models.
``` shell
$ bash ./train_baseline.sh
```

* Run the baseline models.
``` shell
$ mkdir -p pred/val
$ python3 baseline.py --eval_only --checkpoint runs/ktd-baseline/ \
   --eval_dataset val \
   --dataroot [DSTC10 DATA DIRECTORY PATH] \
   --no_labels \
   --output_file pred/val/baseline.ktd.json
$ python3 baseline.py --eval_only --checkpoint runs/ks-all-baseline \
   --eval_all_snippets \
   --dataroot [DSTC10 DATA DIRECTORY PATH] \
   --eval_dataset val \
   --labels_file pred/val/baseline.ktd.json \
   --output_file pred/val/baseline.ks.json
$ python3 baseline.py --generate runs/rg-hml128-kml128-baseline \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot [DSTC10 DATA DIRECTORY PATH] \
   --labels_file pred/val/baseline.ks.json \
   --output_file [DSTC10 TRACK 2 - TASK 2 ROOT]/baseline_val.json
```

* Validate the structure and contents of the system output.
``` shell
$ cd [DSTC10 TRACK 2 - TASK 2 ROOT]
$ python3 scripts/check_results.py --dataset val --dataroot data --outfile baseline_val.json
Found no errors, output file is valid
```

* Evaluate the output.
``` shell
$ python3 scripts/scores.py --dataset val --dataroot data --outfile baseline_val.json --scorefile baseline_val.score.json
```

* Print out the scores.
``` shell
$ cat baseline_val.score.json | jq
{
  "detection": {
    "prec": 0.9850746268656716,
    "rec": 0.6346153846153846,
    "f1": 0.7719298245614035
  },
  "selection": {
    "mrr@5": 0.5405458089668618,
    "r@1": 0.4444444444444444,
    "r@5": 0.6900584795321638
  },
  "generation": {
    "bleu-1": 0.13377595884811364,
    "bleu-2": 0.053869090494959776,
    "bleu-3": 0.023546946263485788,
    "bleu-4": 0.012780881154658353,
    "meteor": 0.14571425967471133,
    "rouge_1": 0.16504300810813982,
    "rouge_2": 0.044347961278204395,
    "rouge_l": 0.11806171728465446
  }
}
```

## [Knover (DSTC9 Track 1 Winner)](https://github.com/PaddlePaddle/Knover/blob/develop/projects/DSTC9-Track1/README.md)

* Clone the DSTC9 Track 1 repository into your working directory.
``` shell
$ git clone https://github.com/PaddlePaddle/Knover.git
$ cd Knover
```

* Install the required python packages.
``` shell
$ pip3 install -e .
$ cd projects/DSTC9-Track1/
```

* Download the models with the following scripts.
``` shell
#!/bin/bash
set -e
function download_tar() {
    remote_path=$1
    local_path=$2
    if [[ ! -e $local_path ]]; then
        echo "Downloading ${local_path} ..."
        wget $remote_path
        the_tar=$(basename ${remote_path})
        the_dir=$(tar tf ${the_tar} | head -n 1)
        tar xf ${the_tar}
        rm ${the_tar}
        local_dirname=$(dirname ${local_path})
        mkdir -p ${local_dirname}
        if [[ $(realpath ${the_dir}) != $(realpath ${local_path}) ]]; then
            mv ${the_dir} ${local_path}
        fi
        echo "${local_path} has been processed."
    else
        echo "${local_path} is exist."
    fi
}
# download dataset
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/data.tar data
# download models
mkdir -p models
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SOP-32L-Context.tar models/SOP-32L-Context
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SOP-32L-Selection.tar models/SOP-32L-Selection
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SU-32L.tar models/SU-32L
```

* Copy the DSTC10 validation data
``` shell
$ mkdir dstc10_data
$ cp [DSTC10 DATA DIRECTORY PATH]/knowledge.json dstc10_data/val_knowledge.json
$ cp [DSTC10 DATA DIRECTORY PATH]/val/logs.json dstc10_data/val_logs.json
$ cp [DSTC10 DATA DIRECTORY PATH]/val/labels.json dstc10_data/val_labels.json
```

* Run the model
``` shell
set -e
export DATA_PATH="$PWD/dstc10_data"
export DATASET_TYPE="val"
export MODEL_PATH="$PWD/models"
export OUTPUT_PATH="$PWD/output"
cd [KNOVER ROOT]
bash ./projects/DSTC9-Track1/task1/infer_with_context.sh
bash ./projects/DSTC9-Track1/task2/infer.sh
bash ./projects/DSTC9-Track1/task3/infer.sh
```

* Validate the structure and contents of the system output.
``` shell
$ cd [DSTC10 TRACK 2 - TASK 2 ROOT]
$ python3 scripts/check_results.py --dataset val --dataroot data --outfile [KNOVER ROOT]/projects/DSTC9-Track1/output/task3_val.output.json
Found no errors, output file is valid
```

* Evaluate the output.
``` shell
$ python3 scripts/scores.py --dataset val --dataroot data --outfile [KNOVER ROOT]/projects/DSTC9-Track1/output/task3_val.output.json --scorefile knover.val.score.json
```

* Print out the scores.
``` shell
$ cat knover.val.score.json | jq
{
  "detection": {
    "prec": 0.9701492537313433,
    "rec": 0.625,
    "f1": 0.760233918128655
  },
  "selection": {
    "mrr@5": 0.578167641325536,
    "r@1": 0.5263157894736842,
    "r@5": 0.6666666666666667
  },
  "generation": {
    "bleu-1": 0.1301018584913103,
    "bleu-2": 0.06346276682616644,
    "bleu-3": 0.032993231472229274,
    "bleu-4": 0.011070432863026678,
    "meteor": 0.15917073781424418,
    "rouge_1": 0.15784915193130467,
    "rouge_2": 0.05362583516598427,
    "rouge_l": 0.12449623331200048
  }
}
```
