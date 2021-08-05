[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Baselines for DSTC10 Track 2 - Task 1

In this task, we take [TripPy](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/tree/master) model trained on the [MultiWOZ 2.1 dataset](https://github.com/budzianowski/multiwoz) as the official baseline.

If you want to publish experimental results with this baselines, please cite the following article:
```
@inproceedings{heck2020trippy,
    title = "{T}rip{P}y: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking",
    author = "Heck, Michael and van Niekerk, Carel and Lubis, Nurul and Geishauser, Christian and
              Lin, Hsien-Chin and Moresi, Marco and Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 21st Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    pages = "35--44",
}
```

The remainder of this document describes how to run the baseline model on the DSTC10 validation dataset.

## How to run it

* Train the TripPy model on the [MultiWOZ 2.1 dataset](https://github.com/budzianowski/multiwoz) following the [instructions](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/tree/master).

* Copy the models into `models` directory.

* Install the required python packages.
```
$ pip3 install -r requirements.txt
```

* Run the model on the data (change the `DATA_DIR` parameter in `dstc10-dst.infer.pipeline.sh`, if required).
```
$ bash ./dstc10-dst.infer.pipeline.sh
```

* Validate the structure and contents of the tracker output.
``` shell
$ cd [DSTC10 TRACK 2 - TASK 1 ROOT]
$ python3 scripts/check_results.py --dataset val --dataroot data --outfile baseline/DSTC10_DST/DST_preds.json 
Found no errors, output file is valid
```

* Evaluate the output.
``` shell
$ python3 scripts/scores.py --dataset val --dataroot data --outfile baseline/DSTC10_DST/DST_preds.json --scorefile baseline.val.score.json --K 5
```

* Print out the scores.
``` shell
$ cat baseline.val.score.json | jq
{
  "joint_goal_accuracy": 0.004273504273504274,
  "slot": {
    "accuracy": 0.7207799145299145,
    "value_prediction": {
      "prec": 0.5977115326708823,
      "rec": 0.3378148400272294,
      "f1": 0.43166249864086115
    },
    "none_prediction": {
      "prec": 0.7473212546269238,
      "rec": 0.9788211278387343,
      "f1": 0.8475475033141847
    }
  }
}
```
