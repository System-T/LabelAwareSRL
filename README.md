# Label-Aware Semantic Role Labeling (SRL)

This repository contains resources for the paper ["Label Definitions Improve Semantic Role Labeling"](https://aclanthology.org/2022.naacl-main.411.pdf) appearing in NAACL 2022 . The resources are going through and internal review process and will be visible soon. We thank you for your patience. 


## Requirements
- Python 3.6.13
- PyTorch 1.2.0
- HuggingFace Transformers 4.6.1
- HuggingFace Datasets 1.8.0
- ScikitLearn 0.24.2

## Overview
Semantic Role Labeling (SRL), with the predicate known a priori, typically consists of two tasks: predicate sense disambiguation (WSD) and argument classification (AC). Most existing work and the previous state-of-the-art method tackled these two tasks separately, without interaction. In this work we focus on AC, which is argubaly the more important task for SRL. All existing work, AFAWK, performed argument classification in a label-blind manner. Namely, the model is tasked to classify tokens into argument labels such as A0, A1, AM-TMP, without any knowledge of what these means. Indeed, the these semantic roles or arguments were carefully defined by linguists, and those definitions are readily available in standard datasets. For example, for the predicate "work", A0 is the _worker_, A1 is the _job_, and A2 is the _employer_.

In this work, we show that by exposing these definitions, models can advance the state-of-the-art of SRL. We work on the standard CoNLL09 dataset. Unless mentioned, we work with the English data.

## CoNLL09 Frame Preparation
> Note: you can skip this step if you do not intend to make changes to the process. The processed frame files are stored as `.pkl` in the `pickles/` folder.

_Frames_ are files that contain information for each predicate sense, including arguments, definitions, examples, and notes. To process the frames, first unzip the two frame sets in `data/`, then run:

> python parse_frames_en.py

Each language requires specific processing, for which only Chinese is implemented for now. For Chinese, run:

> python parse_frames_zh.py

This creates several `.pkl` pickle files. Useful to us is `predicate_role_definition_N|V.pkl` which maps a predicate-sense to the set of arguments and definitions, divided by noun and verb predicates. This will be used in the steps below that add definitions to the data.

## CoNLL09 Data Preparation
1. It all starts with the `.txt` and `.sent` files in `data/conll09/en` (1 for each split). The `.txt` files are the original CoNLL09 data files, while the `.sent` files are already extracted relevant information using `conll09_to_sent.py`. You most likely do NOT need to run this script.

2. To create a purged dataset, namely removing all AC examples whose predicates are not in frame files, run:

> bash purge_data_by_predicate.sh

This produces `.purged.sent` files. For other languages, an example usage is:

> python purge_data_by_predicate.py data/conll09/zh/CoNLL2009-ST-evaluation-Chinese.sent zh

This needs to be repeated for every `sent` file to purge.


3. To add definitions based on the predicted sense for the evaluation data, the predictions from a WSD model is required. To do so, first replace the gold senses in the `.sent` files with predicted sense, by running:

> bash replace_with_predicted_sense_in_sent.sh

The examples above uses WSD with just sense numbers. For predictions with verb-dot-sense, change "no" to "yes" in the arguments.


4. To convert the `.sent` files to token classification for AC and sequence classification for WSD, run:

> python conll09_to_ac_wsd.py en

To do so for all languages, change the argument "en" to "all". This produces a `.ac.tsv` and a `.wsd.tsv` file for each split. 


5. To add definitions to the `.sent` files which have gold senses in them, based on the gold sense, run:

> python add_argdef_to_ac.py en

This produces a `.acd.tsv` file for each split, for both the full and purged data. By default, this is done using the "bucket-contextual" setting (see more in the presentation). 


6. To run few-shot experiments, first create randomly sampled traning data from `CoNLL2009-ST-English-train.ac.tsv` for AC and `CoNLL2009-ST-English-train.acd.tsv` for ACD, by running:

> python build_data_portions.py

- Look for few shot data at `data/conll09/en/few_shot`



## Modeling Experiments

When running each task (AC, ACD, WSD) for the first time, a pickle file will be created containing the mapping between labels and ID's. In future runs, this mapping should remain constant. 

The commands below train models. To make inferences, first download [pre-trained models and pickles](https://ibm.box.com/s/hp2id89ok5xjq7ewtfq00ya6jq8k7s53), unzip them so `results/` and `pickes/` are present, and then replace the `--train` flag in each of the commands with `--pred`. This would create an `_output.tsv` file in the `preds/` directory. With `--pred`, the flags decide which model to load. To load a specific model, use `--load_model path_to_model_checkpoint`.

To use other model architectures such as mBERT, simply change the `--model` argument.

To run few-shot experiments, use `--train_size_index`. For example, `--train_size_index 100_1` for 100 training examples with seed 1. 

W1. Run WSD model, where labels are just sense numbers (e.g. 01) on the full data
> python run_srl.py --train --model roberta-base --task WSD

> To Run WSD model, where labels are verb dot sense numbers (e.g. purchase.01), use `--verbsense`

W2. Run WSD model, where labels are just sense numbers (e.g. 01) on the purged data
> python run_srl.py --train --model roberta-base --task WSD --purged

A1. Run AC model on the full data
> python run_srl.py --train --model roberta-base --task AC

A2. Run AC model on the purged data
> python run_srl.py --train --model roberta-base --task AC --purged

D1. Run ACD model using gold senses on the full data
> python run_srl.py --train --model roberta-base --task ACD

D2. Run ACD model using gold senses on the purged data
> python run_srl.py --train --model roberta-base --task ACD --purged

Note that with ACD the model is always trained on and validated on sets whose definitions come from gold labels. When `--acd_predsense` and `--pred` is specified, it evaluates on test sets whose definitions come from predicted labels. The next two commands, when run with `--train`, train identical models as the previous two commands. They are included for the sake of completeness.

D3 Run ACD model using predicted senses on the full data
> python run_srl.py --train --model roberta-base --task ACD --acd_predsense

D4. Run ACD model using predicted senses on the purged data
> python run_srl.py --train --model roberta-base --task ACD --purged --acd_predsense


## Processing Predictions

### WSD evaluation

To calculate accuracy from a WSD prediction file, run:
> python accuracy.py wsd_prediction_file

### AC evaluation

To calculate argument P, R, F1 from an AC prediction file, run:
> python micro_f1.py ac_prediction_file

To convert an AC prediction file to the CoNLL09 format, run:
> python output_to_conll09.py en ac_prediction_file

If the prediction file has both in- and out-domain splits, the command converts both. 

To evaluate a CoNLL09 output using the official scoring script (THIS CANNOT BE DONE WITH THE PURGED SET), run:
> python conll09_scorer.py en conll09_prediction_file

As above, if the prediction file has both in- and out-domain splits, the command scores both. 

### ACD evaluation

An ACD prediction must first be converted to the same format as AC, by running:
> python process_acd_pred.py --target in|out \[--predsense\] \[--purged\]

This produces an output file prefixed with `ACD_AC_`, which is now in the AC format. Note that this script requires an AC prediction file on the same dataset (i.e. purged vs. unpurged), as a reference for _formatting_. The predictios from this AC prediction file will NOT be used. 

For example,
> python process_acd_pred.py --target in --predsense --purged
would take process `preds/ACD_roberta-base_en_in-domain_predsense_purged_output.tsv`, while using the format from `preds/AC_roberta-base_en_in-domain_purged_output.tsv` , resulting in `preds/ACD_AC_roberta-base_en_in-domain_predsense_purged_output.tsv`.

To run few-shot experiments, use `--train_size`. For example, `--train_size 100_1` for 100 training examples with seed 1. 

Then, follow the steps above in AC evaluation to evaluate.




If you find our work useful, please cite
```
@inproceedings{zhang-etal-2022-label,
    title = "Label Definitions Improve Semantic Role Labeling",
    author = "Zhang, Li and
     Jindal, Ishan and
     Li, Yunyao",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = july,
    year = "2022",
    address = "Seattle, USA",
    publisher = "Association for Computational Linguistics"
}
```
