# BERT and SpanBERT for Coreference Resolution in Russian medical forums 

This repository contains code and models for BS thesis 
written by Egor Yatsishin, Moscow, NRU HSE, Fundamental and Computational Linguistics, 2021.

This work describes the first coreference resolution system for Russian medical forum data
based on transformer architecture. To get the understanding of the data we did some preparatory
work: around 700 of messages from different forums and social networks were read and typical
coreference scenarios were observed, described and classified by hand. 

Following research and production was done on a free basis by request from
Semantic Hub and with support from the mentioned company.

The produced scripts for training and prediction stages of the model are based on the paper [BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091).

The model architecture itself is an extension of the [e2e-coref](https://github.com/kentonl/e2e-coref) model.

## Overview
Semantic Hub provided a JSON-like dataset consisting of 386 254 documents with 
1 693 773 coreferential chains and as much as 2 968 033 tokens inside.

The dataset was adapted for coreference resolution task and converted to CoNLL format. 
The pipeline of the preprocessing can be observed in this [notebook](/preprocessing_pipeline.ipynb). This notebook also contains some data observations and analysis.

The next steps were the conversion from CoNLL to jsonline, training the whole model and then getting the predictions. All these steps can be found in this [notebook](/conll2spanbert.ipynb)

The above mentioned notebooks are made in such a way that represent a complete pipeline which can be implemented step-by-step from the first to the last cell to produce the results expected. The only limit is processing power. Still, for the first time it is better to read the comments and launch the code cell by cell manually.

Current average scores achieved by the model are relatively low due to the model being based on `bert-cased-base` tokenization and vocabulary.
The major park of future work is the implementation of `rubert` to the initial model training.

## Setup
* Install python3 requirements: `pip install -r requirements.txt`
* `import os
os.environ['data_dir'] = "./data"`
* `./setup_all.sh`: This builds the custom kernels

## Pretrained Coreference Models
Please download following files to use the *pretrained coreference models* on your data. If you want to train your own coreference model, you can skip this step.

| Model          | `<model_name>` for download | F1 English(%) |
| -------------- | --------------------------- |:-------------:|
| BERT-base      | bert_base                   | 73.9          |
| SpanBERT-base  | spanbert_base               | 77.7          |
| BERT-large     | bert_large                  | 76.9          |
| SpanBERT-large | spanbert_large              | 79.6          |

`./download_pretrained.sh <model_name>` (e.g,: bert_base, bert_large, spanbert_base, spanbert_large; assumes that `$data_dir` is set) This downloads BERT/SpanBERT models finetuned on OntoNotes. The original/non-finetuned version of SpanBERT weights is available in this [repository](https://github.com/facebookresearch/SpanBERT). You can use these models with `evaluate.py` and `predict.py` (the section on Batched Prediction Instructions)


## Training / Finetuning Instructions
* Finetuning a BERT/SpanBERT *large* model requires access to a 32GB GPU. You might be able to train the large model with a smaller `max_seq_length`, `max_training_sentences`, `ffnn_size`, and `model_heads = false` on a 16GB machine; this will almost certainly result in relatively poorer performance as measured on OntoNotes.
* Running/testing a large pretrained model is still possible on a 16GB GPU. You should be able to finetune the base models on smaller GPUs.

### Setup for training
`./setup_training.sh - $data_dir`. Clear BERT models will be downloaded.

* Experiment configurations are found in `experiments.conf`. Choose an experiment that you would like to run, e.g. `spanbert_base`
* Note that configs without the prefix `train_` load checkpoints already tuned on OntoNotes.
* Training: `GPU=0 python train.py <experiment>`
* Results are stored in the `log_root` directory (see `experiments.conf`) and can be viewed via TensorBoard.
* Evaluation: `GPU=0 python evaluate.py <experiment>`. This currently evaluates on the dev set.


## Batched Prediction Instructions

* Create a file where each line similar to `cased_config_vocab/trial.jsonlines` (make sure to strip the newlines so each line is well-formed json):
```
{
  "clusters": [], # leave this blank
  "doc_key": "nw", # key closest to your domain. "nw" is newswire. See the OntoNotes documentation.
  "sentences": [["[CLS]", "subword1", "##subword1", ".", "[SEP]"]], # list of BERT tokenized segments. Each segment should be less than the max_segment_len in your config
  "speakers": [["[SPL]", "-", "-", "-", "[SPL]"]], # speaker information for each subword in sentences
  "sentence_map": [0, 0, 0, 0, 0], # flat list where each element is the sentence index of the subwords
  "subtoken_map": [0, 0, 0, 1, 1]  # flat list containing original word index for each subword. [CLS]  and the first word share the same index
}
```
  * `clusters` should be left empty and is only used for evaluation purposes.
  * `doc_key` indicates the genre, which can be one of the following: `"bc", "bn", "mz", "nw", "pt", "tc", "wb"`
  * `speakers` indicates the speaker of each word. These can be all empty strings if there is only one known speaker.
* Run `GPU=0 python predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with an additional key `predicted_clusters`.

## Notes
* The current config runs the Independent model.
* When running on test, change the `eval_path` and `conll_eval_path` from dev to test.
* The `model_dir` inside the `log_root` contains `stdout.log`. Check the `max_f1` after 57000 steps. For example
``
2019-06-12 12:43:11,926 - INFO - __main__ - [57000] evaL_f1=0.7694, max_f1=0.7697
``
* You can also load pytorch based model files (ending in `.pt`) which share BERT's architecture. See `pytorch_to_tf.py` for details.

### Important Config Keys
* `log_root`: This is where all models and logs are stored. Check this before running anything.
* `bert_learning_rate`: The learning rate for the BERT parameters. Typically, `1e-5` and `2e-5` work well.
* `task_learning_rate`: The learning rate for the other parameters. Typically, LRs between `0.0001` to `0.0003` work well.
* `init_checkpoint`: The checkpoint file from which BERT parameters are initialized. Both TF and Pytorch checkpoints work as long as they use the same BERT architecture. Use `*ckpt` files for TF and `*pt` for Pytorch.
* `max_segment_len`: The maximum size of the BERT context window. Larger segments work better for SpanBERT while BERT suffers a sharp drop at 512.

### Slurm
If you have access to a slurm GPU cluster, you could use the following for set of commands for training.
* `python tune.py  --generate_configs --data_dir <coref_data_dir>`: This generates multiple configs for tuning (BERT and task) learning rates, embedding models, and `max_segment_len`. This modifies `experiments.conf`. Use `--trial` to print to stdout instead. If you need to generate this from scratch, refer to `basic.conf`.
* `grep "\{best\}" experiments.conf | cut -d = -f 1 > torun.txt`: This creates a list of configs that can be used by the script to launch jobs. You can use a regexp to restrict the list of configs. For example, `grep "\{best\}" experiments.conf | grep "sl512*" | cut -d = -f 1 > torun.txt` will select configs with `max_segment_len = 512`.
* `python tune.py --data_dir <coref_data_dir> --run_jobs`: This launches jobs from torun.txt on the slurm cluster.

### Miscellaneous
* If you like using Colab, check out Jonathan K. Kummerfeld's [notebook](https://colab.research.google.com/drive/1SlERO9Uc9541qv6yH26LJz5IM9j7YVra#scrollTo=H0xPknceFORt).
* Some `g++` versions may not play nicely with this repo. If you get this:
`tensorflow.python.framework.errors_impl.NotFoundError: ./coref_kernels.so: undefined symbol: _ZN10tensorflow12OpDefBuilder4AttrESs`, try removing the flag `-D_GLIBCXX_USE_CXX11_ABI=0` from `setup_all.sh`. Thanks to Naman Jain for the [solution](https://github.com/mandarjoshi90/coref/issues/29).

## Coding fixes and additional scripts

### rucor2conll
Corpus from [RuCor](http://rucoref.maimbava.net/) and script to convert it to CoNNLL format (modified from @lubakit [script](https://github.com/lubakit/pm_coreference_resolution/blob/b19e2004ba5dd13cfe08f5ff1227c5c9a6e30645/bin/rucor2conll.py), thanks to @polyankaglade)

### Other files
[`requirements.txt`](/requirements.txt): 
* update to `MarkupSafe==1.1.1`
* comment out `## scikit-learn==0.19.1` and `## scipy==1.0.0`
* add `tensorflow == 1.14.0`

[`minimize.py`](/minimize.py):
* comment out all `assert`
* fix `get_document()` by adding missing `stats` argument
* fix encoding for Russian by adding `encoding='utf-8'` and `ensure_ascii=False`

[`setup_all.sh`](/setup_all.sh):
* fix by removing `-D_GLIBCXX_USE_CXX11_ABI=0` flag

[`./setup_training.sh`](/setup_all.sh):
* modified for use on custom documents and without OntoNotes
