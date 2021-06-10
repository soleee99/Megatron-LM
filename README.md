# Pretraining GPT
This README 
- explains some options that are custom-added for pretraining with the-pile dataset
- explains additional information about relation between options & how each option is used
Check `examples/pretrain_gpt_distributed_with_mp.sh` for actual script.

----------


### 📌 Using separate paths for eval & test datasets
In the original version, eval and test datasets were made from the data put into `--data-path` by using `--split` to determine the ratio.
Newly added: **`--no-split`, `--eval-path`, `--test-path`**

- For example, for options like below;
  - `00_text_document` and `01_text_document` will be equally-weighted
  - train dataset will be made by taking 95% of `00_text_document` and 95% of `01_text_document`
  - validation dataset will be made by taking 4% of `00_text_document` and 4% of `01_text_document`
  - test dataset will be made by taking 1% of `00_text_document` and 1% of `01_text_document`
```
DATA_PATH="1 00_text_document 1 01_text_document"
...
--data-path $DATA_PATH
--split 95, 4, 1
```

- However, the-pile provides separate `val` and `test` datasets that have been filtered (overlap checking). Thus using `--no-split` like below will;
  - not split up the documents in `DATA_PATH`, but use all of them as the train data
  - take val & test data from the path given as `--eval-path` and `--test-path` 
```
DATA_PATH="1 path_to_00_text_document 1 path_to_01_text_document"
EVAL_PATH=path_to_val_text_document
TEST_PATH=path_to_test_text_document
...
--no-split
--data-path $DATA_PATH
--eval-path $EVAL_PATH
--test-path $TEST_PATH
```

### 📌 Resuming dataloader from checkpoints
This part is not implemented, but just explains how to ensure that dataloader iterator resumes from the checkpoint.
- when no options are given, takes the torch dataloader
- when `--dataloader-type single` is given, enables resuming from the last-used data in the iterator (useful when resuming pretraining from checkpoint)
- when `--dataloader-type cyclic` is given, randomly selects data from the iterator


### 📌 Batch size options
- Each forward on a single GPU takes care of `micro-batch-size` number of batches
- Automatically performs gradient accumulation until processed number of batches add up to `global-batch-size`, and at that point updates the optimizer, and **this counts as one iteration**




