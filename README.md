# _Quality over Quantity: An Effective Large-Scale Dataset Reduction Strategy Based on Pointwise V-Information_





# ✨  Getting Started ✨

## Requirements
Install the packages in requirements.txt

```sh
pip install -r requirements.txt
```

## Dtatasets
- The OCNLI dataset and CMNLI dataset are provided under the corresponding repositories of the papers cited in the experimental section of our paper, and can be downloaded directly.
- The CINLI dataset can be obtained [here](https://github.com/liucongg/NLPDataSet).


## Run the code

In ch_augment.py, instantiate the transformations and call .transform().

```sh
import ch_augment as augment
augment.OCNLIStandardTransformation('./data').transform()
augment.OCNLINullTransformation('./data').transform()
```

Use ch_run_glue_no_trainer.py to train two models: SIM (standard input model), EIM (empty input model).

```sh
python ch_run_glue_no_trainer.py \
  --model_name_or_path hfl/chinese-bert-wwm \
  --tokenizer_name hfl/chinese-bert-wwm \
  --train_file ./data/ocnli_train_standard.csv \
  --validation_file ./data/ocnli_train_standard.csv \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir ./finetuned_models/chinese-bert-wwm_ocnli_std
```

```sh
!python new_ch_run_glue_no_trainer.py \
  --model_name_or_path hfl/chinese-bert-wwm \
  --tokenizer_name hfl/chinese-bert-wwm \
  --train_file ./data/ocnli_train_null.csv \
  --validation_file ./data/ocnli_train_standard.csv \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir ./finetuned_models/chinese-bert-wwm_ocnli_null
  ```
  
Evaluate V-usable information using ch_v_info.py. You can get a CSV file of the pointwise V-information (PVI) values. Then data reduction can be carried out based on this



```sh
from ch_v_info import v_info
v_info(
    f"./data/ocnli_test_standard.csv",
    f"./finetuned_models/chinese-bert-wwm_ocnli_std",
    f"./data/ocnli_test_null.csv",
    f"./finetuned_models/chinese-bert-wwm_ocnli_null",
    'hfl/chinese-bert-wwm',
    out_fn="PVI/chinese-bert-wwm_test.csv"
)
 ```



