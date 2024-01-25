# TriPosT

This repository contains an official implementation of TriPosT, which is described in this paper:


**Teaching Language Models to Self-Improve through Interactive Demonstrations**<br>
*Xiao Yu, Baolin Peng, Michel Galley, Jianfeng Gao, Zhou Yu*

## Dependencies

The core dependencies used in this projects are:
```
transformer deepspeed wandb ray openai
```

- for a full list of library versions, checkout the `requirements.txt` file
- we also provide a Docker image `docker pull jasonyux/tripost:latest`. Everything should already be configured under the `/workspace/` folder if you use this image!

## TriPoST Training

This section contains examples of training TriPoST on each task. Check out:
- Section [FT from ground-truth rationales](#ft-from-ground-truth-rationales) to train the baseline models (i.e., the ones used in `--model_name_or_path` below)
- Section [Other Training Scripts](#other-training-scripts) which contains examples of all baselines used in the paper

**For a quickstart** we have uploaded some of our baseline models (LLaMA-7b finetuned with ground-truth rationale) in this [Google Drive folder](https://drive.google.com/drive/folders/1V52cXN3nRJFeY4FIRnXaju-PuIcgW750?usp=sharing). Since models weights are large in size, we are only able to upload a few of them. Please refer to Section [FT with ground-truth rationales](#ft-with-ground-truth-rationales) for examples of how obtain those for other tasks!

Logical Deduction:
```bash
pytho runners/trainer/train_self_improve_rl_noeval.py \
--output_dir model_checkpoints/logical_deduction/ld_llama-7b_text003_s1 \
--task logical_deduction \
--train_world_dset data/training/logical_deduction/logical_deduction_world_rationale_3-5.jsonl \
--max_input_length 1024 --min_input_length 256 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-6 \
--run_group llama_logical_deduction_rl \
--model_name_or_path model_checkpoints/logical_deduction/baselines/llama7b_rationale_3-5_5epoch_allsamples_s1/checkpoint-440 \
--init_eval_rationale true \
--bf16 True --tf32 True \
--deepspeed configs/ds_config.json \
--verifier_llm text003 \
--collect_train_data_window 360 \
--max_data_length 80 --min_data_length 20 \
--improve_data_ratio 1.5 \
--self_improve_section_weight 1.5 \
--convert_to_turns true \
--verifier_use_scripted False \
--mask_before_att_start_text False
```

Date Understanding:
```bash
python runners/trainer/train_self_improve_rl_noeval.py \
--output_dir model_checkpoints/date_understanding/du_llama-7b_text003_s1 \
--task date_understanding \
--train_world_dset data/training/date_understanding/date_understanding_world_rationale_1-2.jsonl --max_input_length 1024 --min_input_length 256 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-6 \
--run_group llama_date_understanding_rl \
--model_name_or_path model_checkpoints/date_understanding/baselines/llama7b_rationale_1-2_5epoch_allsamples_s1/checkpoint-240 \
--init_eval_rationale true \
--bf16 True --tf32 True \
--deepspeed configs/ds_config.json \
--verifier_llm text003 \
--collect_train_data_window 191 \
--max_data_length 100 --min_data_length 40 \
--improve_data_ratio 1.5 \
--self_improve_section_weight 1.5 \
--convert_to_turns true \
--verifier_use_scripted False \
--mask_before_att_start_text False
```

Multistep Arithmetic:
```bash
python runners/trainer/train_self_improve_rl_noeval.py \
--output_dir model_checkpoints/multistep_arithmetic/scripted_msa_llama7b_s1 \
--task multistep_arithmetic \
--train_world_dset data/training/multistep_arithmetic/multistep_arithmetic_world_rationale_l3-4d2-22.jsonl \
--max_input_length 1024 --min_input_length 256 \
--per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
--learning_rate 1e-6 \
--run_group llama_multistep_arithmetic_rl \
--self_improve_section_weight 1.5 \
--model_name_or_path model_checkpoints/multistep_arithmetic/baselines/llama7b_rationale_l3-4d2-22_5epoch_allsamples_s1/checkpoint-680 \
--init_eval_rationale true \
--bf16 True --tf32 True \
--deepspeed configs/ds_config.json \
--verifier_llm chatgpt \  # dummy value
--collect_train_data_window 550 \
--max_data_length 150 \
--min_data_length 60 \
--improve_data_ratio 1.5 \
--convert_to_turns true \
--verifier_use_scripted true \  # because we are using a scripted verifier
--mask_before_att_start_text False
```

Word Sorting:
```bash
python runners/trainer/train_self_improve_rl_noeval.py \
--output_dir model_checkpoints/word_sort/scripted_ws_llama7b_s1 \
--task word_sort \
--train_world_dset data/training/word_sorting/word_sorting_world_rationale_1-7.jsonl \
--max_input_length 1024 --min_input_length 256 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-6 \
--run_group llama_word_sorting_rl \
--model_name_or_path model_checkpoints/word_sort/baselines/ws_llama7b_rationale_1-7_5epoch_allsamples_s1/checkpoint-440 \
--init_eval_rationale true \
--bf16 True --tf32 True \
--deepspeed configs/ds_config.json \
--verifier_llm chatgpt \  # dummy value
--collect_train_data_window 433 \
--max_data_length 120 \
--min_data_length 60 \
--improve_data_ratio 1.5 \
--self_improve_section_weight 1.5 \
--convert_to_turns true \
--verifier_use_scripted true \  # because we are using a scripted verifier
--mask_before_att_start_text False
```

For LLaMA-2 results, simply switch the `model_name_or_path` to the finetuned LLaMA-2 model.

## Evaluating

You can evaluate the performance of:
1. few-shot prompting LLM
    ```bash
    # e.g. on the multistep arithmetic task
    python runners/tester/eval_llm.py \
    -o model_checkpoints/multistep_arithmetic/baselines/llm/codex_perf.pkl \
    --task multistep_arithmetic \
    --verbose
    ```
2. few-shot prompting LLM + self improvement prompting
    ```bash
    # e.g. on the multistep arithmetic task
    python runners/tester/eval_llm_self_improve.py \
    -o model_checkpoints/multistep_arithmetic/baselines/llm/codex_w_text003_perf.pkl \
    --task multistep_arithmetic \
    --return_if_correct \
    --verbose
    ```
3. finetuned small LM (e.g. LLaMA-7b) + self improvement prompting
    ```bash
    # e.g. on the multistep arithmetic task
    python runners/tester/eval_prompt_self_improve.py \
    -o model_checkpoints/multistep_arithmetic/baselines/prompt/blablabla_perf.pkl \
    --model_name_or_path model_checkpoints/multistep_arithmetic/blablabla/checkpoint-120 \
    --task multistep_arithmetic \
    --return_if_correct \
    --verbose
    ```

## Data

We stored all our training data under the `data` folder.
```
data
├── raw  # full data, including train, valid, and test
├── training  # just used for training
└── validation  # just used for validation
```

Additionally, we open-source the script we used to obtain those data for each task under `runners/data_collector/<task>` folder.

For example:
- LMSI (Huang et al. (2023)) training data at `runners/data_collector/<task>/get_lmsi_data.py`
- LLM-generated rationale `runners/data_collector/<logical_deduction, date_understanding>/get_rationale.py`
- rationale filtering scripts at `runners/data_collector/<logical_deduction, date_understanding>/filtering.py`
- script-generated rationale `runners/data_collector/<multistep_arithmetics, word_sorting>/get_rationale.py`

## Other Training Scripts

This section contain example training scripts for doing finetuning the baseline models, training from LLM demonstrations, and LMSI training.

### FT from ground-truth rationales

Basically performing knowledge distillation from the LLM-genreated or script-generated ground-truth rationales.

```bash
python runners/trainer/train_self_improve.py \
--output_dir model_checkpoints/multistep_arithmetic/baselines/llama7b_rationale_l3-4d2-22_5epoch_allsamples_s1 \
--seed 1 \
--num_train_epochs 5 \
--train_dset data/training/multistep_arithmetic/multistep_arithmetic_baseline_rationale_l3-4d2-22.jsonl \
--eval_dset data/validation/multistep_arithmetic/multistep_arithmetic_baseline_rationale_l3-4d2-22_val.jsonl \
--max_input_length 1024 \
--min_input_length 1024 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-6 \
--run_group llama_multistep_arithmetic_baseline \
--mask_before_att_start_text false \
--deepspeed configs/ds_config.json \
--model_name_or_path model_checkpoints/llama_hf \
--bf16 True --tf32 True \
--task multistep_arithmetic \
--eval_model_wrapper_cls rationale \
--eval_steps 40 \
--save_steps 40 \
--logging_steps 10
```

for LLaMA-2 baseline, swap the `model_name_or_path` to an LLaMA-2 checkpoint (e.g. from huggingface). For other tasks, simply change the `output_dir`, `train_dset`, `eval_dset`, `model_name_or_path`, and `task` accordingly.

### FT from LLM self-improvement demonstrations

here you can choose which LLMs to use as the `verifier` (to generate feedbacks) and the `llm` (to generate an initial attempt, as well as the improvements). We recommend using `Codex` (or `code-davinci-002`) as generating improvements may be costly (since the prompt + attempt may be long!).

```bash
python runners/trainer/train_self_improve_from_demo.py \
--output_dir model_checkpoints/multistep_arithmetic/baselines/llm_llama7b_rationale_l3-4d2-22_5epoch_all_samples_s1_verifier_codex \
--max_data_length 75 \
--min_data_length 30 \
--improve_data_ratio 1.5 \
--task multistep_arithmetic \
--llm code-davinci-002 \
--verifier_llm code-davinci-002 \
--train_world_dset data/training/multistep_arithmetic/multistep_arithmetic_world_rationale_l3-4d2-22.jsonl \
--model_name_or_path model_checkpoints/multistep_arithmetic/baselines/llama7b_rationale_l3-4d2-22_5epoch_allsamples_s1/checkpoint-680 \
--max_input_length 1024 \
--min_input_length 256 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-6 \
--run_group llama_multistep_arithmetic_baseline \
--mask_before_att_start_text false \
--convert_to_turns true \
--bf16 True --tf32 True \
--deepspeed configs/ds_config.json
```

for other tasks, simply change the `output_dir`, `train_world_dset`, `model_name_or_path`, and `task` accordingly.

### LMSI training

First collect data using the script at `runners/data_collector/<task>/get_lmsi_data.py`, then:

```bash
# e.g. training on the multistep arithmetic task
python runners/trainer/train_self_improve.py \
--output_dir model_checkpoints/multistep_arithmetic/to_be_moved/baselines/LMSI_llama7b_s1 \
--num_train_epochs 5 \
--train_dset data/training/multistep_arithmetic/multistep_arithmetic_LMSI_rationale_l3-4d2-22.jsonl \
--eval_dset data/validation/multistep_arithmetic/multistep_arithmetic_baseline_rationale_l3-4d2-22_val.jsonl \
--max_input_length 1024 \
--min_input_length 1024 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-6 \
--run_group llama_multistep_arithmetic_baseline \
--mask_before_att_start_text false \
--deepspeed configs/ds_config.json \
--model_name_or_path model_checkpoints/multistep_arithmetic/baselines/llama7b_rationale_l3-4d2-22_5epoch_allsamples_s1/checkpoint-680 \
--bf16 True --tf32 True \
--task multistep_arithmetic \
--eval_model_wrapper_cls rationale \
--eval_steps 40 \
--save_steps 40 \
--logging_steps 10
```

for other tasks, simply change the `output_dir`, `train_dset`, `eval_dset`, `model_name_or_path`, and `task` accordingly.

## Citation

```
@misc{yu2023teaching,
      title={Teaching Language Models to Self-Improve through Interactive Demonstrations}, 
      author={Xiao Yu and Baolin Peng and Michel Galley and Jianfeng Gao and Zhou Yu},
      year={2023},
      eprint={2310.13522},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
