from typing import List, Dict
from tqdm.auto import tqdm
from transformers import (
    HfArgumentParser, TrainingArguments,
    set_seed, default_data_collator
)
from dataclasses import dataclass, field
from utils.dataset import SelfImproveDataset
from runners.trainer.train_self_improve import (
    LossMaskedTrainer,
    MyTrainingArguments, LoggerArguments, DataArguments, ModelArguments,
    get_wrapped_model, _remove_optimizer_weights
)
from models.base import Evaluator, GenerativeModel
from models.wrappers import (
    LLM_QA,
    LLM_WordSorting, LLM_WordSorting_Feedback_NoCorrect_Tabular, GPT_WordSorting, GPT_WordSort_PseudoSelfImprove,
    LLM_DateUnderstanding, LLM_DateUnderstanding_Feedback_NoCorrect_Tabular, GPT_DateUnderstanding, GPT_DateUnderstanding_PseudoSelfImprove,
    LLM_MultistepArithmetic, LLM_MultistepArithmetic_Feedback_NoCorrect_Tabular, GPT_MultistepArithmetic, GPT_MultistepArithmetic_PseudoSelfImprove,
    LLM_LogicalDeduction, LLM_LogicalDeduction_Feedback_NoCorrect_Tabular, GPT_LogicalDeduction, GPT_LogicalDeduction_PseudoSelfImprove
)
from models.verifier.word_sorting import Scripted_WordSort_Feedback
from models.verifier.multistep_arithmetic import Scripted_MultistepArithmetic_Feedback
from models.evaluation.word_sorting import WordSortingEvaluator, WordSortingTrainingEvaluator
from models.evaluation.date_understanding import DateUnderstandingTrainingEvaluator, DateUnderstandingEvaluator
from models.evaluation.multistep_arithmetic import MultistepArithmeticEvaluator, MultistepArithmeticTrainingEvaluator
from models.evaluation.logical_deduction import LogicalDeductionEvaluator, LogicalDeductionTrainingEvaluator
from models.self_improve.word_sorting import SelfImprove_GPT_WordSorting
from models.self_improve.date_understanding import SelfImprove_GPT_DateUnderstanding
from models.self_improve.multistep_arithmetic import SelfImprove_GPT_MultistepArithmetic
from models.self_improve.logical_deduction import SelfImprove_GPT_LogicalDeduction
from models.rl.base import LLMEditor
from models.rl.word_sorting import LLMEditor_WordSort
from models.rl.date_understanding import LLMEditor_DateUnderstanding
from models.rl.multistep_arithmetic import LLMEditor_MultistepArithmetic
from models.rl.logical_deduction import LLMEditor_LogicalDeduction
from models.filters.base import ParsedTrajectory

import json
import jsonlines
import sys
import os
import transformers
import wandb
import pickle
import random
import ray
import deepspeed
import math
import shutil
import torch
os.environ['WANDB_PROJECT'] = 'TriPosT'
os.environ['prestart_worker_first_driver'] = '0'  # otherwise RAY keep spawning a lot of RAY::IDLE processes
print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
ray.init(num_cpus=16, num_gpus=1)


DEBUG = True if os.environ.get('DEBUG', 'false').lower() == 'true' else False
print(f"DEBUG: {DEBUG}")
RUN_ID = ""  # since we are using ray.remote, we will need to reinit wandb for each remote call


@dataclass
class RLArguments:
    """
    Arguments pertaining to rl style training
    """

    verifier_llm: str = field(
        default="code-davinci-002",
        metadata={"help": "The LLM model to use for verification."},
    )
    improve_llm: str = field(
        default="code-davinci-002",
        metadata={"help": "The LLM model to use for improvement."},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Whether to print out the trajectories."},
    )
    num_iterations: int = field(
        default=3, # to go through the entire training set
        metadata={"help": "Number of iterations to run RL for."},
    )
    max_data_length: int = field(
        default=600,
        metadata={"help": "Maximum number of data points to use for RL."},
    )
    max_data_itr: int = field(
        default=5 if not DEBUG else 3,
        metadata={"help": "Maximum number of iterations to collect data for RL."},
    )
    min_data_length: int = field(
        default=200 if not DEBUG else 30,
        metadata={"help": "Minimum number of data points to use for RL. If not enough, terminate RL."},
    )


@dataclass
class RLDataArguments(DataArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dset: str = field(
        default="",
        metadata={"help": "not used for RL"},
    )
    train_world_dset: str = field(
        default='',
        metadata={"help": "Path to training world dataset. e.g. 'data/training/gsm8k/gsm8k_world_rationale_1-5.jsonl' for gsm8k"},
    )
    end_data_idx: None = field(
        default=None,
        metadata={"help": "End index of newly collected data to train on."},
    )
    collect_train_data_window: int = field(
        default=1000,
        metadata={"help": "Number of data points to collect for RL."},
    )
    train_world_size: str = field(
        default='all',
        metadata={"help": "Your training dataset size. Use for data collection during RL."},
    )
    eval_model_wrapper_cls: str = field(
        default='self-improve',
        metadata={"help": "Model wrapper class for evaluation."},
    )
    convert_to_turns: bool = field(
        default=False,
        metadata={"help": "Whether to convert trajectory data to individual turns."},
    )
    improve_data_ratio: float = field(
        default=1.0,
        metadata={"help": "How much of the collected data contains 'Updated Answer'. @imp_data_ratio:1:1 = num_update:num_non_update:ground_truth. Use -1 to not balance at all"},
    )
    self_improve_section_weight: float = field(
        default=1.0,
        metadata={"help": "How much to weight the self-improve section."},
    )
    verifier_use_scripted: bool = field(
        default=True,
        metadata={"help": "Whether to use scripted verifier for evaluation."},
    )
    save_every_rl_epoch: bool = field(
        default=False,
        metadata={"help": "Whether to save the model every RL epoch."},
    )

    def __post_init__(self):
        if self.train_world_dset == '':
            raise ValueError("Need a train_world_dset file.")
        assert self.eval_model_wrapper_cls == 'self-improve', \
            "RLDataArguments only supports self-improve model wrapper"
        
        # check if the train_world_dset is correct
        if self.task == 'word_sort':
            assert('ws_' in self.train_world_dset or 'word_sorting_' in self.train_world_dset)
        elif self.task == 'date_understanding':
            assert('date_understanding_' in self.train_world_dset)
        elif self.task == 'multistep_arithmetic':
            assert('multistep_arithmetic_' in self.train_world_dset)
        elif self.task == 'logical_deduction':
            assert('logical_deduction_' in self.train_world_dset)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
        if self.train_world_size == 'all':
            with jsonlines.open(self.train_world_dset) as reader:
                training_data = list(reader)
            self.train_world_size = len(training_data)
            print(f"train_world_size is set to {self.train_world_size}")
        else:
            self.train_world_size = int(self.train_world_size)
        assert self.train_world_size >= self.collect_train_data_window, \
            f"train_world_size {self.train_world_size} must be larger than collect_train_data_window {self.collect_train_data_window}"
        
        if self.end_data_idx is not None:
            print("WARNING: end_data_idx is not None, this means you are not training on the entire newly collected data during RL.")

        # check verifier_use_scripted setting
        if self.verifier_use_scripted:
            supported_tasks = ['word_sort', 'multistep_arithmetic']
            if self.task not in supported_tasks:
                raise ValueError(f"verifier_use_scripted is only supported for {supported_tasks} tasks.")
        print(f"verifier_use_scripted: {self.verifier_use_scripted}")
        print(f"{self.improve_data_ratio=}")
        return
    

@dataclass
class RLModelArguments(ModelArguments):
    init_eval_rationale: bool = field(
        default=False,
        metadata={"help": "Whether the first model is a rationale based model. If False, wrapper class becomes self-improve."},
    )

    def __post_init__(self):
        if self.init_eval_rationale and "baseline" not in self.model_name_or_path:
            raise ValueError("init_eval_rationale is True but model_name_or_path does not contain 'baseline'")
        if not self.init_eval_rationale and "baseline" in self.model_name_or_path:
            raise ValueError("init_eval_rationale is False but model_name_or_path contains 'baseline'")
        return


@dataclass
class RLTrainingArguments(MyTrainingArguments):
    num_train_epochs: int = 4 if not DEBUG else 1
    logging_steps: int = 20 if not DEBUG else 5
    do_eval: bool = False
    do_predict: bool = False
    save_strategy: str = "epoch"
    save_steps: int = 150 if not DEBUG else 30  # not used
    save_total_limit: int = 1
    load_best_model_at_end: bool = False


def get_tester(args: RLDataArguments):
    evaluator: Evaluator
    if args.task == 'word_sort':
        evaluator = WordSortingEvaluator(
            split="validation", 
            subtask='all', 
            eval_start_idx=0,
            eval_end_idx=None if not DEBUG else 32,
            batch=4,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'date_understanding':
        evaluator = DateUnderstandingEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None if not DEBUG else 32,
            batch=4,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'multistep_arithmetic':
        evaluator = MultistepArithmeticEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None if not DEBUG else 32,
            batch=4,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'logical_deduction':
        evaluator = LogicalDeductionEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None if not DEBUG else 32,
            batch=4,
            shuffle=False,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return evaluator


@ray.remote(num_gpus=1)
def test(itr: int, checkpoint_path, tokenizer, data_args: RLDataArguments, training_args: RLTrainingArguments):
    global RUN_ID
    # evaluate and save data, 42 is used in the evaluator file
    set_seed(42)
    evaluator = get_tester(data_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model = model.cuda()

    wrapped_model = get_wrapped_model(model, tokenizer, data_args)

    correct_choice, pred_choice, preds, result = evaluator.evaluate(wrapped_model)
    if RUN_ID != '':
        wandb.init(id=RUN_ID, resume='must')
        wandb.log(result)
        wandb.finish()
    
    output_data = {
        'logs': getattr(wrapped_model, 'logs', None),
        'correct_choice': correct_choice,
        'pred_choice': pred_choice,
        'preds': preds,
    }
    eval_file_path = os.path.join(training_args.output_dir, f'test_data_itr{itr}.pkl')
    with open(eval_file_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(json.dumps(result, indent=2))
    return


def get_evaluator(itr, args: RLDataArguments):
    window_size = args.collect_train_data_window if not DEBUG else 48
    # so that if full data is 6700, it goes 6000:None, and then 0:1000
    start_idx = (itr * window_size) % args.train_world_size
    start_idx = (start_idx // window_size) * window_size
    end_idx = None if start_idx + window_size > args.train_world_size else start_idx + window_size

    evaluator: Evaluator
    if args.task == 'word_sort':
        evaluator =  WordSortingTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=4,
            shuffle=False,
            verbose=DEBUG
        )
    elif args.task == 'date_understanding':
        evaluator = DateUnderstandingTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=4,
            shuffle=False,
            verbose=DEBUG
        )
    elif args.task == 'multistep_arithmetic':
        evaluator = MultistepArithmeticTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=4,
            shuffle=False,
            verbose=DEBUG
        )
    elif args.task == 'logical_deduction':
        evaluator = LogicalDeductionTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=4,
            shuffle=False,
            verbose=DEBUG
        )
    return evaluator


def get_wrapped_model_for_data(itr, model, tokenizer, data_args: RLDataArguments, model_args: RLModelArguments):
    wrapped_model: GenerativeModel
    gen_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
    }
    if itr == 0 and model_args.init_eval_rationale:
        if data_args.task == 'word_sort':
            print('using GPT_WordSort_PseudoSelfImprove')
            wrapped_model = GPT_WordSort_PseudoSelfImprove(
                model, tokenizer, additional_info=" Let's think step by step.\n",
                input_max_length=256, max_new_tokens=1024, gen_kwargs=gen_kwargs
            )
        elif data_args.task == 'date_understanding':
            print('using GPT_DateUnderstanidng_PseudoSelfImprove')
            wrapped_model = GPT_DateUnderstanding_PseudoSelfImprove(
                model, tokenizer, additional_info=" Let's think step by step.\n", 
                input_max_length=256, max_new_tokens=512, gen_kwargs=gen_kwargs
            )
        elif data_args.task == 'multistep_arithmetic':
            additional_info = (
                'Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). '
                'So, remember to always compute the expressions inside parentheses or brackets first.'
            )
            wrapped_model = GPT_MultistepArithmetic_PseudoSelfImprove(
                model, tokenizer, additional_info=f" Let's think step by step. {additional_info}\n", 
                input_max_length=256, max_new_tokens=512, gen_kwargs=gen_kwargs
            )
        elif data_args.task == 'logical_deduction':
            wrapped_model = GPT_LogicalDeduction_PseudoSelfImprove(
                model, tokenizer, additional_info=f""" Let's think step by step. Let "??" represents 0 or more objects, and "?" represents exactly 1 object.\n""",
                input_max_length=256, max_new_tokens=1024, gen_kwargs=gen_kwargs
            )
        else:
            raise NotImplementedError
    else:
        if data_args.task == 'word_sort':
            if data_args.eval_model_wrapper_cls == 'self-improve':
                wrapped_model = SelfImprove_GPT_WordSorting(model, tokenizer, manual_prompt=data_args.convert_to_turns, gen_kwargs=gen_kwargs)
            elif data_args.eval_model_wrapper_cls == 'ao':
                wrapped_model = GPT_WordSorting(model, tokenizer, additional_info='', input_max_length=128, max_new_tokens=128, gen_kwargs=gen_kwargs)
            else:
                wrapped_model = GPT_WordSorting(model, tokenizer, additional_info=" Let's think step by step.\n", input_max_length=128, max_new_tokens=1024, gen_kwargs=gen_kwargs)
        elif data_args.task == 'date_understanding':
            if data_args.eval_model_wrapper_cls == 'self-improve':
                wrapped_model = SelfImprove_GPT_DateUnderstanding(model, tokenizer, manual_prompt=data_args.convert_to_turns, gen_kwargs=gen_kwargs)
            elif data_args.eval_model_wrapper_cls == 'ao':
                wrapped_model = GPT_DateUnderstanding(model, tokenizer, additional_info='', input_max_length=128, max_new_tokens=10, gen_kwargs=gen_kwargs)
            else:
                wrapped_model = GPT_DateUnderstanding(model, tokenizer, additional_info=" Let's think step by step.\n", input_max_length=256, max_new_tokens=512, gen_kwargs=gen_kwargs)
        elif data_args.task == 'multistep_arithmetic':
            additional_info = (
                'Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). '
                'So, remember to always compute the expressions inside parentheses or brackets first.'
            )
            if data_args.eval_model_wrapper_cls == 'self-improve':
                wrapped_model = SelfImprove_GPT_MultistepArithmetic(model, tokenizer, manual_prompt=data_args.convert_to_turns, additional_info=f" Let's think step by step. {additional_info}")
            elif data_args.eval_model_wrapper_cls == 'ao':
                wrapped_model = GPT_MultistepArithmetic(model, tokenizer, additional_info='', input_max_length=256, max_new_tokens=10)
            else:
                wrapped_model = GPT_MultistepArithmetic(model, tokenizer, additional_info=f" Let's think step by step. {additional_info}\n", input_max_length=256, max_new_tokens=1024)
        elif data_args.task == 'logical_deduction':
            additional_info = """Let "??" represents 0 or more objects, and "?" represents exactly 1 object."""
            if data_args.eval_model_wrapper_cls == 'self-improve':
                wrapped_model = SelfImprove_GPT_LogicalDeduction(model, tokenizer, manual_prompt=data_args.convert_to_turns, additional_info=f" Let's think step by step. {additional_info}")
            elif data_args.eval_model_wrapper_cls == 'ao':
                wrapped_model = GPT_LogicalDeduction(model, tokenizer, additional_info='', input_max_length=400, max_new_tokens=10)
            else:
                wrapped_model = GPT_LogicalDeduction(model, tokenizer, additional_info=f" Let's think step by step. {additional_info}\n", input_max_length=400, max_new_tokens=512)
        else:
            raise NotImplementedError(f'Unknown task {data_args.task}')
    return wrapped_model


def evaluate(itr, model, tokenizer, data_args: RLDataArguments, model_args: RLModelArguments):
    wrapped_model = get_wrapped_model_for_data(itr, model, tokenizer, data_args, model_args)

    evaluator = get_evaluator(itr, data_args)
    correct_choice, pred_choice, preds, result = evaluator.evaluate(wrapped_model)

    question_to_rationale = {}
    for data in evaluator.dataset:
        question = data['question'].strip()
        if 'formatted_choices' in data:
            formatted_choices = data['formatted_choices']
            question += '\nOptions:\n' + formatted_choices
        attempted_answer = data['rationale'].strip()
        question_to_rationale[question] = attempted_answer
    return correct_choice, pred_choice, preds, question_to_rationale


def get_editor(args: RLArguments, data_args:RLDataArguments):
    improve_model: LLM_QA
    verifier_model: GenerativeModel
    editor: LLMEditor
    if data_args.task == 'word_sort':
        improve_model = LLM_WordSorting(
            model_name=args.improve_llm,
            verbose=False,
            temperature=0.7,
            max_tokens=2048,
        )
        if data_args.verifier_use_scripted:
            verifier_model = Scripted_WordSort_Feedback(
                fb_wrong_segment=True,
                fb_reason=True,
                fb_specific_reason=True
            )
        else:
            verifier_model = LLM_WordSorting_Feedback_NoCorrect_Tabular(
                model_name=args.verifier_llm,
                verbose=False,
                temperature=0.0,
                max_tokens=2048,
            )

        editor = LLMEditor_WordSort(
            improve_model,
            verifier_model,
            verbose=args.verbose if not DEBUG else True,
        )
    elif data_args.task == 'date_understanding':
        improve_model = LLM_DateUnderstanding(
            model_name=args.improve_llm,
            verbose=False,
            temperature=0.7,
            max_tokens=2048,
        )

        verifier_model = LLM_DateUnderstanding_Feedback_NoCorrect_Tabular(
            model_name=args.verifier_llm,
            verbose=False,
            temperature=0.0,
            max_tokens=2048,
        )

        # generates feedback only when answer is wrong. If answer is correct the feedback is scripted
        editor = LLMEditor_DateUnderstanding(
            improve_model,
            verifier_model,
            verbose=args.verbose if not DEBUG else True,
        )
    elif data_args.task == 'multistep_arithmetic':
        improve_model = LLM_MultistepArithmetic(
            model_name=args.improve_llm,
            verbose=False,
            temperature=0.7,
            max_tokens=2048,
        )

        if data_args.verifier_use_scripted:
            verifier_model = Scripted_MultistepArithmetic_Feedback(
                fb_wrong_segment=True,
                fb_reason=True,
                fb_specific_reason=True
            )
        else:
            verifier_model = LLM_MultistepArithmetic_Feedback_NoCorrect_Tabular(
                model_name=args.verifier_llm,
                verbose=False,
                temperature=0.0,
                max_tokens=2048,
            )

        editor = LLMEditor_MultistepArithmetic(
            improve_model,
            verifier_model,
            verbose=args.verbose if not DEBUG else True,
        )
    elif data_args.task == 'logical_deduction':
        improve_model = LLM_LogicalDeduction(
            model_name=args.improve_llm,
            verbose=False,
            temperature=0.7,
            max_tokens=2048,
        )

        verifier_model = LLM_LogicalDeduction_Feedback_NoCorrect_Tabular(
            model_name=args.verifier_llm,
            verbose=False,
            temperature=0.0,
            max_tokens=2048,
        )

        editor = LLMEditor_LogicalDeduction(
            improve_model,
            verifier_model,
            verbose=args.verbose if not DEBUG else True,
        )
    else:
        raise NotImplementedError
    
    assert improve_model.additional_info.strip() == editor.additional_info, \
        f"{improve_model.additional_info=}\n{editor.additional_info=}"
    return editor


def __collect_data(args: RLArguments, data_args:RLDataArguments, all_correct_choice, all_pred_choice, all_preds, unique_data: set, log_file_path: str):
    set_seed(42)

    editor = get_editor(args, data_args)
    corrected_trajectories = editor.edit(all_preds, all_pred_choice, all_correct_choice)
    
    # edit the trajectories
    new_all_data = []
    num_duplicates = 0
    for data in corrected_trajectories:
        if data in unique_data:
            num_duplicates += 1
            continue
        unique_data.add(data)
        if "Updated Answer" not in data and "final response is also correct" in data:
            text_att_start_text = "Q:"
        else:
            text_att_start_text = "\nFeedback:"
        new_all_data.append({
            "text": data,
            "text_att_start_text": text_att_start_text
        })
    jsonlines.open(log_file_path, "w").write_all(editor.log)
    print(f"to process trajectories: {len(corrected_trajectories)}")
    print(f"num duplicates: {num_duplicates}")
    return new_all_data, unique_data, editor


def __filter_n_convert_n_addgt_trajectories(all_data, data_args: RLDataArguments, additional_info = ""):
    all_converted_data: Dict[str, list] = {'non_improvement': [], 'improvement': []}
    for data in all_data:
        # convert to turns, and include non-improvement trajectories
        if 'Updated' not in data['text']:
            all_converted_data['non_improvement'].append(data)
        else:
            if data_args.convert_to_turns:
                parsed_data = ParsedTrajectory(data['text'])
                parsed_data_turns = parsed_data.parse_trajectory()
                question = parsed_data_turns.pop(0)

                while len(parsed_data_turns) > 0:
                    att = parsed_data_turns.pop(0)
                    fb = parsed_data_turns.pop(0)
                    if len(parsed_data_turns) > 0:
                        imp = parsed_data_turns[0]
                        formatted_data = f"""
                        {question}
                        Answer: Let's think step by step. {additional_info}
                        {att}
                        Feedback: {fb}
                        Updated Answer: Let's think step by step. {additional_info}
                        {imp}
                        """.replace("    ", "").replace(' \n', '\n').strip()
                        all_converted_data['improvement'].append({
                            "text": formatted_data,
                            "text_att_start_text": "\nFeedback:"
                        })
                    else:
                        formatted_data = f"""
                        {question}
                        Answer: Let's think step by step. {additional_info}
                        {att}
                        Feedback: {fb}
                        """.replace("    ", "").replace(' \n', '\n').strip()
                        all_converted_data['non_improvement'].append({
                            "text": formatted_data,
                            "text_att_start_text": "\nFeedback:"
                        })
            else:
                all_converted_data['improvement'].append(data)
    
    # finally consider the proportion
    # use 1:1:8 for non_improvement:gt:improvement
    num_improvement = len(all_converted_data['improvement'])
    if data_args.improve_data_ratio > 0.0:
        get_num_non_improvement = math.ceil(num_improvement / data_args.improve_data_ratio)
        get_num_non_improvement = min(get_num_non_improvement, len(all_converted_data['non_improvement']))
    else:
        # ablation study: do not balance the data
        get_num_non_improvement = len(all_converted_data['non_improvement'])
    if get_num_non_improvement == 0:
        non_improvement_examples_sampled = []
    else:
        non_improvement_examples_sampled = random.sample(all_converted_data['non_improvement'], get_num_non_improvement)

    all_collected_data = non_improvement_examples_sampled + all_converted_data['improvement']

    stats = {
        'num_non_improvement': len(non_improvement_examples_sampled), 
        'num_improvement': len(all_converted_data['improvement'])
    }
    for k, v in stats.items():
        print(f"{k}: {v}")
    return all_collected_data, stats


@ray.remote(num_gpus=1)
def collect_data(itr: int, checkpoint_path, tokenizer, data_args: RLDataArguments, rl_args: RLArguments, model_args: RLModelArguments, training_args: RLTrainingArguments):
    global RUN_ID
    print(f"collecting data for itr {itr}. Using checkpoint {checkpoint_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path
    )
    model = model.cuda()
    
    all_preds = []
    all_new_data: List[dict] = []
    all_unique_data: set = set()
    all_update_stats = {}

    i = 0
    max_itr = rl_args.max_data_itr
    while len(all_new_data) < rl_args.max_data_length and i < max_itr:
        set_seed(40 + i)
        # evaluate and save data, 42 is used in the evaluator file
        correct_choice, pred_choice, all_preds, question_to_rationale = evaluate(itr, model, tokenizer, data_args, model_args)
        editor_log_file_path = os.path.join(training_args.output_dir, f"editor_log_itr{itr}_i{i}.jsonl")
        new_data, all_unique_data, editor = __collect_data(rl_args, data_args, correct_choice, pred_choice, all_preds, all_unique_data, editor_log_file_path)
        update_stats = editor.update_stats
        new_data, _update_stats = __filter_n_convert_n_addgt_trajectories(new_data, data_args, editor.additional_info)
        all_new_data.extend(new_data)

        # update all_update_stats
        update_stats.update(_update_stats)
        for k, v in update_stats.items():
            if k not in all_update_stats:
                all_update_stats[k] = 0
            all_update_stats[k] += v
        print(f"itr {i} collected {len(new_data)} data, total {len(all_new_data)} data")
        i += 1

    gt_examples = []
    # only for GT trajectories we need to check for duplicates, since the other ones are checked in __collect_data
    for question, rationale in question_to_rationale.items():
        num_steps = len(rationale.split('\n')) - 1
        feedback = f"Step (1) to step ({num_steps}) are correct. The final response is also correct."
        formatted_data = f"""
        Q: {question.strip()}
        Answer: Let's think step by step. {editor.additional_info}
        {rationale.strip()}
        Feedback: {feedback}
        """.replace('    ', '').replace(' \n', '\n').strip()
        if formatted_data in all_unique_data:
            continue
        gt_examples.append({
            "text": formatted_data,
            "text_att_start_text": "Q:"  # this is gold, so learn all
        })
        all_unique_data.add(formatted_data)
    gt_num_samples = min(all_update_stats['num_non_improvement'], len(gt_examples))
    gt_examples_sampled = random.sample(gt_examples, gt_num_samples)
    all_new_data.extend(gt_examples_sampled)
    
    num_trajectory = 0
    for d in all_new_data:
        if 'Updated' in d['text']:
            num_trajectory += 1
    
    train_file_path = os.path.join(training_args.output_dir, f'train_itr{itr}.jsonl')
    random.shuffle(all_new_data)
    jsonlines.open(train_file_path, "w").write_all(all_new_data)
    print(f"Collected {len(all_new_data)} training data")

    if RUN_ID != "":
        wandb.init(id=RUN_ID, resume='must')
        wandb.log({
            **all_update_stats,
            'num_ground_truth': gt_num_samples,
            'new_train_data_size': len(all_new_data),
        })
        wandb.finish()
    return all_new_data


def to_dataset(args: RLDataArguments, tokenizer, training_data):
    dset_type = args.dset_type
    if dset_type in ['word_sorting', 'date_understanding', 'multistep_arithmetic', 'logical_deduction']:
        min_input_length = args.min_input_length if not DEBUG else 256
        max_input_length = args.max_input_length if not DEBUG else 512
        train_dset = SelfImproveDataset(
            training_data, tokenizer,
            end_data_idx=None,
            min_input_length=min_input_length, max_input_length=max_input_length,
            mask_before_att_start_text=args.mask_before_att_start_text,
            self_improve_section_weight=args.self_improve_section_weight,
            shuffle=True
        )
        dummy_eval_dset = SelfImproveDataset(
            training_data, tokenizer,
            end_data_idx=int(0.1*len(training_data))+1,
            min_input_length=min_input_length, max_input_length=max_input_length,
            mask_before_att_start_text=args.mask_before_att_start_text,
            shuffle=True
        )
    else:
        raise ValueError(f"Unknown dset_type {dset_type}")
    num_train_excluded = len(training_data) - len(train_dset)
    print(f"Excluded {num_train_excluded} training examples, loaded {len(train_dset)} examples")
    return train_dset, dummy_eval_dset


def _remove_or_rename_all_checkpoint_weights(save_dir, prefix:str, rename=True):
    print(f"Removing or renaming all checkpoint weights in {save_dir}")
    for checkpoint_dir_name in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dir_name)
        if os.path.isdir(checkpoint_dir) and checkpoint_dir_name.startswith('checkpoint'):
            if rename:
                new_checkpoint_dir = os.path.join(save_dir, f"{prefix}-{checkpoint_dir_name}")
                print(f"Renaming checkpoint {checkpoint_dir} to {new_checkpoint_dir}")
                os.rename(checkpoint_dir, new_checkpoint_dir)
            else:
                print(f"Removing checkpoint {checkpoint_dir}")
                shutil.rmtree(checkpoint_dir)
    return


def _find_last_checkpoint(save_dir):
    for checkpoint_dir_name in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dir_name)
        if os.path.isdir(checkpoint_dir) and checkpoint_dir_name.startswith('checkpoint'):
            return checkpoint_dir
    raise ValueError(f"Cannot find checkpoint in {save_dir}")


@ray.remote(num_gpus=1)
def train(
        itr, checkpoint_path, tokenizer, training_data,
        data_args: RLDataArguments, training_args: TrainingArguments):
    global RUN_ID
    set_seed(training_args.seed)

    train_dset, eval_dset = to_dataset(data_args, tokenizer, training_data)
    
    if training_args.deepspeed:
        # since this is in ray remote, we need to manually setup deepspeed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9997'
        os.environ['RANK'] = "0"
        os.environ['LOCAL_RANK'] = "0"
        os.environ['WORLD_SIZE'] = "1"
        deepspeed.init_distributed()
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path
    )
    # remove all weights, since we will save new ones at the end of the training
    _remove_or_rename_all_checkpoint_weights(training_args.output_dir, f'itr{itr}', rename=data_args.save_every_rl_epoch)

    if RUN_ID != "":
        wandb.init(id=RUN_ID, resume='must')
    
    # for some reason trainer will STILL do eval, so we feed in a dummy eval dataset
    training_args.do_eval = False
    trainer = LossMaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if RUN_ID != "":
        wandb.finish()
    
    _remove_optimizer_weights(training_args.output_dir)
    last_checkpoint = _find_last_checkpoint(training_args.output_dir)
    # last checkpoint
    print(f"Last checkpoint: {last_checkpoint}")
    return last_checkpoint


def init(model_args: RLModelArguments, data_args: RLDataArguments, logger_args: LoggerArguments, training_args: RLTrainingArguments):
    global RUN_ID
    model_name: str = model_args.model_name_or_path
    try:
        # backward compatibility
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            padding=True, truncation=True, return_tensors="pt"
        )
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            padding=True, truncation=True, return_tensors="pt",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.eos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # if it is already initialized, huggingface will use it
    all_args = {
        'rl_args': vars(rl_args),
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'logger_args': vars(logger_args),
        'training_args': training_args.to_dict()
    }
    if 'wandb' in training_args.report_to:
        RUN_ID = wandb.util.generate_id()
        _ = wandb.init(
            id=RUN_ID,
            project=os.environ['WANDB_PROJECT'],
            name=training_args.output_dir.split("/")[-1] or None,
            group=logger_args.run_group,
            config=all_args,
            resume="allow",
        )
        wandb.finish()
    return model_name, tokenizer


def itr_0_load_data_if_exist(train_path):
    print(f"Loading training data from {train_path}")
    with jsonlines.open(train_path, "r") as f:
        new_train_data = list(f)
    if RUN_ID != "":
        wandb.init(id=RUN_ID, resume='must')
        wandb.log({
            'new_train_data_size': len(new_train_data),
        })
        wandb.finish()
    return new_train_data


def _get_existing_checkpoint_path(training_args: TrainingArguments):
    base_dir = training_args.output_dir
    if os.path.exists(base_dir):
        for checkpoint_dir_name in os.listdir(base_dir):
            checkpoint_dir = os.path.join(base_dir, checkpoint_dir_name)
            if os.path.isdir(checkpoint_dir) and checkpoint_dir_name.startswith('checkpoint'):
                return checkpoint_dir
    return None


def itr_0_resume_testing_if_exist(itr, tokenizer, new_train_data, data_args, training_args: TrainingArguments):
    existing_checkpoint_path = _get_existing_checkpoint_path(training_args)
    use_existing = input(f"Found existing checkpoint: {existing_checkpoint_path}, use it? (y/n) ")
    if use_existing.strip() == 'y':
        print("Using existing checkpoint")
        checkpoint_path = existing_checkpoint_path
    else:
        checkpoint_path = ray.get(train.remote(itr, checkpoint_path, tokenizer, new_train_data, data_args, training_args))
    return checkpoint_path


def main(rl_args: RLArguments, model_args: RLModelArguments, data_args: RLDataArguments, logger_args: LoggerArguments, training_args: RLTrainingArguments):
    set_seed(42)
    # initalize model
    checkpoint_path, tokenizer = init(model_args, data_args, logger_args, training_args)
    
    num_iterations = rl_args.num_iterations
    for itr in tqdm(range(num_iterations), desc="RL Iterations"):
        # use ray remote to clear memory after each **function**
        # if data exists (e.g. we crashed before), load it
        train_path = os.path.join(training_args.output_dir, f"train_itr0.jsonl")
        if itr == 0 and os.path.exists(train_path):
            new_train_data = itr_0_load_data_if_exist(train_path)
        else:
            new_train_data = ray.get(collect_data.remote(itr, checkpoint_path, tokenizer, data_args, rl_args, model_args, training_args))
        
        if len(new_train_data) < rl_args.min_data_length:
            print("Not enough training data, terminating")
            break
        
        # trainer changes the training args, so we need to 'deepcopy' it. However, for some reason doing deepcopy makes trainer.save not work
        # if checkpoint exists, ask if we want to use it (e.g. maybe we crashed during evaluation before)
        training_args_copy = TrainingArguments(**training_args.to_dict())
        if itr == 0 and _get_existing_checkpoint_path(training_args_copy) is not None:
            checkpoint_path = itr_0_resume_testing_if_exist(itr, tokenizer, new_train_data, data_args, training_args_copy)
        else:
            checkpoint_path = ray.get(train.remote(itr, checkpoint_path, tokenizer, new_train_data, data_args, training_args_copy))
        ray.get(test.remote(itr, checkpoint_path, tokenizer, data_args, training_args))
    return


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(RLArguments, RLModelArguments, RLDataArguments, LoggerArguments, RLTrainingArguments),
        description="TriPosT"
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        rl_args, model_args, data_args, logger_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        rl_args, model_args, data_args, logger_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    print('received rl_args:')
    print(json.dumps(vars(rl_args), indent=2, sort_keys=True))
    print('received model_args:')
    print(json.dumps(vars(model_args), indent=2, sort_keys=True))
    print('received data_args:')
    print(json.dumps(vars(data_args), indent=2, sort_keys=True))
    print('received logger_args:')
    print(json.dumps(vars(logger_args), indent=2, sort_keys=True))
    print('received training_args:')
    print(json.dumps(training_args.to_dict(), indent=2, sort_keys=True))
    
    # save config to model_args.model_save_path
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'all_args.json'), 'w') as f:
        all_args = {
            'rl_args': vars(rl_args),
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict()
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    # train
    main(rl_args, model_args, data_args, logger_args, training_args)