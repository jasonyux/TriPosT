from typing import List, Dict
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
    LLM_WordSorting, LLM_WordSorting_Feedback_wCorrect_Tabular,
    LLM_DateUnderstanding, LLM_DateUnderstanding_Feedback_wCorrect_Tabular,
    LLM_MultistepArithmetic, LLM_MultistepArithmetic_Feedback_wCorrect_Tabular,
    LLM_LogicalDeduction, LLM_LogicalDeduction_Feedback_wCorrect_Tabular
)
from models.evaluation.word_sorting import WordSortingEvaluator, WordSortingTrainingEvaluator
from models.evaluation.date_understanding import DateUnderstandingEvaluator, DateUnderstandingTrainingEvaluator
from models.evaluation.multistep_arithmetic import MultistepArithmeticEvaluator, MultistepArithmeticTrainingEvaluator
from models.evaluation.logical_deduction import LogicalDeductionEvaluator, LogicalDeductionTrainingEvaluator
from models.rl.multistep_arithmetic import LLMwVerifier_MultistepArithmetic
from models.rl.logical_deduction import LLMwVerifier_LogicalDeduction
from models.rl.date_understanding import LLMwVerifier_DateUnderstanding
from models.rl.word_sorting import LLMwVerifier_WordSorting
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
os.environ['WANDB_PROJECT'] = 'TriPosT'
print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


DEBUG = True if os.environ.get('DEBUG', 'false').lower() == 'true' else False
print(f"DEBUG: {DEBUG}")
RUN_ID = ""  # since we are using ray.remote, we will need to reinit wandb for each remote call


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
    eval_model_wrapper_cls: str = field(
        default='self-improve',
        metadata={"help": "Model wrapper class for evaluation."},
    )
    convert_to_turns: bool = field(
        default=True,
        metadata={"help": "Whether to convert trajectory data to individual turns."},
    )
    improve_data_ratio: float = field(
        default=1.5,
        metadata={"help": "How much of the collected data contains 'Updated Answer'. @imp_data_ratio:1:1 = num_update:num_non_update:ground_truth."},
    )
    self_improve_section_weight: float = field(
        default=1.0,
        metadata={"help": "How much to weight the self-improve section."},
    )
    verifier_use_scripted: bool = field(
        default=False,  # we do LLM here
        metadata={"help": "Whether to use scripted verifier for evaluation."},
    )
    verifier_llm: str = field(
        default="code-davinci-002",
        metadata={"help": "The LLM model to use for verification."},
    )
    llm: str = field(
        default="code-davinci-002",
        metadata={"help": "The LLM model to use for init answer AND improvement."},
    )
    return_if_correct: bool = field(
        default=True,
        metadata={"help": "Whether to return feedback based on checking the ground truth."},
    )
    max_data_length: int = field(
        default=120,
        metadata={"help": "Maximum number of data points to use for SL."},
    )
    max_data_itr: int = field(
        default=3,
        metadata={"help": "Maximum number of iterations to collect data for SL."},
    )
    min_data_length: int = field(
        default=40 if not DEBUG else 20,
        metadata={"help": "Minimum number of data points to use for RL. If not enough, terminate SL."},
    )

    def __post_init__(self):
        if self.train_world_dset == '':
            raise ValueError("Need a train_world_dset file.")
        assert self.eval_model_wrapper_cls == 'self-improve', \
            "RLDataArguments only supports self-improve model wrapper"
        assert self.verifier_use_scripted == False, \
            "We should be using fully LLM here."
        assert self.return_if_correct == True, \
            "We should be using fully LLM here with checking the ground truth for data collection."
        
        # check if the train_world_dset is correct
        if self.task == 'word_sort':
            assert('ws_' in self.train_world_dset or 'word_sorting_' in self.train_world_dset)
        elif self.task == 'gsm8k':
            assert('gsm8k_' in self.train_world_dset)
        elif self.task == 'date_understanding':
            assert('date_understanding_' in self.train_world_dset)
        elif self.task == 'hyperbaton':
            assert('hyperbaton_' in self.train_world_dset)
        elif self.task == 'multistep_arithmetic':
            assert('multistep_arithmetic_' in self.train_world_dset)
        elif self.task == 'colored_objects':
            assert('colored_objects_' in self.train_world_dset)
        elif self.task == 'logical_deduction':
            assert('logical_deduction_' in self.train_world_dset)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
        if self.end_data_idx is not None:
            print("WARNING: end_data_idx is not None, this means you are not training on the entire newly collected data during RL.")
        return


# copied from runners/trainer/train_self_improve_rl_noeval.py
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


def get_evaluator(args: RLDataArguments):
    start_idx = 0
    end_idx = None if not DEBUG else 100

    evaluator: Evaluator
    if args.task == 'word_sort':
        evaluator =  WordSortingTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=1,
            shuffle=False,
            verbose=DEBUG
        )
    elif args.task == 'date_understanding':
        evaluator = DateUnderstandingTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=1,
            shuffle=False,
            verbose=DEBUG
        )
    elif args.task == 'multistep_arithmetic':
        evaluator = MultistepArithmeticTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=1,
            shuffle=False,
            verbose=DEBUG
        )
    elif args.task == 'logical_deduction':
        evaluator = LogicalDeductionTrainingEvaluator(
            train_file=args.train_world_dset,
            eval_start_idx=start_idx,
            eval_end_idx=end_idx,
            batch=1,
            shuffle=False,
            verbose=DEBUG
        )
    return evaluator


def get_tester(args):
    evaluator: Evaluator
    if args.task == 'multistep_arithmetic':
        evaluator = MultistepArithmeticEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None,
            batch=4,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'logical_deduction':
        evaluator = LogicalDeductionEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None,
            batch=4,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'word_sort':
        evaluator = WordSortingEvaluator(
            split="validation", 
            subtask='all', 
            eval_start_idx=0,
            eval_end_idx=None,
            batch=4,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'date_understanding':
        evaluator = DateUnderstandingEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None,
            batch=4,
            shuffle=False,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return evaluator


def get_wrapped_model_for_data(args):
    wrapped_model: GenerativeModel
    if args.task == 'multistep_arithmetic':
        # additional info should be added in the wrapper already
        wrapped_model = LLM_MultistepArithmetic(
            model_name=args.llm,
            verbose=False,
            temperature=1.0,  # since we need multiple iterations
            max_tokens=2048
        )
        improve_model = LLM_MultistepArithmetic(
            model_name=args.llm,
            verbose=False,
            temperature=0.0,  # since we need multiple iterations
            max_tokens=2048
        )
        assert(len(wrapped_model.additional_info.strip()) > 0)
        verifier_model = LLM_MultistepArithmetic_Feedback_wCorrect_Tabular(
			model_name=args.verifier_llm,
			verbose=False,
			temperature=0.0,
			max_tokens=2048,
		)
        llm_w_verifier = LLMwVerifier_MultistepArithmetic(
            init_ans_model=wrapped_model,
            improve_model=improve_model,
            verifier_model=verifier_model,
            verbose=True,
            is_eval=not args.return_if_correct,
        )
    elif args.task == 'logical_deduction':
        wrapped_model = LLM_LogicalDeduction(
            model_name=args.llm,
            verbose=False,
            temperature=1.0,
            max_tokens=2048
        )
        improve_model = LLM_LogicalDeduction(
            model_name=args.llm,
            verbose=False,
            temperature=0.0,
            max_tokens=2048
        )
        assert(len(wrapped_model.additional_info.strip()) > 0)
        verifier_model = LLM_LogicalDeduction_Feedback_wCorrect_Tabular(
			model_name=args.verifier_llm,
			verbose=False,
			temperature=0.0,
			max_tokens=2048,
		)
        llm_w_verifier = LLMwVerifier_LogicalDeduction(
            init_ans_model=wrapped_model,
            improve_model=improve_model,
            verifier_model=verifier_model,
            verbose=True,
            is_eval=not args.return_if_correct,  # False is used for Table 1. If we switch this to True, it will actually be worse
        )
    elif args.task == 'date_understanding':
        wrapped_model = LLM_DateUnderstanding(
            model_name=args.llm,
            verbose=False,
            temperature=1.0,
            max_tokens=2048
        )
        improve_model = LLM_DateUnderstanding(
            model_name=args.llm,
            verbose=False,
            temperature=0.0,
            max_tokens=2048
        )
        verifier_model = LLM_DateUnderstanding_Feedback_wCorrect_Tabular(
			model_name=args.verifier_llm,
			verbose=False,
			temperature=0.0,
			max_tokens=2048,
		)
        llm_w_verifier = LLMwVerifier_DateUnderstanding(
            init_ans_model=wrapped_model,
            improve_model=improve_model,
            verifier_model=verifier_model,
            verbose=True,
            is_eval=not args.return_if_correct,  # False is used for Table 1. If we switch this to True, it will actually be worse
        )
    elif args.task == 'word_sort':
        wrapped_model = LLM_WordSorting(
            model_name=args.llm,
            verbose=False,
            temperature=1.0,
            max_tokens=2048
        )
        improve_model = LLM_WordSorting(
            model_name=args.llm,
            verbose=False,
            temperature=0.0,
            max_tokens=2048
        )
        verifier_model = LLM_WordSorting_Feedback_wCorrect_Tabular(
			model_name=args.verifier_llm,
			verbose=False,
			temperature=0.0,
			max_tokens=2048,
		)
        llm_w_verifier = LLMwVerifier_WordSorting(
            init_ans_model=wrapped_model,
            improve_model=improve_model,
            verifier_model=verifier_model,
            verbose=True,
            is_eval=not args.return_if_correct,  # False is used for Table 1. If we switch this to True, it will actually be worse
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return llm_w_verifier


def evaluate(args, log_file_path):
    wrapped_model = get_wrapped_model_for_data(args)
    task_additional_info = wrapped_model.additional_info
    evaluator = get_evaluator(args)

    correct_choice, pred_choice, preds, result = evaluator.evaluate(wrapped_model)
    trajectory_logs = wrapped_model.logs

    jsonlines.open(log_file_path, "w").write_all(trajectory_logs)

    question_to_rationale = {}
    for data in evaluator.dataset:
        question = data['question'].strip()
        if 'formatted_choices' in data:
            formatted_choices = data['formatted_choices']
            question += '\nOptions:\n' + formatted_choices
        attempted_answer = data['rationale'].strip()
        question_to_rationale[question] = attempted_answer
    return task_additional_info, trajectory_logs, question_to_rationale


def __format_collected_data(task_additional_info, trajectory_logs, unique_data):
    set_seed(42)

    formatted_trajectories = []

    for k, v in trajectory_logs.items():
        question = k
        last_feedback = v[-1]['feedback']
        if "final response is also correct" not in last_feedback:
            continue  # wrong answer
        
        first_attempt = v[0]['llm_output']['attempted_answer']
        first_feedback = v[0]['feedback']

        if "Q:" in question:
            out_text = f"{question}\n"
        else:
            out_text = f"Q: {question}\n"
        out_text += f"Answer: Let's think step by step.{task_additional_info}\n{first_attempt}\n"
        out_text += f"Feedback: {first_feedback}\n"
    
        for attempt in v[1:]:
            new_answer = attempt['llm_output']['attempted_answer']
            feedback = attempt['feedback']

            out_text += f"Updated Answer: Let's think step by step.{task_additional_info}\n{new_answer}\n"
            out_text += f"Feedback: {feedback}\n"

        out_text = out_text.strip()
        formatted_trajectories.append(out_text)
    
    # edit the trajectories
    new_all_data = []
    num_duplicates = 0
    for data in formatted_trajectories:
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
    print(f"to process trajectories: {len(formatted_trajectories)}")
    print(f"num duplicates: {num_duplicates}")
    return new_all_data, unique_data


def __convert_n_balance_trajectories(all_data, data_args, task_additional_info):
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
                        Answer: Let's think step by step. {task_additional_info}
                        {att}
                        Feedback: {fb}
                        Updated Answer: Let's think step by step. {task_additional_info}
                        {imp}
                        """.replace("    ", "").replace(' \n', '\n').strip()
                        all_converted_data['improvement'].append({
                            "text": formatted_data,
                            "text_att_start_text": "\nFeedback:"
                        })
                    else:
                        formatted_data = f"""
                        {question}
                        Answer: Let's think step by step. {task_additional_info}
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
    # use 1:1:1.5 for non_improvement:gt:improvement
    num_improvement = len(all_converted_data['improvement'])
    get_num_non_improvement = math.ceil(num_improvement / data_args.improve_data_ratio)
    get_num_non_improvement = min(get_num_non_improvement, len(all_converted_data['non_improvement']))
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


def collect_data(training_args, data_args):
    global RUN_ID
    print(f"collecting data")

    all_new_data: List[dict] = []
    all_unique_data: set = set()
    all_update_stats = {}

    i = 0
    max_itr = data_args.max_data_itr
    task_additional_info = ''
    while len(all_new_data) < data_args.max_data_length and i < max_itr:
        set_seed(40 + i)
        # evaluate and save data, 42 is used in the evaluator file
        log_file_path = os.path.join(training_args.output_dir, f'eval_logs_{i}.jsonl')
        task_additional_info, trajectory_logs, question_to_rationale = evaluate(data_args, log_file_path)
        
        new_data, all_unique_data = __format_collected_data(task_additional_info, trajectory_logs, all_unique_data)
        new_data, update_stats = __convert_n_balance_trajectories(new_data, data_args, task_additional_info)
        all_new_data.extend(new_data)

        # update all_update_stats
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
        Answer: Let's think step by step. {task_additional_info}
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
    
    train_file_path = os.path.join(training_args.output_dir, f'train_data.jsonl')
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
    num_train_excluded = len(training_data) - len(train_dset)
    print(f"Excluded {num_train_excluded} training examples, loaded {len(train_dset)} examples")
    return train_dset, dummy_eval_dset


def _find_last_checkpoint(save_dir):
    for checkpoint_dir_name in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dir_name)
        if os.path.isdir(checkpoint_dir) and checkpoint_dir_name.startswith('checkpoint'):
            return checkpoint_dir
    raise ValueError(f"Cannot find checkpoint in {save_dir}")


@ray.remote(num_gpus=1)
def train(
        checkpoint_path, tokenizer, training_data,
        data_args: RLDataArguments, training_args: TrainingArguments):
    global RUN_ID
    set_seed(training_args.seed)

    train_dset, eval_dset = to_dataset(data_args, tokenizer, training_data)
    
    if training_args.deepspeed:
        # since this is in ray remote, we need to manually setup deepspeed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9998'
        os.environ['RANK'] = "0"
        os.environ['LOCAL_RANK'] = "0"
        os.environ['WORLD_SIZE'] = "1"
        deepspeed.init_distributed()
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path
    )

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


@ray.remote(num_gpus=1)
def test(checkpoint_path, tokenizer, data_args: RLDataArguments, training_args: RLTrainingArguments):
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
    eval_file_path = os.path.join(training_args.output_dir, f'test_data.pkl')
    with open(eval_file_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(json.dumps(result, indent=2))
    return


def init(model_args: ModelArguments, data_args: RLDataArguments, logger_args: LoggerArguments, training_args: RLTrainingArguments):
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


def main(model_args: ModelArguments, data_args: RLDataArguments, logger_args: LoggerArguments, training_args: RLTrainingArguments):
    set_seed(42)
    # initalize model
    checkpoint_path, tokenizer = init(model_args, data_args, logger_args, training_args)
    
    train_path = os.path.join(training_args.output_dir, f"train_data.jsonl")
    if os.path.exists(train_path):
        new_train_data = itr_0_load_data_if_exist(train_path)
    else:
        new_train_data = collect_data(training_args, data_args)
    
    if len(new_train_data) < data_args.min_data_length:
        print("Not enough training data, terminating")
        return
    
    # trainer changes the training args, so we need to 'deepcopy' it. However, for some reason doing deepcopy makes trainer.save not work
    training_args_copy = TrainingArguments(**training_args.to_dict())
    checkpoint_path = ray.get(train.remote(checkpoint_path, tokenizer, new_train_data, data_args, training_args_copy))

    ray.get(test.remote(checkpoint_path, tokenizer, data_args, training_args))
    return


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, RLDataArguments, LoggerArguments, RLTrainingArguments),
        description="TriPosT"
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, logger_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, logger_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
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
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict()
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    # train
    main(model_args, data_args, logger_args, training_args)