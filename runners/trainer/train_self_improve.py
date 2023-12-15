from transformers import (
    default_data_collator, set_seed,
    Trainer, TrainingArguments,
    HfArgumentParser,
)
from accelerate.utils import DistributedType
from utils.dataset import SelfImproveDataset
from models.wrappers import (
    GenerativeModel, 
    GPT_WordSorting, GPT_DateUnderstanding,
    GPT_MultistepArithmetic, GPT_LogicalDeduction
)
from models.base import Evaluator
from models.evaluation.word_sorting import WordSortingEvaluator
from models.evaluation.date_understanding import DateUnderstandingEvaluator
from models.evaluation.multistep_arithmetic import MultistepArithmeticEvaluator
from models.evaluation.logical_deduction import LogicalDeductionEvaluator
from models.self_improve.word_sorting import SelfImprove_GPT_WordSorting
from models.self_improve.date_understanding import SelfImprove_GPT_DateUnderstanding
from models.self_improve.multistep_arithmetic import SelfImprove_GPT_MultistepArithmetic
from models.self_improve.logical_deduction import SelfImprove_GPT_LogicalDeduction
from dataclasses import dataclass, field
from typing import Union

import transformers
import torch
import jsonlines
import json
import torch
import os
import sys
import wandb
import pickle
import shutil
os.environ['WANDB_PROJECT'] = 'llm_companion'


class LossMaskedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_mask = inputs.pop('loss_mask')
        weights = inputs.pop('weights')

        _, outputs = super().compute_loss(model, inputs, True)

        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()
        shift_loss_mask = loss_mask[..., 1:].float().contiguous()
        shift_weights = weights[..., 1:].float().contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = (loss_ * shift_loss_mask.view(-1) * shift_weights.view(-1)).sum() / shift_loss_mask.sum()

        return (loss, outputs) if return_outputs else loss


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    Arguments overriding some default TrainingArguments
    """
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."}
    )
    num_train_epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The peak learning rate for the scheduler."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."}
    )
    warmup_ratio: float = field(
        default=0.2,
        metadata={"help": "Ratio of warmup steps to total steps."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay to apply to the optimizer."}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report to wandb or not"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every X updates steps."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training."}
    )
    eval_steps: int = field(
        default=200,
        metadata={"help": "Run an evaluation every X steps."}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "The metric to use to compare two different models."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy to adopt during training."}
    )
    save_steps: int = field(
        default=200,
        metadata={"help": "Save checkpoint every X steps."}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir."}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load the best model found during training at the end of training."}
    )
    seed: int = field(
        default=42,
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/opt-iml-1.3b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dset_type: str = field(
        default="word_sorting",
        metadata={"help": "The type of dataset to use (logical_deduction, word_sorting)"}
    )
    train_dset: str = field(
        # default="data/training/word_sorting/ws_self_improve.jsonl",  # please do supply, in case accidens happen
        default='',
        metadata={"help": "Path to training dataset"}
    )
    eval_dset: str = field(
        # default="data/validation/word_sorting/ws_self_improve_val.jsonl",
        default='',
        metadata={"help": "Path to vallidation dataset"}
    )
    min_input_length: int = field(
        default=2048,
        metadata={"help": "minimum seq length for tokenization"}
    )
    max_input_length: int = field(
        default=2048,
        metadata={"help": "maximum seq length for tokenization"}
    )
    mask_before_att_start_text: bool = field(
        default=False,
        metadata={"help": "Whether to mask the text before the attention start token (i.e. not learn them)"}
    )
    end_data_idx: Union[int, None] = field(
        default=None,
        metadata={"help": "The index of the last data point to use"}
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the data"}
    )
    task: str = field(
        default="word_sort",
        metadata={"help": "The class to use for the eval evaluator"},
    )
    eval_model_wrapper_cls: str = field(
        default="self-improve",
        metadata={"help": "The class to use for the eval model wrapper"},
    )
    convert_to_turns: bool = field(
        default=False,
        metadata={"help": "Whether to convert trajectory data to individual turns."},
    )

    def __post_init__(self):
        if self.train_dset == '' or self.eval_dset == '':
            raise ValueError("Need both a training/validation file.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_dset  == '':
            extension = self.train_dset.split(".")[-1]
            assert extension in valid_extensions, \
                "`train_file` should be a jsonlines file."
        if self.eval_dset  == '':
            extension = self.eval_dset.split(".")[-1]
            assert extension in valid_extensions, \
                    "`eval_dset` should be a jsonlines file."
        # specific to how I named the training dset files
        if self.eval_model_wrapper_cls == 'self-improve':
            assert 'self_improve' in self.train_dset, \
                "self-improve model wrapper requires self_improve training dset"
        elif self.eval_model_wrapper_cls == 'rationale':
            assert 'rationale' in self.train_dset, \
                "rationale model wrapper requires rationale training dset"
        elif self.eval_model_wrapper_cls == 'ao':
            assert 'answeronly' in self.train_dset or 'ao_' in self.train_dset, \
                "answeronly model wrapper requires answeronly training dset"
        else:
            raise ValueError(f"Unknown eval_model_wrapper_cls: {self.eval_model_wrapper_cls}")
        
        if self.task == 'word_sort':
            assert('ws_' in self.train_dset or 'word_sorting_' in self.train_dset)
        elif self.task == 'date_understanding':
            assert('date_understanding_' in self.train_dset)
        elif self.task == 'multistep_arithmetic':
            assert('multistep_arithmetic_' in self.train_dset)
        elif self.task == "logical_deduction":
            assert('logical_deduction_' in self.train_dset)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return


@dataclass
class LoggerArguments:
    """
    Arguments pertaining to using wandb for logging
    """

    run_group: str = field(
        default="debug",
        metadata={"help": "wandb run group"}
    )


def to_dataset(args: DataArguments, tokenizer):
    train_file_path = args.train_dset
    eval_file_path = args.eval_dset
    dset_type = args.dset_type
    
    with jsonlines.open(train_file_path) as reader:
        training_data = list(reader)
    with jsonlines.open(eval_file_path) as reader:
        eval_data = list(reader)
    
    if dset_type in [
            'word_sorting', 'logical_deduction',
            'date_understanding', 'multistep_arithmetic',
        ]:
        train_dset = SelfImproveDataset(
            training_data, tokenizer,
            end_data_idx=args.end_data_idx,
            min_input_length=args.min_input_length, max_input_length=args.max_input_length,
            mask_before_att_start_text=args.mask_before_att_start_text,
            shuffle=args.shuffle
        )
        eval_dset = SelfImproveDataset(
            eval_data, tokenizer,
            end_data_idx=args.end_data_idx,
            min_input_length=args.min_input_length, max_input_length=args.max_input_length,
            mask_before_att_start_text=args.mask_before_att_start_text
        )
    else:
        raise ValueError(f"Unknown dset_type: {dset_type}")
    num_train_excluded = len(training_data) - len(train_dset)
    num_eval_excluded = len(eval_data) - len(eval_dset)
    print(f"Excluded {num_train_excluded} training examples, loaded {len(train_dset)} examples")
    print(f"Excluded {num_eval_excluded} eval examples, loaded {len(eval_dset)} examples")
    return train_dset, eval_dset

def get_evaluator(args: DataArguments):
    evaluator: Evaluator
    if args.task == 'word_sort':
        evaluator = WordSortingEvaluator(
            split="validation", 
            subtask='all', 
            eval_start_idx=0,
            eval_end_idx=None,
            batch=4,
            shuffle=False,
            verbose=True
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
    elif args.task == 'multistep_arithmetic':
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
    return evaluator


def get_wrapped_model(model, tokenizer, args: DataArguments):
    wrapped_model: GenerativeModel
    if args.task == 'word_sort':
        if args.eval_model_wrapper_cls == 'self-improve':
            wrapped_model = SelfImprove_GPT_WordSorting(model, tokenizer, manual_prompt=args.convert_to_turns)
        elif args.eval_model_wrapper_cls == 'ao':
            wrapped_model = GPT_WordSorting(model, tokenizer, additional_info='', input_max_length=128, max_new_tokens=128)
        else:
            wrapped_model = GPT_WordSorting(model, tokenizer, additional_info=" Let's think step by step.\n", input_max_length=256, max_new_tokens=1024)
    elif args.task == 'date_understanding':
        if args.eval_model_wrapper_cls == 'self-improve':
            wrapped_model = SelfImprove_GPT_DateUnderstanding(model, tokenizer, manual_prompt=args.convert_to_turns)
        elif args.eval_model_wrapper_cls == 'ao':
            wrapped_model = GPT_DateUnderstanding(model, tokenizer, additional_info='', input_max_length=128, max_new_tokens=10)
        else:
            wrapped_model = GPT_DateUnderstanding(model, tokenizer, additional_info=" Let's think step by step.\n", input_max_length=256, max_new_tokens=512)
    elif args.task == 'multistep_arithmetic':
        additional_info = (
            'Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). '
            'So, remember to always compute the expressions inside parentheses or brackets first.'
        )
        if args.eval_model_wrapper_cls == 'self-improve':
            wrapped_model = SelfImprove_GPT_MultistepArithmetic(model, tokenizer, manual_prompt=args.convert_to_turns, additional_info=f" Let's think step by step. {additional_info}")
        elif args.eval_model_wrapper_cls == 'ao':
            wrapped_model = GPT_MultistepArithmetic(model, tokenizer, additional_info='', input_max_length=256, max_new_tokens=10)
        else:
            wrapped_model = GPT_MultistepArithmetic(model, tokenizer, additional_info=f" Let's think step by step. {additional_info}\n", input_max_length=256, max_new_tokens=1024)
    elif args.task == 'logical_deduction':
        additional_info = 'Let "??" represents 0 or more objects, and "?" represents exactly 1 object.'
        if args.eval_model_wrapper_cls == 'self-improve':
            wrapped_model = SelfImprove_GPT_LogicalDeduction(model, tokenizer, manual_prompt=args.convert_to_turns, additional_info=f" Let's think step by step. {additional_info}")
        elif args.eval_model_wrapper_cls == 'ao':
            wrapped_model = GPT_LogicalDeduction(model, tokenizer, additional_info='', input_max_length=256, max_new_tokens=10)
        else:
            wrapped_model = GPT_LogicalDeduction(model, tokenizer, additional_info=f" Let's think step by step. {additional_info}\n", input_max_length=256, max_new_tokens=1024)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return wrapped_model


def _remove_optimizer_weights(save_dir):
    for checkpoint_dirs in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dirs)
        if os.path.isdir(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith('global_step'):
                    optimizer_dir = os.path.join(checkpoint_dir, file)
                    # remove the entire folder. This is used by deepspeed to store optimizer states
                    print('removing', optimizer_dir)
                    shutil.rmtree(optimizer_dir)
                elif file.startswith("optimizer.pt"):
                    optimizer_file = os.path.join(checkpoint_dir, file)
                    print('removing', optimizer_file)
                    os.remove(optimizer_file)
    return


def main(model_args: ModelArguments, data_args: DataArguments, logger_args: LoggerArguments, training_args: MyTrainingArguments):
    set_seed(training_args.seed)

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
        tokenizer.pad_token_id = tokenizer.eos_token_id  # eos_token_id or unk_token_id
    print(f'using {tokenizer.pad_token=}, {tokenizer.pad_token_id=}')
    print(f'{tokenizer.all_special_tokens=}')
    
    def model_init():
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name
        )

    train_dset, eval_dset = to_dataset(data_args, tokenizer)

    # if it is already initialized, huggingface will use it
    all_args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'logger_args': vars(logger_args),
        'training_args': training_args.to_dict()
    }
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    if 'wandb' in training_args.report_to:
        run = wandb.init(
            project=os.environ['WANDB_PROJECT'],
            name=training_args.output_dir.split("/")[-1] or None,
            group=logger_args.run_group,
            config=all_args,
        )
    trainer = LossMaskedTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    # evaluate and save data, 42 is used in the evaluator file
    set_seed(42)
    evaluator = get_evaluator(data_args)

    model = trainer.model
    wrapped_model = get_wrapped_model(model, tokenizer, data_args)

    correct_choice, pred_choice, preds, result = evaluator.evaluate(wrapped_model)
    if 'wandb' in training_args.report_to:
        wandb.log(result)
    
    output_data = {
        'logs': getattr(wrapped_model, 'logs', None),
        'correct_choice': correct_choice,
        'pred_choice': pred_choice,
        'preds': preds,
    }
    eval_file_path = os.path.join(training_args.output_dir, 'eval_data.pkl')
    with open(eval_file_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(json.dumps(result, indent=2))

    _remove_optimizer_weights(training_args.output_dir)
    wandb.finish()
    return


if __name__ == '__main__':
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, DataArguments, LoggerArguments, MyTrainingArguments),
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