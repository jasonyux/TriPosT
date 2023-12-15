from models.wrappers import (
    GenerativeModel,
    LLM_WordSorting, LLM_DateUnderstanding, 
    LLM_MultistepArithmetic, LLM_LogicalDeduction
)
from models.base import Evaluator
from models.evaluation.word_sorting import WordSortingEvaluator
from models.evaluation.date_understanding import DateUnderstandingEvaluator
from models.evaluation.multistep_arithmetic import MultistepArithmeticEvaluator
from models.evaluation.logical_deduction import LogicalDeductionEvaluator
from transformers import set_seed
import pickle


def get_evaluator(args):
    evaluator: Evaluator
    if args.task == 'word_sort':
        evaluator = WordSortingEvaluator(
            split="validation", 
            subtask='all', 
            eval_start_idx=0,
            eval_end_idx=None,
            batch=1,
            shuffle=False,
            verbose=True
        )
    elif args.task == 'date_understanding':
        evaluator = DateUnderstandingEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None,
            batch=1,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'multistep_arithmetic':
        evaluator = MultistepArithmeticEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None,
            batch=1,
            shuffle=False,
            verbose=False
        )
    elif args.task == 'logical_deduction':
        evaluator = LogicalDeductionEvaluator(
            split="validation",
            subtask="all",
            eval_start_idx=0,
            eval_end_idx=None,
            batch=1,
            shuffle=False,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return evaluator


def get_wrapped_model(args):
    wrapped_model: GenerativeModel
    if args.task == 'word_sort':
        wrapped_model = LLM_WordSorting(
            model_name=args.llm,
            verbose=args.verbose,
            temperature=0.0,
            max_tokens=2048
        )
    elif args.task == 'date_understanding':
        wrapped_model = LLM_DateUnderstanding(
            model_name=args.llm,
            verbose=args.verbose,
            temperature=0.0,
            max_tokens=2048
        )
    elif args.task == 'multistep_arithmetic':
        # additional info should be added in the wrapper already
        wrapped_model = LLM_MultistepArithmetic(
            model_name=args.llm,
            verbose=args.verbose,
            temperature=0.0,
            max_tokens=2048
        )
        assert(len(wrapped_model.additional_info.strip()) > 0)
    elif args.task == 'logical_deduction':
        wrapped_model = LLM_LogicalDeduction(
            model_name=args.llm,
            verbose=args.verbose,
            temperature=0.0,
            max_tokens=2048
        )
        assert(len(wrapped_model.additional_info.strip()) > 0)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return wrapped_model

def main(args):
    set_seed(42)

    evaluator = get_evaluator(args)

    print(f"Using LLM {args.llm}")
    model = get_wrapped_model(args)

    # just prompt to evaluate
    output_data = {}
    print("evaluating WITHOUT a verifier/self-improving module")
    correct_choice, pred_choice, preds, result = evaluator.evaluate(model)
    if hasattr(model, 'logs'):
        output_data['logs'] = model.logs

    output_data['correct_choice'] = correct_choice
    output_data['pred_choice'] = pred_choice
    output_data['preds'] = preds
    with open(args.o, 'wb') as f:
        pickle.dump(output_data, f)


if __name__ == "__main__":
    # example
    # python runners/tester/eval_llm.py \
    # -o model_checkpoints/multistep_arithmetic/baselines/llm/codex_perf.pkl \
    # --task multistep_arithmetic \
    # --verbose
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help='Destination to save output file'
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="code-davinci-002",
        help='LLM model name'
    )
    parser.add_argument(
        "--task",
        type=str,
        default='multistep_arithmetic',
        choices=['word_sort', 'date_understanding', 'multistep_arithmetic', 'logical_deduction'],
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help='Whether to print out feedback'
    )
    args = parser.parse_args()

    main(args)