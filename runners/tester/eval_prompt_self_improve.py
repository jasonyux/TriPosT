from models.wrappers import (
    GenerativeModel,
    LLM_MultistepArithmetic, LLM_LogicalDeduction,
    LLM_MultistepArithmetic_Feedback_wCorrect_Tabular,
    LLM_LogicalDeduction_Feedback_wCorrect_Tabular,
)
from models.base import Evaluator
from models.evaluation.multistep_arithmetic import MultistepArithmeticEvaluator
from models.evaluation.logical_deduction import LogicalDeductionEvaluator
from models.rl.multistep_arithmetic import LLMwVerifier_MultistepArithmetic
from models.rl.logical_deduction import LLMwVerifier_LogicalDeduction
from transformers import set_seed
import pickle


def get_evaluator(args):
    evaluator: Evaluator
    if args.task == 'multistep_arithmetic':
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
    if args.task == 'multistep_arithmetic':
        # additional info should be added in the wrapper already
        wrapped_model = LLM_MultistepArithmetic(
            model_name=args.model_name_or_path,
            verbose=args.verbose,
            temperature=1e-5,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
        )
        assert(len(wrapped_model.additional_info.strip()) > 0)
        verifier_model = LLM_MultistepArithmetic_Feedback_wCorrect_Tabular(
            model_name=args.model_name_or_path,
            verbose=args.verbose,
            temperature=1e-5,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )
        llm_w_verifier = LLMwVerifier_MultistepArithmetic(
            init_ans_model=wrapped_model,
            improve_model=wrapped_model,
            verifier_model=verifier_model,
            verbose=True,
            is_eval=not args.return_if_correct,
        )
    elif args.task == 'logical_deduction':
        wrapped_model = LLM_LogicalDeduction(
            model_name=args.model_name_or_path,
            verbose=args.verbose,
            temperature=1e-5,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
        )
        assert(len(wrapped_model.additional_info.strip()) > 0)
        verifier_model = LLM_LogicalDeduction_Feedback_wCorrect_Tabular(
            model_name=args.model_name_or_path,
            verbose=args.verbose,
            temperature=1e-5,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1
        )
        llm_w_verifier = LLMwVerifier_LogicalDeduction(
            init_ans_model=wrapped_model,
            improve_model=wrapped_model,
            verifier_model=verifier_model,
            verbose=True,
            is_eval=not args.return_if_correct,  # False is used for Table 1. If we switch this to True, it will actually be worse
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return llm_w_verifier

def main(args):
    set_seed(42)

    evaluator = get_evaluator(args)

    print(f"Using LLM {args.model_name_or_path}")
    model = get_wrapped_model(args)

    # just prompt to evaluate
    output_data = {}
    print("evaluating WITH a verifier/self-improving module")
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
    # python runners/tester/eval_prompt_self_improve.py \
    # -o model_checkpoints/multistep_arithmetic/baselines/prompt/blablabla_perf.pkl \
    # --model_name_or_path model_checkpoints/multistep_arithmetic/blablabla/checkpoint-120 \
    # --task multistep_arithmetic \
    # --return_if_correct \
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
        "--model_name_or_path",
        type=str,
        default="model_checkpoints/debug",
        help='huggingface checkpoint path'
    )
    parser.add_argument(
        "--return_if_correct",
        action='store_true',
        help='Whether ask the feedback to return "answer is correct..." immediately when the predicted answer is the same as gold.'
    )
    parser.add_argument(
        "--task",
        type=str,
        default='multistep_arithmetic',
        choices=['multistep_arithmetic', 'logical_deduction'],
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help='Whether to print out feedback'
    )
    args = parser.parse_args()

    main(args)