from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from runners.trainer.train_self_improve import (
    get_wrapped_model,
    WordSortingEvaluator
)
from typing import Any
import jsonlines
import argparse


def is_pred_correct(correct_ans: Any, pred_ans: Any, task_output: str):
	if task_output == 'number':
		is_correct = float(correct_ans) == float(pred_ans)
	elif task_output == 'choice':
		is_correct = correct_ans.lower().strip() == pred_ans.lower().strip()
	elif task_output == 'generic':
		is_correct = correct_ans == pred_ans
	else:
		raise NotImplementedError
	return is_correct


def get_lmsi_data(wrapped_model, evaluator, ori_path: str, save_path: str, task_type, debug=False):
    # increase temperature and sample
    temperature = 1.2  # used by LMSI
    wrapped_model.gen_kwargs = {"temperature": temperature, "do_sample": True}

    ## to ensure the two dataset covers the same questions
    original_training_questions = set()
    with jsonlines.open(ori_path, 'r') as reader:
        original_train_data = list(reader)
        for d in original_train_data:
            original_training_questions.add(d["meta_data"]["question"].strip())

    ## collect data
    collected_data = {}
    num_iterations = 2 if debug else 5
    for itr in range(num_iterations):
        print(f"Iteration {itr}")
        correct_choice, pred_choice, preds, _ = evaluator.evaluate(wrapped_model)

        # get the questions back
        for i in range(len(preds)):
            pred = preds[i]
            pred_ans = pred_choice[i]
            correct = correct_choice[i]
            is_correct = is_pred_correct(correct, pred_ans, task_type)
            if is_correct:
                orig_data = evaluator.dataset[i]
                assert is_pred_correct(orig_data["answer"], correct, task_type)
                orig_question = orig_data["question"].strip()
                if orig_question not in original_training_questions:
                    continue
                
                reformatted_data = f"""
                Q: {orig_question}
                {pred.strip()}
                """.replace(" "*4, "").strip()
                if orig_question not in collected_data:
                    collected_data[orig_question] = set()
                collected_data[orig_question].add(reformatted_data)

                ## LMSI augmentation
                last_step = pred.split("\n")[-1].strip()
                answer_snippet = last_step.split("(Final response)")[-1].strip()
                augmented_data = f"""
                Q: {orig_question}
                Answer: {answer_snippet}
                """.replace(" "*4, "").strip()
                collected_data[orig_question].add(augmented_data)
        
        # save data
        print(f"Saving data to {save_path}")
        linearized_data = []
        for data in collected_data.values():
            for d in data:
                linearized_data.append({
                    "text": d,
                    "text_att_start_text": "\nAnswer:"
                })
            
        with jsonlines.open(save_path, mode="w") as writer:
            writer.write_all(linearized_data)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="model_checkpoints/word_sort/baselines/ws_llama7b_rationale_1-7_5epoch_allsamples_s0/checkpoint-400",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode"
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_ckpt,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>"
    )

    set_seed(42)

    data_args = argparse.Namespace(
        task = 'word_sort',
        eval_model_wrapper_cls = 'rationale'
    )

    evaluator = WordSortingEvaluator(
        split="train",
        subtask="all",
        eval_start_idx=0,
        eval_end_idx=40 if args.debug else None,
        batch=4,
        shuffle=False,
        verbose=False,
    )

    wrapped_model = get_wrapped_model(model, tokenizer, data_args)

    get_lmsi_data(
        wrapped_model,
        evaluator,
        ori_path="data/training/word_sorting/word_sorting_baseline_rationale_1-7.jsonl",
        save_path="data/training/word_sorting/word_sorting_LMSI_rationale_1-7.jsonl",
        task_type="generic",
        debug=args.debug
    )