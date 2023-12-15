from models.base import GenerativeModel, Evaluator
from tqdm.auto import tqdm
from copy import deepcopy

import math
import numpy as np
import random
import re
import jsonlines


class MultistepArithmeticEvaluator(Evaluator):
    def __init__(self, 
                 split="validation", subtask='all', eval_start_idx=0, eval_end_idx=None, 
                 batch=1, shuffle=False, verbose=False, 
                 dset_raw_file="data/raw/multistep_arithmetic_scripted_rationales.jsonl"):
        if split == 'test':
            split = 'validation'
            print("Warning: using validation split for test set evaluation.")
        self.split = split
        self.subtask = subtask
        self.eval_end_idx = eval_end_idx
        self.eval_start_idx = eval_start_idx
        self.batch = batch  # batch size
        self.verbose = verbose
        self.dset_raw_file = dset_raw_file
        self.dataset = self.prepare_data(shuffle=shuffle)
        return

    def _process_rationale(self, rationale, answer: str):        
        data = {
            'rationale': rationale.strip(),
            'answer': answer.strip(),
        }
        return data
    
    def _format_options(self, options: list):
        return '\n'.join(options)

    def prepare_data(self, shuffle=False):
        # read and organize data
        with jsonlines.open(self.dset_raw_file) as reader:
            dset_raw = list(reader)
        dataset = {}
        for d in dset_raw:
            split = d['meta_data']['new_split']  # use our split, which takes step > 2 in training to validation
            if split not in dataset:
                dataset[split] = []
            dataset[split].append(d)
        
        # process data (copy pasted code)
        all_preprocessed_data = []
        ori_data = [d for d in dataset[self.split]]
        if shuffle:
            random.seed(42)
            random.shuffle(ori_data)
        for data in ori_data[self.eval_start_idx:self.eval_end_idx]:
            # remove Q:
            question = data['question'][2:].replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace("  ", " ").strip()
            rationale = data['rationale']
            answer = data['correct_answer']

            processed_rationale_data = self._process_rationale(rationale, answer)
            processed_rationale_data['question'] = question
            processed_rationale_data['meta_data'] = data['meta_data']  # keep this for later
            all_preprocessed_data.append(processed_rationale_data)
        print(f"Loaded {len(all_preprocessed_data)} data points from {self.split} split.")
        return all_preprocessed_data
    
    # shared methods for all GSM8k evaluators
    def process_answer(self, generated_output: str):
        if isinstance(generated_output, float):
            return generated_output
        # step by step solutions
        try:
            final_answer = generated_output.split('\n')[-1].strip()
            if 'Feedback' in final_answer:
                final_answer = generated_output.split('\n')[-2]
            if not final_answer.endswith('.'):
                final_answer += '.'
            numeric_outputs = re.findall(r'the answer is (.*)\.', final_answer)
            if len(numeric_outputs) != 1:
                return np.inf  # just wrong
            numeric = float(numeric_outputs[0])
            return numeric
        except:
            return np.inf
        return
    
    def report_results(self, input_data, correct_rationales, correct_answers, pred_answers, noprint=False):
        result = {
            'l3-4d2': {'total': 0, 'correct': 0},
            'l3-4d22': {'total': 0, 'correct': 0},
            'l3-4d222': {'total': 0, 'correct': 0}, # unseen
            'l5-6d2': {'total': 0, 'correct': 0}, # unseen
            'l5-6d22': {'total': 0, 'correct': 0}, # unseen
            'l5-6d222': {'total': 0, 'correct': 0}, # unseen
            'total': {'total': 0, 'correct': 0},
        }
        for in_data, correct_rationale, correct_answer, pred_answer in zip(input_data, correct_rationales, correct_answers, pred_answers):
            meta_data = in_data['meta_data']
            depth = meta_data['depth_levels']
            depth_str = "".join([str(d) for d in depth])
            length = meta_data['length']
            if length < 5:
                subtask_str = f'l3-4d{depth_str}'
            else:
                subtask_str = f'l5-6d{depth_str}'
            
            result[subtask_str]['total'] += 1
            result['total']['total'] += 1
            if float(correct_answer) == float(pred_answer):
                result[subtask_str]['correct'] += 1
                result['total']['correct'] += 1
        for subtask_str, res in result.items():
            acc = -1.0 if res['total'] == 0 else res['correct'] / res['total']
            result[subtask_str]['acc'] = acc
            if not noprint:
                print(f"Subtask {subtask_str}: {acc} out of {res['total']}")
        return result
    
    def report_short_results(self, input_data, correct_rationales, correct_answers, pred_answers):
        full_result = self.report_results(input_data, correct_rationales, correct_answers, pred_answers, noprint=True)
        # remove some keys to make it more readable
        _curr_res = deepcopy(full_result)
        for k, v in full_result.items():
            for kk, vv in v.items():
                if kk != 'acc':
                    del _curr_res[k][kk]
                else:
                    _curr_res[k][kk] = round(vv, 2)
        return _curr_res
    
    def _get_batched_data(self):
        total_ = len(self.dataset)

        out_batches = []
        curr_batch = []
        for data in self.dataset:
            if total_ == 0:
                break
            # evaluate
            meta_data = data['meta_data']
            depth = meta_data['depth_levels']
            length = meta_data['length']
            if self.subtask == 'l3-4d2-22' and (length > 4 or depth not in [[2], [2, 2]]):
                continue
            elif self.subtask == 'l3-4' and length > 4:
                continue
            elif self.subtask == 'l5-6' and (length < 5 or length > 6):
                continue
            
            correct_rationale = data['rationale']
            correct_answer = data['answer']
            curr_data = {
                'input_data': data,
                'correct_rationale': correct_rationale,
                'correct_answer': correct_answer
            }
            curr_batch.append(curr_data)
            total_ -= 1
            if len(curr_batch) == self.batch:
                out_batches.append(curr_batch)
                curr_batch = []
        if len(curr_batch) > 0:
            out_batches.append(curr_batch)
        return out_batches

    def evaluate(self, model: GenerativeModel):
        total_ = len(self.dataset)
        batched_total = math.ceil(total_ / self.batch)
        pbar = tqdm(total=batched_total, desc="Evaluating")

        all_input_data = []  # hyperbaton needs the question to split hard/easy tasks
        correct_answers = []
        correct_rationales = []
        pred_answers = []
        preds = []
        batched_data = self._get_batched_data()
        for batch in batched_data:
            if len(batch) == 1:
                input_data = batch[0]['input_data']
                _input_data = [input_data]
                correct_rationale = [batch[0]['correct_rationale']]
                correct_answer = [batch[0]['correct_answer'].lower()]
            else:
                input_data = {
                    'batched_input': [data['input_data'] for data in batch],
                }
                _input_data = input_data['batched_input']
                correct_rationale = [data['correct_rationale'] for data in batch]
                correct_answer = [data['correct_answer'].lower() for data in batch]
            all_input_data.extend(_input_data)
            correct_rationales.extend(correct_rationale)
            correct_answers.extend(correct_answer)
            
            out = model.generate(input_data)
            if isinstance(out, list):
                preds.extend(out)
            else:
                preds.append(out)
            
            if len(batch) > 1:
                answer = []
                for o in out:
                    answer_i = self.process_answer(o)
                    answer.append(answer_i)
                pred_answers.extend(answer)
            else:
                if isinstance(out, list):
                    out = out[0]
                answer = self.process_answer(out)
                pred_answers.append(answer)
                answer = [answer]  # for printing
            

            if self.verbose:
                print(f"Input: {input_data}")
                print(f"Raw Output: {out}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Predicted Answer: {answer}")
                print()
            
            pbar.update(1)

            curr_res = self.report_short_results(all_input_data, correct_rationales, correct_answers, pred_answers)
            pbar.set_postfix(curr_res)
        pbar.close()
        
        # check answers
        result = self.report_results(all_input_data, correct_rationales, correct_answers, pred_answers)
        return correct_answers, pred_answers, preds, result
    

class MultistepArithmeticTrainingEvaluator(MultistepArithmeticEvaluator):
    def __init__(self, train_file, eval_start_idx=0, eval_end_idx=None, batch=1, shuffle=False, verbose=False):
        self.train_file = train_file
        self.eval_end_idx = eval_end_idx
        self.eval_start_idx = eval_start_idx
        self.batch = batch  # batch size
        self.verbose = verbose
        self.dataset = self.prepare_data(shuffle=shuffle)
        return
    
    def _get_batched_data(self):
        out_batches = []
        curr_batch = []
        for data in self.dataset:
            correct_rationale = data['rationale']
            correct_answer = data['answer']
            curr_data = {
                'input_data': data,
                'correct_rationale': correct_rationale,
                'correct_answer': correct_answer
            }
            curr_batch.append(curr_data)
            if len(curr_batch) == self.batch:
                out_batches.append(curr_batch)
                curr_batch = []
        if len(curr_batch) > 0:
            out_batches.append(curr_batch)
        return out_batches

    def prepare_data(self, shuffle=False):
        all_preprocessed_data = []
        with jsonlines.open(self.train_file) as reader:
            all_data = list(reader)
        all_preprocessed_data = [data['meta_data'] for data in all_data]
        all_preprocessed_data = all_preprocessed_data[self.eval_start_idx:self.eval_end_idx]
        if shuffle:
            random.seed(42)
            random.shuffle(all_preprocessed_data)
        return all_preprocessed_data