from datasets import load_dataset
from models.base import GenerativeModel, Evaluator
from tqdm.auto import tqdm
from typing import Union, List
from copy import deepcopy
import re
import math
import random
import jsonlines


class WordSortingEvaluator(Evaluator):
    def __init__(self, 
                 split="validation", subtask='all', eval_start_idx=0, eval_end_idx=None, 
                 batch=1, verbose=False, shuffle=False,
                 dset_raw_file="data/raw/word_sorting_scripted_rationales.jsonl"):
        self.split = split
        self.subtask = subtask
        self.eval_start_idx = eval_start_idx
        self.eval_end_idx = eval_end_idx
        self.batch = batch  # batch size
        self.verbose = verbose
        self.dset_raw_file = dset_raw_file
        self.dataset = self.prepare_data(shuffle=shuffle)
        return
    
    def prepare_data(self, shuffle=False):
        with jsonlines.open(self.dset_raw_file) as reader:
            dset_raw = list(reader)
        dataset = {}
        for d in dset_raw:
            split = d['meta_data']['new_split']  # use our split, which takes step > 2 in training to validation
            if split not in dataset:
                dataset[split] = []
            dataset[split].append(d)
        
        all_preprocessed_data = []
        ori_data = [d for d in dataset[self.split]]
        if shuffle:
            random.seed(42)
            random.shuffle(ori_data)
        for data in ori_data[self.eval_start_idx:self.eval_end_idx]:
            question = data['question'][2:].replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace("  ", " ").strip()
            answer = data['correct_answer']
            rationale = data['rationale']
            processed_rationale_data = {
                'question': question,
                'answer': answer,
                'rationale': rationale,
            }
            all_preprocessed_data.append(processed_rationale_data)
        print(f"Loaded {len(all_preprocessed_data)} data points from {self.split} split.")
        return all_preprocessed_data
    
    def process_answer(self, generated_output: Union[str, List[str]]):
        if isinstance(generated_output, list):
            return generated_output
        # step by step solutions
        try:
            last_step = generated_output.split('\n')[-1].strip()
            if 'Feedback:' in last_step:
                last_step = generated_output.split('\n')[-2].strip()
            final_sorting = re.findall(r'the answer is:(.*)\.', last_step)[0].strip()
            
            final_sorted_words = final_sorting.split()
            final_sorted_words = [word.strip() for word in final_sorted_words]
        except:
            final_sorted_words = []
        return final_sorted_words
    
    def report_results(self, correct_answers, pred_answers, noprint=False):
        result = {
            '2-3': {'total': 0, 'correct': 0},
            '4-7': {'total': 0, 'correct': 0},
            '8-10': {'total': 0, 'correct': 0},
            '11-15': {'total': 0, 'correct': 0},
            '15+': {'total': 0, 'correct': 0},
            'total': {'total': 0, 'correct': 0},
        }
        for correct_answer, pred_answer in zip(correct_answers, pred_answers):
            subtask = len(correct_answer)
            if subtask <= 3:
                subtask_str = '2-3'
            elif subtask <= 7:
                subtask_str = '4-7'
            elif subtask <= 10:
                subtask_str = '8-10'
            elif subtask <= 15:
                subtask_str = '11-15'
            else:
                subtask_str = '15+'
            result[subtask_str]['total'] += 1
            result['total']['total'] += 1
            if correct_answer == pred_answer:
                result[subtask_str]['correct'] += 1
                result['total']['correct'] += 1
        for subtask_str, res in result.items():
            acc = -1.0 if res['total'] == 0 else res['correct'] / res['total']
            result[subtask_str]['acc'] = acc
            if not noprint:
                print(f"Subtask {subtask_str}: {acc} out of {res['total']}")
        return result
    
    def report_short_results(self, correct_answers, pred_answers):
        full_result = self.report_results(correct_answers, pred_answers, noprint=True)
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
            correct_ordering = data['answer']
            num_words = len(correct_ordering)
            if self.subtask == '1-7' and num_words > 7:
                continue
            elif self.subtask == '1-10' and num_words > 10:
                continue
            elif self.subtask == '11-15' and (num_words > 15 or num_words < 11):
                continue
            elif self.subtask == '15+' and num_words <= 15:
                continue

            curr_data = {
                'input_data': data,
                'correct_ordering': correct_ordering,
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

        correct_answers = []
        pred_answers = []
        preds = []
        batched_data = self._get_batched_data()
        for batch in batched_data:
            if len(batch) == 1:
                input_data = batch[0]['input_data']
                correct_answer = [batch[0]['correct_ordering']]
            else:
                input_data = {
                    'batched_input': [data['input_data'] for data in batch],
                }
                correct_answer = [data['correct_ordering'] for data in batch]
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

            curr_res = self.report_short_results(correct_answers, pred_answers)
            pbar.set_postfix(curr_res)
        pbar.close()
        
        # check answers
        result = self.report_results(correct_answers, pred_answers)
        return correct_answers, pred_answers, preds, result


class WordSortingTrainingEvaluator(WordSortingEvaluator):
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
                'correct_answer': correct_answer,  # may be used by the feedback generator
                'correct_ordering': correct_answer  # used by the evaluator
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