from models.filters.base import ParsedTrajectory
from models.base import GenerativeModel
from models.wrappers import GPT_QA, TEMPLATE_MODEL_QA_INIT_ANSWER, TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER
from typing import List, Dict
from utils.utils import findall

import re
import nltk
import torch


TEMPLATE_SELFIMPROVE_BASE_FEEDBACK = """
{history}
Feedback:
""".strip()


TEMPLATE_SELFIMPROVE_BASE_UPDATE = """
{history}
Updated Answer: Let's think step by step.
""".strip()


class LLMwVerifier(GenerativeModel):
    additional_info = ""
    
    """helper methods common for using LLM as verifier (feedback generator)
    """
    def _format_verifier_output(self, verifier_output):
        return verifier_output
    
    def _check_attempted_answer(self, attempted_answer, prev_answer):
        if attempted_answer.strip() == prev_answer.strip():
            return False
        steps = attempted_answer.split("\n")
        last_step = steps[-1].strip()
        if not last_step.lower().startswith('(final response)'):
            return False
        if 'the answer is' not in last_step:
            return False
        for step in steps:
            # started a new question by itself
            if 'Q:' in step:
                return False
        return True
    
    def clean_feedback(self, text:str):
        # avoid sent tokenizing them
        # for logical deduction
        text = text.replace(' ?? ', '<dqmark>')
        text = text.replace(' ? ', '<qmark>')

        # clean a feedback
        sentences = nltk.sent_tokenize(text)
        out_sentences = sentences[:1]
        for sent in sentences[1:]:
            if not sent.endswith('.'):
                continue
            # if 'is incorrect' in sent:
            #     out_sentences.append(sent)
            # elif 'because' in sent.lower():
            #     out_sentences.append(sent)
            # elif 'should be' in sent.lower():
            #     out_sentences.append(sent)
            out_sentences.append(sent)

        # replace the special tokens back
        out = ' '.join(out_sentences)

        # for logical deduction
        out = out.replace('<dqmark>', ' ?? ')
        out = out.replace('<qmark>', ' ? ')
        return out

    def _post_process_feedback(self, feedback):
        if 'final response is also correct' in feedback.lower():
            return feedback
        # clean up the feedback
        feedback = self.clean_feedback(feedback)
        # check if it starts looking at more than in step (x). " In step (4) "
        # removed: sometimes in the explaination we WILL mention other prior steps
        # found_locations = []
        # for i in findall('in step (', feedback.lower()):
        #     found_locations.append(i)
        # if len(found_locations) > 1:
        #     feedback = feedback[:found_locations[1]]
        return feedback.strip()
    
    def _check_feedback_format_for_bad_rationale(self, feedback):
        length = len(feedback.split('\n'))
        if length > 1:
            return False
        if 'is incorrect' not in feedback:
            return False  # it should be wrong
        if 'because' not in feedback:
            return False
        return True
    
    def check_if_done(self, feedback):
        return "final response is also correct" in feedback.lower() or "[ERROR]" in feedback

    def _save_log(self, log_key, data):
        if log_key not in self.logs:
            self.logs[log_key] = []
        self.logs[log_key].append(data)
        return
    
    def _to_log_key(self, input_data):
        question = input_data['question'].strip()
        formatted = f"""
        Q: {question}
        """.replace('    ', '').strip()
        if formatted in self.logs:
            print(formatted)
            raise ValueError("Duplicate question")
        return formatted


class SelfImprove_GPT_QA(GPT_QA):
    ACTION_CONTINUE_GENERATION = "continue_generation"
    ACTION_DONE = "done"

    def __init__(self,
            model, tokenizer, 
            manual_prompt=False, input_max_length=1024, max_new_tokens=1024, 
            additional_info=" Let's think step by step.", gen_kwargs={}):
        super().__init__(model, tokenizer, input_max_length, max_new_tokens)
        self.manual_prompt = manual_prompt  # wheher if it generates an entire sequence until the end or need to manually prompt it
        self.additional_info = additional_info
        self.num_attempts = 10
        self.num_continue_generation = 3
        self.logs = []
        self.gen_kwargs = gen_kwargs
        return
    
    def _is_step_legit(self, step:str):
        # true if legit
        lower_step = step.strip().lower()
        if 'Options'.lower() in lower_step:
            return True
        if 'List:'.lower() in lower_step:
            return True
        if re.search(r"^\([a-zA-Z]\)", lower_step) is not None:  # option (A), (B), ...
            return True
        if 'Feedback'.lower() in lower_step:
            return True
        if 'Answer'.lower() in lower_step:
            return True
        if 'Updated Answer'.lower() in lower_step:
            return True
        if re.search(r"^\(\d+[\.\d+]*\)", lower_step) is not None:  # step number
            return True
        if 'Final response'.lower() in lower_step:
            return True
        return False
    
    def _get_step_id(self, step:str):
        # true if legit
        if not self._is_step_legit(step):
            return None
        
        lower_step = step.strip().lower()
        if 'Options'.lower() in lower_step:
            return 'options'
        if 'List:'.lower() in lower_step:
            return 'list'
        if re.search(r"^\([a-zA-Z]\)", lower_step) is not None:  # option (A), (B), ...
            return re.search(r"^\([a-zA-Z]\)", lower_step).group(0)
        if 'Feedback'.lower() in lower_step:
            return 'feedback'
        if 'Answer'.lower() in lower_step:
            return 'answer'
        if 'Updated Answer'.lower() in lower_step:
            return 'updated_answer'
        if re.search(r"^\(\d+[\.\d+]*\)", lower_step) is not None:  # step number
            return re.search(r"^\(\d+[\.\d+]*\)", lower_step).group(0)
        if 'Final response'.lower() in lower_step:
            return 'final_response'
        # should not happen
        print("Unknown step in _get_step_id", step)
        return None
    
    def _format_task_output(self, output_text):
        output_text = output_text.strip()
        # remove new questions it generated
        steps = output_text.split("\n")
        cleaned_steps = steps[:1]  # first step is question
        prev_step_id = 'question'
        for step in steps[1:]:
            # started a new question by itself
            # llama generates this at the end sometimes
            step = step.replace("⁇", "").strip()
            if not self._is_step_legit(step):
                break
            # remove repeating steps
            step_id = self._get_step_id(step)
            if step_id is None or step_id == prev_step_id:
                break
            prev_step_id = step_id
            cleaned_steps.append(step)
        cleaned_response = "\n".join(cleaned_steps)
        return cleaned_response
    
    def _generate(self, input_text):
        self.tokenizer.truncation_side = "left"
        input_ids = self.tokenizer(
            input_text, 
            max_length=self.input_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to("cuda")

        model_output = self.model.generate(
            input_ids,
            **self.gen_kwargs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            early_stopping=True,
            num_return_sequences=1
        )
        decoded_model_output = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        finished_generation = model_output[0][-1] == 2  # EOS token for llama
        cleaned_model_output = self._format_task_output(decoded_model_output)
        return cleaned_model_output, finished_generation
    
    # def _fixed_last_padding_id(self, input_ids, attention_mask):
    #     # ignore everything before non_pad_token_start
    #     non_pad_token_start = 0
    #     for i, i_id in enumerate(input_ids):
    #         if i_id != self.tokenizer.pad_token_id:
    #             non_pad_token_start = i
    #             break
    #     # ignore everything after non_pad_token_end + 1
    #     first_eos_after_non_pad_token = len(input_ids) - 1
    #     for i, i_id in enumerate(input_ids):
    #         if i < non_pad_token_start:
    #             continue
    #         if i_id == self.tokenizer.eos_token_id:
    #             first_eos_after_non_pad_token = i
    #             break
    #     attention_mask[:non_pad_token_start] = False
    #     attention_mask[first_eos_after_non_pad_token+1:] = False
    #     return attention_mask
    
    # def _fixed_padding(self, encoded_input):
    #     batched_input_ids = encoded_input["input_ids"]
    #     attention_mask = torch.ones_like(batched_input_ids, dtype=torch.bool)

    #     fixed_attention_masks = []
    #     for i in range(len(batched_input_ids)):
    #         input_ids = batched_input_ids[i]
    #         att_mask = attention_mask[i]
    #         fixed_att_mask = self._fixed_last_padding_id(input_ids, att_mask)
    #         fixed_attention_masks.append(fixed_att_mask)

    #     encoded_input['attention_mask'] = torch.stack(fixed_attention_masks)
    #     return encoded_input
    
    def _batch_generate(self, input_texts: List[str]):
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        encoded_input = self.tokenizer(
            input_texts, 
            max_length=self.input_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        if 'token_type_ids' in encoded_input:
            encoded_input.pop('token_type_ids')
        # encoded_input = self._fixed_padding(encoded_input)  # when we pad left with eos, this is needed
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to(self.model.device)

        model_output = self.model.generate(
            **encoded_input,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            early_stopping=True,
        )
        decoded_model_output = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        generated_eos = []
        for i, o in enumerate(model_output):
            if o[-1] in [0,1,2]:  # EOS, BOS, UNK token for llama
                generated_eos.append(True)
            else:
                generated_eos.append(False)
        cleaned_model_output = []
        for decoded_model_output_ in decoded_model_output:
            cleaned_model_output.append(self._format_task_output(decoded_model_output_))
        return cleaned_model_output, generated_eos
    
    def _made_progress(self, old_answer, new_answer):
        if old_answer == '':
            return True
        old_answer_last_step = old_answer.split('\n')[-1]
        new_answer_last_step = new_answer.split('\n')[-1]
        # still on the same step
        if old_answer_last_step in new_answer_last_step:
            return False
        return True
    
    def _format_next_step(self, new_answer: str, input_data: dict):
        # assumes new_answer is cleaned response
        parsed_steps = ParsedTrajectory(new_answer)
        if new_answer.startswith("Q:"):
            parse_trajectory = parsed_steps.parse_trajectory()
        else:
            question = input_data['question']
            parse_trajectory = parsed_steps.parse_trajectory_wo_question(question)
        question = parse_trajectory[0]
        last_action = parse_trajectory[-1].strip()
        # if feedback is the last one, then consider updating it
        if 'Feedback' in last_action:
            last_attempt = parse_trajectory[-2].strip()
            new_history = f"""
            {question}
            Answer: Let's think step by step.
            {last_attempt}
            Feedback: {last_action}
            Updated Answer: Let's think step by step.
            """.replace("    ", "").strip()
        else:
            new_history = f"""
            {question}
            Answer: Let's think step by step.
            {last_action}
            Feedback:
            """.replace("    ", "").strip()
        return new_history
    
    def _is_repeating_itself(self, line):
        # if it is solving over 15 steps then probably it is just repeating itself
        found_step_num = re.search(r"^\((\d+)(\.\d+)*\)", line)
        if found_step_num is not None and len(found_step_num.groups()) > 0:
            step_num = found_step_num.groups()[0]
            if step_num.isnumeric() and int(step_num) > 15:
                return True
        
        # see https://stackoverflow.com/questions/29481088/how-can-i-tell-if-a-string-repeats-itself-in-python
        last_50_char = line[-50:]
        i = (last_50_char+last_50_char).find(last_50_char, 1, -1)
        if i != -1:
            return True

        # the above takes care of the case when line contains no space but is repeating itself
        # when there is space, here is a simple one. 
        # if the last five words are all the same, then it's repeating itself
        words = last_50_char.split()
        if len(words) < 5:
            return True
        if len(set(words[-5:])) == 1:
            return True
        return False
    
    def _can_manual_prompt(self, last_step):
        if 'Feedback' in last_step:
            return True
        if '(final response)' in last_step.lower():
            return True
        return False
    
    def get_next_action(self, input_data, attempted_answer, new_answer, has_eos):
        # assumes output_answer is cleaned response, or a list of sorted words
        if isinstance(new_answer, list):
            data = {
                'raw_answer': new_answer,
                'final_answer': new_answer,
                'has_eos': has_eos,
                'next_action': self.ACTION_DONE,
                'info': 'direct_answer'
            }
            return self.ACTION_DONE, data
        
        last_step = new_answer.split('\n')[-1]
        if not self._made_progress(attempted_answer, new_answer):
            next_action = self.ACTION_DONE
            data = {
                'raw_answer': new_answer,
                'final_answer': new_answer,
                'has_eos': has_eos,
                'next_action': self.ACTION_DONE,
                'info': 'no_progress'
            }
        elif self._is_repeating_itself(last_step):
            next_action = self.ACTION_DONE
            data = {
                'raw_answer': new_answer,
                'final_answer': new_answer,
                'has_eos': has_eos,
                'next_action': self.ACTION_DONE,
                'info': 'repeating_itself'
            }
        elif 'final response is also correct'.lower() in last_step.lower():
            next_action = self.ACTION_DONE
            output_steps = new_answer.split('\n')[:-1]
            data = {
                'raw_answer': new_answer,
                'final_answer': '\n'.join(output_steps).strip(),
                'has_eos': has_eos,
                'next_action': self.ACTION_DONE,
                'info': 'final_response_is_also_correct'
            }
        elif has_eos and self.manual_prompt and self._can_manual_prompt(last_step):
            # then we need to manually convert data to Updated Answer, etc.
            # and we know the last step is not 'final response is also correct'
            next_action = self.ACTION_CONTINUE_GENERATION
            data = {
                'raw_answer': new_answer,
                'attempted_answer': new_answer,
                'has_eos': has_eos,
                'history': self._format_next_step(new_answer, input_data),
                'next_action': self.ACTION_CONTINUE_GENERATION,
                'info': 'manual_prompt'
            }
        else:
            # continue to generate
            next_action = self.ACTION_CONTINUE_GENERATION
            data = {
                'raw_answer': new_answer,
                'attempted_answer': new_answer,
                'has_eos': has_eos,
                'history': new_answer,
                'next_action': self.ACTION_CONTINUE_GENERATION,
                'info': 'continue_generation'
            }
        return next_action, data

    def get_next_action_batched(self, input_datas, input_texts, tasks, attempted_answers, new_answers, has_eoses):
        next_actions = []
        for i, input_text in enumerate(input_texts):
            input_data = input_datas[i]
            attempted_answer = attempted_answers[i]
            new_answer = new_answers[i]
            has_eos = has_eoses[i]
            next_action, data = self.get_next_action(input_data, attempted_answer, new_answer, has_eos)
            next_actions.append(next_action)
            tasks[input_text] = data
        return next_actions, tasks
    
    def _continue_generation(self, input_data, data, output_answer):
        input_text = data['history']
        new_output_answer, finished = self._generate(input_text)
        return new_output_answer, finished
    
    def _get_question(self, input_data):
        return input_data['question'].strip()
    
    def generate_until_done(self, input_data, data, input_text, attempted_answer):
        attempt = 0
        num_continue_generation = 1
        __curr_log = []
        next_action = self.ACTION_CONTINUE_GENERATION
        while next_action != self.ACTION_DONE:
            __curr_log.append({
                'raw_answer': attempted_answer,
                'input_text': input_text,
                'output_answer': attempted_answer,
                'next_action': next_action
            })
            if attempt == self.num_attempts or num_continue_generation == self.num_continue_generation:
                break
            
            if next_action == self.ACTION_CONTINUE_GENERATION:
                new_answer, finished = self._continue_generation(input_data, data, attempted_answer)
                attempt -= 1  # don't count this as an attempt
            else:
                raise ValueError(f"Unknown action {next_action}")
            prev_action = next_action
            next_action, data = self.get_next_action(input_data, attempted_answer, new_answer, finished)
            # continue generate one response at most three times
            if prev_action == self.ACTION_CONTINUE_GENERATION and next_action == self.ACTION_CONTINUE_GENERATION:
                num_continue_generation += 1
            else:
                num_continue_generation = 1  # reset
            
            attempted_answer = new_answer
            attempt += 1
        
        __curr_log.append({
            'input_text': input_text,
            'raw_answer': attempted_answer,
            'output_answer': attempted_answer,
            'next_action': next_action
        })
        return data, __curr_log

    def generate_until_done_batched(self, input_datas, input_texts, tasks):
        meta_data = {}
        next_actions = []
        for input_text in input_texts:
            # intiliaze meta data
            meta_data[input_text] = {
                'attempt': 0,
                'num_continue_generation': 1,
            }
            # init
            next_action = tasks[input_text]['next_action']
            next_actions.append(next_action)

        # init logs
        gen_logs = [[] for _ in input_texts]
        for i, t in enumerate(input_texts):
            next_action = tasks[t]['next_action']
            if 'final_answer' in tasks[t]:
                output_answer = tasks[t]['final_answer']
            else:
                output_answer = tasks[t]['attempted_answer']
            gen_logs[i].append({
                'input_text': t,
                'raw_answer': tasks[t]['raw_answer'],
                'info': tasks[t]['info'],
                'output_answer': output_answer,
                'next_action': next_action
            })
        # start generating
        while any([next_action != self.ACTION_DONE for next_action in next_actions]):
            # gather which tasks need to continue
            new_model_input_texts = []
            continue_gen_idx = []
            for i, t in enumerate(input_texts):
                if next_actions[i] == self.ACTION_DONE:
                    continue
                if tasks[t]['next_action'] == self.ACTION_CONTINUE_GENERATION:
                    new_model_input_texts.append(tasks[t]['history'])
                    meta_data[t]['attempt'] -= 1  # don't count this as an attempt
                continue_gen_idx.append(i)
            
            # generate for all tasks
            new_answers, finished = self._batch_generate(new_model_input_texts)
            
            # update next_actions
            all_attempted_answers = [] 
            all_has_eoses = []
            for t in input_texts:
                if 'final_answer' in tasks[t]:
                    all_attempted_answers.append(tasks[t]['final_answer'])
                else:
                    all_attempted_answers.append(tasks[t]['attempted_answer'])
                all_has_eoses.append(tasks[t]['has_eos'])
            all_new_answers = [a for a in all_attempted_answers]
            all_new_has_eoses = [a for a in all_has_eoses]
            for i, idx in enumerate(continue_gen_idx):
                all_new_answers[idx] = new_answers[i]
                all_new_has_eoses[idx] = finished[i]
            prev_actions = next_actions
            # this will also update tasks
            next_actions, tasks= self.get_next_action_batched(
                input_datas, input_texts, tasks, 
                all_attempted_answers, all_new_answers, all_new_has_eoses
            )
            
            # update meta data and update gen_logs
            for idx in continue_gen_idx:
                prev_act = prev_actions[idx]
                next_act = next_actions[idx]
                t = input_texts[idx]
                if prev_act == self.ACTION_CONTINUE_GENERATION and next_act == self.ACTION_CONTINUE_GENERATION:
                    meta_data[t]['num_continue_generation'] += 1
                else:
                    meta_data[t]['num_continue_generation'] = 1
                meta_data[t]['attempt'] += 1
                # force termination
                if meta_data[t]['attempt'] >= self.num_attempts or meta_data[t]['num_continue_generation'] >= self.num_continue_generation:
                    next_actions[idx] = self.ACTION_DONE
                    tasks[t]['final_answer'] = tasks[t]['attempted_answer']
                # update gen_logs, if done the log before function return will be used
                if 'final_answer' in tasks[t]:
                    output_answer = tasks[t]['final_answer']
                else:
                    output_answer = tasks[t]['attempted_answer']
                gen_logs[idx].append({
                    'input_text': t,
                    'raw_answer': tasks[t]['raw_answer'],
                    'info': tasks[t]['info'],
                    'output_answer': output_answer,
                    'next_action': next_actions[idx]
                })
        return tasks, gen_logs
    
    def _prepare_input(self, input_data: dict):
        question = self._get_question(input_data)
        if 'formatted_choices' in input_data:
            formatted_choices = input_data['formatted_choices']
            input_text = TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER.format(question=question, formatted_choices=formatted_choices, additional_info=self.additional_info)
        else:
            input_text = TEMPLATE_MODEL_QA_INIT_ANSWER.format(question=question, additional_info=self.additional_info)
        return input_text

    def generate(self, input_data: dict, **gen_kwargs):
        tasks: Dict[str, dict] = {}
        if 'batched_input' in input_data:
            input_texts = []
            for input_data_ in input_data['batched_input']:
                input_text_ = self._prepare_input(input_data_)
                input_texts.append(input_text_)
                tasks[input_text_] = {}
            assert(len(input_texts) == len(tasks))
        else:
            # TODO: legacy code that evaluator sometimes input a single string
            input_text = self._prepare_input(input_data)
            input_texts = [input_text]
            tasks[input_text] = {}
            input_data['batched_input'] = [input_data]
        batched_input_data = input_data['batched_input']

        attempted_answers, has_eoses = self._batch_generate(input_texts)
        # actions are updated into the tasks
        _, tasks = self.get_next_action_batched(batched_input_data, input_texts, tasks, ['' for _ in attempted_answers], attempted_answers, has_eoses)

        tasks, gen_logs = self.generate_until_done_batched(batched_input_data, input_texts, tasks)
        
        self.logs.extend(gen_logs)
        out_answers = []
        for input_t in input_texts:
            if 'final_answer' in tasks[input_t]:
                out_answers.append(tasks[input_t]['final_answer'])
            else:
                out_answers.append(tasks[input_t]['attempted_answer'])
        return out_answers


class SelfImprove_woFeedback_GPT_QA(SelfImprove_GPT_QA):
    def __init__(self, model, tokenizer, input_max_length=1024, max_new_tokens=1024):
        super().__init__(model, tokenizer, input_max_length, max_new_tokens)
        encoded_end_token = tokenizer.encode("Updated")
        assert(len(encoded_end_token) == 2)
        self.end_token_id = encoded_end_token[-1]
        raise NotImplementedError("This class has not been tested yet.")
    
    def _format_task_output(self, output_text):
        output_text = super()._format_task_output(output_text).strip()
        # remove the last Feedback line if there is, or the updated answer line
        output_steps = output_text.split('\n')
        last_step = output_steps[-1].lower()
        # while 'Feedback:'.lower() in last_step or 'Updated Answer:'.lower() in last_step:
        #     output_steps = output_steps[:-1]
        #     last_step = output_steps[-1].lower()
        if 'Feedback:'.lower() in last_step:
            output_steps = output_steps[:-1]
        attempt_only_text = '\n'.join(output_steps).strip()
        return attempt_only_text
    
    def get_next_action(self, input_data, attempted_answer, new_answer):
        # assumes output_answer is cleaned response, or a list of sorted words
        if isinstance(new_answer, list):
            data = {
                'final_answer': new_answer
            }
            return self.ACTION_DONE, data
        
        last_step = new_answer.split('\n')[-1]
        if not self._made_progress(attempted_answer, new_answer):
            next_action = self.ACTION_DONE
            data = {
                'final_answer': attempted_answer
            }
        elif 'Feedback:'.lower() in last_step.lower():
            raise ValueError("Should not have feedback")
        elif 'Updated Answer:'.lower() in last_step.lower():
            raise ValueError("Should not have updated answer")
        else:
            # continue to generate
            next_action = self.ACTION_CONTINUE_GENERATION
            data = {
                'history': new_answer
            }
        return next_action, data
    
    def _generate(self, input_text):
        self.tokenizer.truncation_side = "left"
        input_ids = self.tokenizer(
            input_text, 
            max_length=self.input_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to("cuda")

        model_output = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            early_stopping=True,
            num_return_sequences=1,
            eos_token_id = [self.tokenizer.eos_token_id, self.end_token_id],
        )
        decoded_model_output = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        cleaned_model_output = self._format_task_output(decoded_model_output)
        return cleaned_model_output