from prompts.date_understanding import *
from runners.utils import chain_run_wrapper
from typing import List
from utils.utils import print_red, print_green
from models.rl.base import LLMEditor
from models.filters.date_understanding import ParsedAttempt_DateUnderstanding, ParsedFeedback_DateUnderstanding
from models.self_improve.date_understanding import LLMwVerifier_DateUnderstanding
import re


class LLMEditor_DateUnderstanding(LLMwVerifier_DateUnderstanding, LLMEditor):
    FB_ERROR  = '[ERROR]'
    FB_NOEDIT = '[NOEDIT]'

    def __init__(
            self,
            improve_model,
            verifier_model,
            max_updates=5,
            verbose=False,
            save_log=True):
        super().__init__(
            None,
            improve_model,
            verifier_model,
            tokenizer=None,
            max_updates=max_updates,
            verbose=False,
            save_log=save_log
        )
        self.verbose = verbose
        self.update_stats = {
            'num_updated': 0,
            'num_updated_correct': 0,
            'num_original': 0,
        }
        self.log = []
        return
    
    def _check_attempted_answer(self, llm_output, prev_answer, prev_feedback):
        attempted_answer = llm_output['attempted_answer']
        parsed_attempt = ParsedAttempt_DateUnderstanding(attempted_answer)
        parsed_prev_feedback = ParsedFeedback_DateUnderstanding(prev_feedback)
        if not parsed_attempt.is_parsable():
            if self.verbose:
                print_red(f'attempted answer is not parsable: {attempted_answer}')
            return False
        if not parsed_attempt.made_progress(prev_answer, parsed_prev_feedback):
            if self.verbose:
                print_red(f'attempted answer did not make progress: {attempted_answer}')
            return False
        # # since we are improving, being invalid here is okay
        # attempted_answer_w_question = llm_output['question'] + '\n' + attempted_answer
        # if not parsed_attempt_w_question.has_valid_content():
        #     if self.verbose:
        #         print_red(f'attempted answer has invalid content: {attempted_answer}')
        #     return False
        return True

    def get_filtered_verifier_feedback(self, question, attempt, correct_answer):        
        # if the answer is correct, the feedback is fixed
        last_step = attempt.split('\n')[-1]
        choice_outputs = re.findall(r'the answer is \(([a-zA-Z])\)', last_step)
        if len(choice_outputs) != 1:
            return "[ERROR]"
        final_chioce = choice_outputs[0]
        
        num_steps = len(attempt.split('\n')) - 1
        if correct_answer.lower() == final_chioce.lower():
            return f"Step (1) to step ({num_steps}) are correct. The final response is also correct."

        # otherwise, generate feedback
        llm_input = {
            'question': question,
            'attempted_answer': attempt,
            'correct_answer': correct_answer,
        }
        output = self.verifier_model.generate(llm_input, out_dict=True)
        generated_feedback = output['feedback']
        parsed_feedback = ParsedFeedback_DateUnderstanding(generated_feedback)
        if not parsed_feedback.is_parsable():
            if self.verbose:
                print_red(f'invalid format: {generated_feedback}')
            return "[ERROR]"
        if not parsed_feedback.has_valid_content(attempt):
            if self.verbose:
                print_red(f'invalid content: {generated_feedback}')
            return "[ERROR]"
        return generated_feedback

    def get_llm_feedbacks(self, question: str, attempts: List[str], correct_ans: float):
        llm_feedbacks = []
        for attempt in attempts:
            attempt = attempt.strip()
            feedback = self.get_filtered_verifier_feedback(question, attempt, str(correct_ans))
            llm_feedbacks.append(feedback)
        return llm_feedbacks

    def get_to_edit_feedback(self, attempts: List[str], gen_feedbacks: List[str], llm_feedbacks: List[str], correct_ans: float):
        for gen_attempt, gen_feedback, llm_feedback in zip(attempts, gen_feedbacks, llm_feedbacks):
            if self.FB_ERROR in llm_feedback:
                return None, self.FB_ERROR
            # check if gen_feedback is just wrong.
            is_llm_feedback_valid = False
            is_gen_feedback_valid = False
            
            parsed_llm_feedback = ParsedFeedback_DateUnderstanding(llm_feedback)
            parsed_gen_feedback = ParsedFeedback_DateUnderstanding(gen_feedback)

            if parsed_llm_feedback.is_parsable() and parsed_llm_feedback.has_valid_content(gen_attempt):
                is_llm_feedback_valid = True
            if parsed_gen_feedback.is_parsable() and parsed_gen_feedback.has_valid_content(gen_attempt):
                is_gen_feedback_valid = True
            
            if not is_llm_feedback_valid:
                if self.verbose:
                    print_red(f'llm_feedback is not parsable or has invalid content: {llm_feedback}')
                return None, self.FB_ERROR
            elif is_llm_feedback_valid and not is_gen_feedback_valid:
                return gen_feedback, llm_feedback
            # if both valid
            # check if both feedback are talking about 1) the same error step, 2) error part
            if not parsed_llm_feedback.is_identical(parsed_gen_feedback):
                return gen_feedback, llm_feedback
        # everything is the same
        return gen_feedback, self.FB_NOEDIT

    def get_llm_answer_until_done(self, llm_prompt, correct_ans: float):
        llm_output = {}
        updated = 0
        done = self.check_if_done(llm_prompt['feedback'])
        # question, generated attmept, and LLM feedback
        trajectory = [llm_prompt['question'], llm_prompt['attempted_answer'], llm_prompt['feedback']]
        bad_improvement = False
        while not done and updated < self.max_updates:
            prev_feedback = llm_prompt['feedback']
            prev_answer = llm_prompt['attempted_answer']
            parsed_prev_feedback = ParsedFeedback_DateUnderstanding(prev_feedback)
            prev_partial_answer = self._get_prev_partial_answer(prev_answer, parsed_prev_feedback)
            llm_prompt['prev_partial_answer'] = '\n' + prev_partial_answer

            llm_output = chain_run_wrapper(self.chain_get_update, llm_prompt)
            attempted_answer = prev_partial_answer + llm_output.pop('updated_answer').strip()
            parsed_attempted_answer = ParsedAttempt_DateUnderstanding(attempted_answer)
            llm_output['attempted_answer'] = parsed_attempted_answer.clean_attempt()
            llm_output['correct_answer'] = correct_ans

            # check feedback
            legit_update = self._check_attempted_answer(llm_output, prev_answer, prev_feedback)

            if not legit_update:
                if self.verbose:
                    print_red(f"LLM output is a not legit update based on previous answers and feedback.")
                    print_red(f"Prev answer: {prev_answer}")
                    print_red(f"Prev feedback: {prev_feedback}")
                    print_red(f"LLM output: {llm_output['attempted_answer']}")
                break

            question = llm_output['question']
            attempt = llm_output['attempted_answer']
            correct_answer = llm_output['correct_answer']
            attempted_answer_w_question = question + '\n' + attempted_answer
            parsed_attempt_w_question = ParsedAttempt_DateUnderstanding(attempted_answer_w_question)
            feedback = self.get_filtered_verifier_feedback(question, attempt, correct_answer)
            if self.verbose:
                print("Curr attempt:", attempt)
                print("Feedback:", feedback)
            
            if "ERROR" in feedback:
                break
            done = self.check_if_done(feedback) and parsed_attempt_w_question.has_valid_content()

            parsed_feedback = ParsedFeedback_DateUnderstanding(feedback)
            parsed_prev_feedback = ParsedFeedback_DateUnderstanding(prev_feedback)
            if not parsed_feedback.made_progress(parsed_prev_feedback):
                if self.verbose:
                    print_red("Feedback progress not made.")
                    print_red(f"Prev feedback: {prev_feedback}")
                    print_red(f"New feedback: {feedback}")
                # bad data
                bad_improvement = True
            
            updated += 1
            llm_prompt = {
                **llm_output,
                "feedback": feedback,
                "examples": EXAMPLES_DATE_UNDERSTANDING_UPDATE  # EXAMPLES_UPDATE
            }
            trajectory.append(llm_output['attempted_answer'])
            trajectory.append(feedback)
        # llm did not make it
        if 'final response is also correct' not in trajectory[-1].lower():
            if self.verbose:
                print_red(f"LLM did not make it. {trajectory}")
            return None
        if bad_improvement:
            # just keep the first feedback
            trajectory = trajectory[:3]
            if self.verbose:
                print_red(f"Bad improvement, returning just the first step. {trajectory}")
        return trajectory
    
    def format_trajectory(self, trajectory: List[str]):
        out_text = ""
        question, init_attempt, init_feedback = trajectory[0], trajectory[1], trajectory[2]
        cleaned_feedback = self._post_process_feedback(init_feedback)
        out_text += f"Q: {question}\n"
        out_text += f"Answer: Let's think step by step.\n{init_attempt}\n"
        out_text += f"Feedback: {cleaned_feedback}\n"

        is_attempt = True
        for step in trajectory[3:]:
            step = step.strip()
            if is_attempt:
                out_text += f"Updated Answer: Let's think step by step.\n{step}\n"
            else:
                cleaned_feedback = self._post_process_feedback(step).strip()
                out_text += f"Feedback: {cleaned_feedback}\n"
            is_attempt = not is_attempt
        return out_text.strip()

    def _edit_single(self, pred: str, pred_ans: float, correct_ans: float):        
        question = pred.split('\n')[0]
        if 'Q:' not in question:
            print_red(f"Question not found. {question}")
            return None
        question = question.replace('Q:', '').strip()
        # a multi-choice question
        if 'options' in pred.split('\n')[1].lower():
            option_texts = pred.split('\n')[1:]
            for option_text in option_texts:
                if 'answer:' in option_text.lower():
                    break
                question += '\n' + option_text
            question = question.strip()
        
        attempts, gen_feedbacks = self._split_by_feedback(pred)
        # check if there are any attmept that is too long
        for att in attempts:
            num_words = len(att.split(' '))
            if num_words > 200:
                if self.verbose:
                    print_red(f"Too long attempt: {att}")
                return None
        llm_feedbacks = self.get_llm_feedbacks(question, attempts, correct_ans)
        to_edit_feedback, new_feedback = self.get_to_edit_feedback(attempts, gen_feedbacks, llm_feedbacks, correct_ans)
        if new_feedback == self.FB_ERROR:
            return None
        elif new_feedback == self.FB_NOEDIT:
            if pred_ans != correct_ans:
                return None
            # correct
            return pred + '\n' + to_edit_feedback
        
        # edit
        to_feedback_attempt = None
        ok_trajectory = []
        for attempt, feedback in zip(attempts, gen_feedbacks):
            ok_trajectory.append(attempt)
            if feedback == to_edit_feedback:
                to_feedback_attempt = attempt
                break
            ok_trajectory.append(feedback)
        if self.verbose:
            print("changing feedback:", to_edit_feedback)
            print("to_feedback_attempt:", new_feedback)
        llm_prompt = {
            "examples": EXAMPLES_DATE_UNDERSTANDING_UPDATE,  # EXAMPLES_UPDATE
            "question": question,
            "attempted_answer": to_feedback_attempt,
            "feedback": new_feedback,
        }
        llm_fixed_trajectory = self.get_llm_answer_until_done(llm_prompt, correct_ans)
        if llm_fixed_trajectory is None:
            return None

        # final trajectory is to merge the llm trajectory with the original trajectory
        question = llm_fixed_trajectory.pop(0)
        new_trajectory = [question, *ok_trajectory, *llm_fixed_trajectory[1:]]
        print_green(f"Merging {ok_trajectory} with {llm_fixed_trajectory[1:]}")
        return self.format_trajectory(new_trajectory)

    def edit(self, preds: List[str], pred_ans: List[float], correct_ans: List[float]):
        edited_traj = []
        for p, p_ans, c_ans in zip(preds, pred_ans, correct_ans):
            try:
                # this cleaning will be especially useful when the model just start to learn how to self improve
                p_ori = p
                p = self.clean_pred(p)
                if self.verbose:
                    print("Original pred:", p_ori)
                    print("Cleaned pred:", p)
                    print('\n\n\n')
                if p is None:
                    self.log.append({'original': p, 'edited': 'FORMAT ERROR'})
                    if self.verbose:
                        print_red("FORMAT ERROR")
                    continue
                if p_ans == c_ans and ParsedAttempt_DateUnderstanding(p).has_valid_content():
                    ori_p = p
                    p = self._add_last_feedback(p)
                    if p is None:
                        self.log.append({'original': ori_p, 'edited': 'CANNOT ADD LAST FEEDBACK'})
                        continue
                    edited_traj.append(p)
                    self.update_stats['num_original'] += 1
                    self.log.append({'original': p, 'edited': 'CORRECT'})
                    if self.verbose:
                        print_green("CORRECT")
                    continue
                self.update_stats['num_updated'] += 1
                
                traj = self._edit_single(p, p_ans, c_ans)
                if traj is not None:
                    if self.verbose:
                        print("Original:\n", p)
                        print("Edited:\n", traj)
                        print('\n\n\n')
                    edited_traj.append(traj)
                    self.update_stats['num_updated_correct'] += 1
                self.log.append({'original': p, 'edited': 'none' if traj is None else traj})
            except Exception as e:
                print("Exception:", e)
                print_red(f"Pred: {p}")
                print_red(f"Correct ans: {c_ans}")
                print('\n\n\n')
                self.log.append({'original': p, 'edited': 'EXCEPTION'})
            if self.verbose:
                print('\n\n\n')
        return edited_traj