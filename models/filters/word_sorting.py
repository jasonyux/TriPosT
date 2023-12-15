from models.filters.base import ParsedAttempt, ParsedFeedback
from dateutil.parser import parse
from utils.constants import CHARACTERS
from utils.utils import print_red
import re


class ParsedAttempt_WordSorting(ParsedAttempt):
    def made_progress(self, prev_answer:str, prev_feedback:ParsedFeedback):
        attempted_answer = self.attempt_str.strip()
        if attempted_answer == prev_answer.strip():
            return False
        
        # check if steps other than the feedback is changed
        error_step, error_part = prev_feedback.parse_feedback()
        prev_steps = prev_answer.split("\n")
        if error_step == 'final':
            # last step
            error_step_int = len(prev_steps)
        else:
            error_step_int = 0
            for step in prev_steps:
                error_step_int += 1
                if step.startswith(f'({error_step})'):
                    break
        
        new_steps = attempted_answer.split("\n")
        for i in range(error_step_int):
            # the error step should be changed
            prev_step = prev_steps[i].lower().strip()
            new_step = new_steps[i].lower().strip()
            if i == error_step_int - 1:
                if prev_step == new_step:
                    print_red(f"Error not changed. Feedback: {prev_feedback.feedback_str}")
                    print_red(f"error {error_step_int}, but prev step {prev_step} and new step {new_step}")
                    return False
                # for checking error port, there is a special case
                if 'is duplicated' in prev_feedback.feedback_str:
                    if new_step.lower().count(error_part.lower()) > 1:
                        print_red(f"Error part still in the step (duplicate). Feedback: {prev_feedback.feedback_str}")
                        print_red(f"error {error_step_int}, but prev step {prev_step} and new step {new_step}")
                        return False
                else:
                    if error_part.lower() in new_step.lower():
                        print_red(f"Error part still in the step. Feedback: {prev_feedback.feedback_str}")
                        print_red(f"error {error_step_int}, but prev step {prev_step} and new step {new_step}")
                        return False
            elif prev_step != new_step:
                # since now we prompt with previous attempts, this should never be the case
                print_red(f"[BUG] Prev steps changed. Feedback is {prev_feedback.feedback_str}")
                print_red(f"error {error_step_int}, but prev step {prev_step} and new step {new_step}")
                return False
        return True
    
    def has_valid_content(self):
        # the best way should be to use the scripted evaluator
        # but this is unavailable for non-scripted methods. Hence here we write simple checks
        all_steps = self.attempt_str.split("\n")
        for step in all_steps:
            step = step.strip()
            if "Hence" in step:
                if "?" in step:
                    return False
        return True
    

class ParsedFeedback_WordSorting(ParsedFeedback):
    def parse_feedback(self):
        found_error_steps = re.findall(r"In step \((\d+[\.\d+]*)\) the part", self.feedback_str)
        if len(found_error_steps) == 0 and 'In step (Final response) the part' not in self.feedback_str:
            return None, None
        if 'In step (Final response) the part' in self.feedback_str:
            # when comparing, the text 'final' is acutally larger than any number
            error_step = 'final'
        else:
            error_step = found_error_steps[0]
        found_error_parts = re.findall(r'the part "(.*)" is incorrect', self.feedback_str)
        if len(found_error_parts) == 0:
            return error_step, None
        error_part = found_error_parts[0].strip()
        return error_step, error_part
    
    def has_valid_content(self, feedbacked_attempt: str) -> bool:
        return True