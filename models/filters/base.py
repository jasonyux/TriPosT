from abc import ABC, abstractmethod
from typing import List
from utils.utils import print_red
import re


class ParsedFeedback(ABC):
    def __init__(self, feedback_str: str):
        self.feedback_str = feedback_str
    
    @abstractmethod
    def has_valid_content(self, feedbacked_attempt: str) -> bool:
        raise NotImplementedError
    
    def parse_feedback(self):
        found_error_steps = re.findall(r"In step \((\d+)\) the part", self.feedback_str)
        if len(found_error_steps) == 0 and 'In step (Final response) the part' not in self.feedback_str:
            return None, None
        if 'In step (Final response) the part' in self.feedback_str:
            error_step = float('inf')
        else:
            error_step = int(found_error_steps[0])
        found_error_parts = re.findall(r'the part "(.*)" is incorrect', self.feedback_str)
        if len(found_error_parts) == 0:
            return error_step, None
        error_part = found_error_parts[0]
        return error_step, error_part
    
    def is_parsable(self):
        if "ERROR" in self.feedback_str:
            return False
        
        feedback = self.feedback_str
        length = len(feedback.split('\n'))
        if length > 1:
            return False
        if 'is incorrect' not in feedback:
            return False  # it should be wrong
        if 'because' not in feedback:
            return False
        
        error_step, error_part = self.parse_feedback()
        if error_step is None or error_part is None:
            return False
        return True
    
    def made_progress(self, prev_feedback):
        if "final response is also correct" in self.feedback_str.lower():
            return True
        curr_err_step, curr_err_part = self.parse_feedback()
        prev_err_step, prev_err_part = prev_feedback.parse_feedback()
        if curr_err_step is None or prev_err_step is None:
            print_red(f'curr_err_step or prev_err_step is None')
            print_red(f'curr_err_step: {self.feedback_str}')
            print_red(f'prev_err_step: {prev_err_step.feedback_str}')
            return False
        if curr_err_step == prev_err_step and curr_err_part == prev_err_part:
            return False
        if curr_err_step < prev_err_step:
            return False
        return True
    
    def is_included(self, a, b):
        if a in b:
            return True
        if b in a:
            return True
        return False

    def is_identical(self, new_feedback):
        curr_error_step, curr_error_part = self.parse_feedback()
        new_error_step , new_error_part  = new_feedback.parse_feedback()

        if curr_error_step != new_error_step:
            return False
        if not self.is_included(curr_error_part, new_error_part):
            return False
        return True


class ParsedAttempt(ABC):
    def __init__(self, attempt_str: str) -> None:
        """attempt_str SHOULD look like:
        (1) xxx
        (2) xxx
        (Final response) xxx the answer is xxx.
        """
        self.attempt_str = attempt_str

    @abstractmethod
    def has_valid_content(self) -> bool:
        raise NotImplementedError

    def clean_attempt(self):
        all_steps = self.attempt_str.split('\n')
        out_steps = []
        prev_step_header = ""
        for step in all_steps:
            # new question, hallucination
            if "Q:".lower() in step.lower():
                break
            elif "Options:".lower() in step.lower():
                break
            elif len(re.findall(r"^\(([a-zA-Z])\)", step)) == 1:
                break
            elif "Updated Answer:".lower() in step.lower():
                break
            elif "Feedback:".lower() in step.lower():
                break
            elif "Answer:".lower() in step.lower():
                # this we don't need
                continue
            elif "(Final response)".lower() in step.lower():
                header = "(Final response)"
            elif len(re.findall(r"^\((\d+[\.\d+]*)\)", step)) > 0:
                header = re.findall(r"^\((\d+[\.\d+]*)\)", step)[0]
            else:
                if self.verbose:
                    print_red(f"Unknown step format {step}")
                return None
            if header == prev_step_header:
                continue
            prev_step_header = header
            out_steps.append(step)
        if len(out_steps) == 0:
            print_red(f"Empty attempt after cleaning: {self.attempt_str}")
            return "ERROR" + self.attempt_str
        return '\n'.join(out_steps).strip()

    def is_parsable(self):
        if "ERROR" in self.attempt_str:
            return False
        
        attempted_answer = self.attempt_str.strip()
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
    
    def made_progress(self, prev_answer:str, prev_feedback:ParsedFeedback):
        attempted_answer = self.attempt_str.strip()
        if attempted_answer == prev_answer.strip():
            return False
        
        # check if steps other than the feedback is changed
        error_step, error_part = prev_feedback.parse_feedback()
        prev_steps = prev_answer.split("\n")
        if error_step == float('inf'):
            # last step
            error_step_int = len(prev_steps)
        else:
            error_step_int = int(error_step)
        new_steps = attempted_answer.split("\n")
        for i in range(error_step_int):
            # the error step should be changed
            prev_step = prev_steps[i].lower().strip()
            new_step = new_steps[i].lower().strip()
            if i == error_step_int - 1:
                if prev_step == new_step:
                    print_red(f"Error not changed. Feedback: {prev_feedback.feedback_str}")
                    print_red(f"error {error_step_int}, but prev step {prev_steps[i]} and new step {new_steps[i]}")
                    return False
                if error_part in new_step.lower():
                    print_red(f"Error part still in the step. Feedback: {prev_feedback.feedback_str}")
                    print_red(f"error {error_step_int}, but prev step {prev_steps[i]} and new step {new_steps[i]}")
                    return False
            elif prev_step != new_step:
                # since now we prompt with previous attempts, this should never be the case
                print_red(f"[BUG] Prev steps changed. Feedback is {prev_feedback.feedback_str}")
                print_red(f"error {error_step_int}, but prev step {prev_steps[i]} and new step {new_steps[i]}")
                return False
        return True


class ParsedTrajectory:
    def __init__(self, trajectory_data:str):
        self.trajectory_data = trajectory_data

    def _split_by_feedback(self, start_step: int = 1):
        outputs = []  # attempt, fb, attempt, fb, ...
        curr_attempt: List[str] = []
        # self.trajectory_data = self._add_last_feedback()
        is_multiple_chioce = False
        for step in self.trajectory_data.split('\n')[start_step:]:
            step = step.strip()
            if step.startswith('Options:'):
                is_multiple_chioce = True
                continue
            if is_multiple_chioce and len(re.findall(r"^\(([a-zA-Z])\)", step)) > 0:
                continue
            if step.startswith('List:'):
                continue

            if "answer:" in step.lower() or "updated answer:" in step.lower():
                continue
            if 'Feedback:' in step:
                # flush the attempt
                outputs.append('\n'.join(curr_attempt))
                # flush the feedback
                step = step.replace('Feedback:', '').strip()
                outputs.append(step)
                # reset
                curr_attempt = []
            else:
                curr_attempt.append(step)

        # e.g. only attempt, no feedback (happens during generation that model didn't finish)
        if len(curr_attempt) > 0:
            outputs.append('\n'.join(curr_attempt))
        return outputs

    def parse_trajectory(self):
        """returns [question, attempt, feedback, attempt, feedback, ...]
        """
        # first find the question
        all_steps = self.trajectory_data.split('\n')
        question = all_steps[0]
        if 'Q:' not in question:
            print_red(f"Question not found. {question}")
            return []
        # a multi-choice question
        if all_steps[1].strip().startswith('Options:'):
            for option_text in all_steps[1:]:
                if 'answer:' in option_text.lower():
                    break
                question += '\n' + option_text
        # word sorting question
        if 'List:' in all_steps[1]:
            question += '\n' + all_steps[1]
        question = question.strip()
        
        # parse the rest
        parsed_steps = self._split_by_feedback()
        parsed_steps = [question] + parsed_steps
        return parsed_steps
    
    def parse_trajectory_wo_question(self, question:str):
        """returns [attempt, feedback, attempt, feedback, ...]
        """
        parsed_steps = self._split_by_feedback(start_step=0)
        parsed_steps = [question] + parsed_steps
        return parsed_steps