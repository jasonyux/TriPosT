from models.filters.base import ParsedFeedback
from typing import List
from utils.utils import print_red
import re


class LLMEditor:
    """shared functions for LLM Editor
    """
    def _add_last_feedback(self, pred: str):
        all_steps = pred.split('\n')
        if 'Feedback:' in all_steps[-1]:
            return pred
        elif 'final response' in all_steps[-1].lower():
            last_num_step = all_steps[-2]
        else:
            print_red(f'last step is not final response or feedback {pred}')
            return None
        step_num = re.findall(r"^\((\d+[\.\d+]*)\).*", last_num_step)[0]
        feedback = f"Feedback: Step (1) to step ({step_num}) are correct. The final response is also correct."
        return pred + '\n' + feedback
    
    def _split_by_feedback(self, pred: str):
        attempts = []
        all_feedbacks = []
        curr_attempt: List[str] = []
        pred = self._add_last_feedback(pred)
        is_multiple_chioce = False
        for step in pred.split('\n')[1:]:
            if "options:" in step.lower():
                is_multiple_chioce = True
                continue
            if is_multiple_chioce and len(re.findall(r"^\(([a-zA-Z])\)", step)) > 0:
                continue
            if "list:" in step.lower():
                # for word sorting
                continue

            if "answer:" in step.lower() or "updated answer:" in step.lower():
                continue
            if 'Feedback:' in step:
                step = step.replace('Feedback:', '').strip()
                all_feedbacks.append(step)
                attempts.append('\n'.join(curr_attempt))
                curr_attempt = []
            else:
                curr_attempt.append(step)
        return attempts, all_feedbacks
    
    def clean_pred(self, p: str):
        all_steps = p.split('\n')
        out_steps = []
        prev_step_header = ""
        for step in all_steps:
            if "Q:".lower() in step.lower():
                header = "Q:"
            elif "Options:".lower() in step.lower():
                header = "Options:"
            elif "List:".lower() in step.lower():
                # for word sorting
                header = "List:"
            elif len(re.findall(r"^\(([a-zA-Z])\)", step)) == 1:
                header = re.findall(r"^\(([a-zA-Z])\)", step)[0]
            elif "Updated Answer:".lower() in step.lower():
                header = "Updated Answer:"
            elif "Answer:".lower() in step.lower():
                header = "Answer:"
            elif "Feedback:".lower() in step.lower():
                header = "Feedback:"
            elif "(Final response)".lower() in step.lower():
                header = "(Final response)"
            elif len(re.findall(r"^\((\d+[\.\d+]*)\)", step)) > 0:
                header = re.findall(r"^\((\d+[\.\d+]*)\)", step)[0]
            else:
                print_red(f"Unknown step format {step}")
                return None
            if header == prev_step_header:
                continue
            if prev_step_header != "" and header == "Q:":
                # new question
                break
            prev_step_header = header
            out_steps.append(step)
        return '\n'.join(out_steps)
    
    def _get_prev_partial_answer(self, prev_answer, prev_feedback:ParsedFeedback):
        error_step, _ = prev_feedback.parse_feedback()
        if error_step is None:
            print_red(f"Error step not found. Feedback: {prev_feedback.feedback_str}")
            raise ValueError
        
        prev_steps = prev_answer.split("\n")
        if error_step == float("inf"):
            return "\n".join(prev_steps[:-1]) + "\n"
        error_step_int = int(error_step) - 1
        if error_step_int == 0:
            return ""
        return "\n".join(prev_steps[:error_step_int]) + "\n"