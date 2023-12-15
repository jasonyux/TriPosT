from models.filters.base import ParsedAttempt, ParsedFeedback
from utils.utils import print_red
import re


class ParsedAttempt_MultistepArithmetic(ParsedAttempt):
    def has_valid_content(self):
        lines = self.attempt_str.split("\n")
        last_step = lines[-2]
        final_response = lines[-1]
        if not final_response.endswith('.'):
            final_response += '.'
        numeric_outputs = re.findall(r'the answer is (.*)\.', final_response)
        if len(numeric_outputs) != 1:
            print_red("Error in parsing numeric output:", final_response)
            return False
        try:
            final_num_str = numeric_outputs[0]
            final_num = float(final_num_str)
        except:
            print("Error in converting parsed number:", final_response)
            return False
        
        if f'={final_num_str}.' not in last_step and f'= {final_num_str}.' not in last_step:
            print_red(f'Last step {final_num_str=} inconsistent with final response:')
            print_red(last_step)
            print_red(final_response)
            return False
        return True


class ParsedFeedback_MultistepArithmetic(ParsedFeedback):
    def has_valid_content(self, feedbacked_attempt: str) -> bool:
        return True