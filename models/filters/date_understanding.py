from models.filters.base import ParsedAttempt, ParsedFeedback
from dateutil.parser import parse
from utils.constants import CHARACTERS
import re


class ParsedAttempt_DateUnderstanding(ParsedAttempt):
    def has_valid_content(self):
        """
        check of the chosen option is actually the last step
        this function needs the question to be included in the attempt
        """
        lines = self.attempt_str.split("\n")
        all_choices = []
        for line in lines:
            if len(re.findall(r"^\(([a-zA-Z])\)", line)) > 0:
                all_choices.append(line)
        final_answer = lines[-1]
        final_choice = re.search(r"answer is \(([a-zA-Z])\).*", final_answer)
            
        if final_choice is None or len(final_choice.groups()) == 0:
            return False
        final_choice = final_choice.groups()[0]
        final_choice_idx = CHARACTERS.index(final_choice.upper())
        final_chosen_date = all_choices[final_choice_idx].split(')')[-1].strip()

        last_step = lines[-2]
        textual_dates = re.findall("is (\w+\.? \d+\w*?, \d+)", last_step)
        for found_date in textual_dates:
            try:
                if parse(found_date) == parse(final_chosen_date):
                    return True
            except:
                continue
        
        all_words = last_step.split()
        for word in all_words:
            try:
                if parse(word) == parse(final_chosen_date):
                    return True
            except:
                continue
        return False
    

class ParsedFeedback_DateUnderstanding(ParsedFeedback):        
    def has_valid_content(self, feedbacked_attempt: str) -> bool:
        return True