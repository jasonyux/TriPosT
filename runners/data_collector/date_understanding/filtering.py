import re
from dateutil.parser import parse
from utils.constants import CHARACTERS


def multiple_choice_filter(attempt: str, correct_choice: str, formatted_choices: str):
    all_choices = formatted_choices.split('\n')
    correct_choice_idx = CHARACTERS.index(correct_choice)
    correct_date = all_choices[correct_choice_idx].split(')')[-1].strip()

    last_step = attempt.split('\n')[-2]

    textual_dates = re.findall("is (\w+\.? \d+\w*?, \d+)", last_step)
    for found_date in textual_dates:
        try:
            if parse(found_date) == parse(correct_date):
                return True
        except:
            continue
    
    all_words = last_step.split()
    has_match = False
    for word in all_words:
        try:
            if parse(word) == parse(correct_date):
                return True
        except:
            continue
    return has_match