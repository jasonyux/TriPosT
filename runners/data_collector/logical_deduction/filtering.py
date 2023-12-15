import re
from utils.constants import CHARACTERS
import string


def normalize_string(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def remove_dash(text):
        return text.replace('-', '')

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_dash(remove_articles(remove_punc(lower(s)))))


def multiple_choice_filter(attempt: str, correct_choice: str, formatted_choices: str):
    all_choices = formatted_choices.split('\n')
    correct_choice_idx = CHARACTERS.index(correct_choice)
    correct_choice_content = all_choices[correct_choice_idx].split(')')[-1].strip()
    normalized_correct_choice_content = normalize_string(correct_choice_content)
    critical_tokens = normalized_correct_choice_content.split()
    for i, token in enumerate(critical_tokens):
        if token in ['is', 'are']:
            critical_tokens.remove(token)
        if token.endswith('s'):
            critical_tokens[i] = token[:-1]
        

    second_last_step = attempt.split('\n')[-2]
    normalized_second_last_step = normalize_string(second_last_step)
    last_step = attempt.split('\n')[-1]
    normalized_last_step = normalize_string(last_step)

    if not all([token in normalized_last_step for token in critical_tokens]):
        return False
    if not all([token in normalized_second_last_step for token in critical_tokens]):
        return False
    return True