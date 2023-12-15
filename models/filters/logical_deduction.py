from models.filters.base import ParsedAttempt, ParsedFeedback
from utils.utils import print_red
import re
import string


def normalize_string(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_dash(text):
        return text.replace('-', '')

    return white_space_fix(remove_dash(remove_articles(remove_punc(lower(s)))))


class ParsedAttempt_LogicalDeduction(ParsedAttempt):
    def has_valid_content(self):
        if 'Options' not in self.attempt_str:
            print("Options missing, skipping")
            return True
        
        lines = self.attempt_str.split("\n")
        last_step = lines[-1]
        second_last_step = lines[-2]
        picked_choice = re.search(r'the answer is \(([a-zA-Z])\)\.*', last_step)
        if picked_choice is None:
            return False
        picked_choice = picked_choice.group(1)

        correct_choice_content = ""
        for l in lines:
            if l.startswith(f'({picked_choice})'):
                correct_choice_content = l.split(')')[-1].strip()
                break
        if correct_choice_content == "":
            print_red("correct_choice_content == ''")
            return False
        
        normalized_correct_choice_content = normalize_string(correct_choice_content)
        critical_tokens = normalized_correct_choice_content.split()
        for i, token in enumerate(critical_tokens):
            if token in ['is', 'are']:
                critical_tokens.remove(token)
            if token.endswith('s'):
                critical_tokens[i] = token[:-1]
        
        normalized_second_last_step = normalize_string(second_last_step)
        normalized_last_step = normalize_string(last_step)

        if not all([token in normalized_last_step for token in critical_tokens]):
            print_red("not all([token in normalized_last_step for token in critical_tokens])")
            print_red(f"{critical_tokens=}")
            print_red(f"{normalized_last_step=}")
            return False
        if not all([token in normalized_second_last_step for token in critical_tokens]):
            print_red("not all([token in normalized_second_last_step for token in critical_tokens])")
            print_red(f"{critical_tokens=}")
            print_red(f"{normalized_second_last_step=}")
            return False
        return True


class ParsedFeedback_LogicalDeduction(ParsedFeedback):
    def has_valid_content(self, feedbacked_attempt: str) -> bool:
        return True