import re

from copy import deepcopy
from anytree import Node, PreOrderIter, RenderTree
from models.base import GenerativeModel
from models.verifier.base import ScriptedVerifier


ORDER_TO_NUM = {
    'first': 0,
    'second': 1,
    'third': 2,
    'fourth': 3,
    'fifth': 4,
    'sixth': 5,
    'seventh': 6,
    'eighth': 7,
    'ninth': 8,
    'tenth': 9,
    'eleventh': 10,
    'twelfth': 11,
    'thirteenth': 12,
    'fourteenth': 13,
    'fifteenth': 14,
    'sixteenth': 15,
    'seventeenth': 16,
    'eighteenth': 17,
    'nineteenth': 18,
    'twentieth': 19,
    'twenty-first': 20,
}


NUM_TO_ORDER = {v: k for k, v in ORDER_TO_NUM.items()}


class WS_Verifier(ScriptedVerifier):
    ACT_SORT = 'sort'  # check if the sorting is the same as what I would have from its previous step
    ACT_TO_NUM = 'to_num'  # convert each word to number
    ACT_TO_SUBPART = 'to_subpart'  # decide which subpart to sort
    ACT_LETTER_TO_SORT = 'letter_to_sort'  # decide which letter to sort
    ACT_SELF_CONSISTENT = 'self_consistent'  # check self-consistency of LHS and RHS in the same step is consistent
    ACT_COMBINE = 'combine'  # combine prev results
    ACT_FINAL = 'final'  # final result

    def __init__(self,
            fb_wrong_segment: bool = True,
            fb_reason: bool = True,
            fb_specific_reason: bool = True):
        """used to verify a rationale and return textual feedback

        Args:
            - fb_wrong_segment : whether to mention the wrong segment in the feedback
            - fb_reason: whether to add a reason for why a mistake is made in the feedback. If false, no reasoning will be given.
            - fb_specific_reason : whether to add a specific reason for why a mistake is made in the feedback. Otherwise, a generic reason is given.
        """
        self.error_stats = {
            self.ACT_SORT: 0,
            self.ACT_TO_NUM: 0,
            self.ACT_TO_SUBPART: 0,
            self.ACT_LETTER_TO_SORT: 0,
            self.ACT_SELF_CONSISTENT: 0,
            self.ACT_COMBINE: 0,
            self.ACT_FINAL: 0,
            'total': 0,
        }
        self.fb_wrong_segment = fb_wrong_segment
        self.fb_reason = fb_reason
        self.fb_specific_reason = fb_specific_reason
        return

    def _is_step_larger(self, src_step: str, target_step: str):
        src_step = src_step.strip()
        target_step = target_step.strip()
        
        if src_step == target_step:
            return False
        if 'Final response' in src_step:
            return True
        elif 'Final response' in target_step:
            return False
        
        all_src_steps = src_step.split('.')
        all_target_steps = target_step.split('.')
        # compare each step
        for src_step, target_step in zip(all_src_steps, all_target_steps):
            if int(src_step) == int(target_step):
                continue
            return int(src_step) > int(target_step)
        if len(all_src_steps) > len(all_target_steps):
            return True
        elif len(all_src_steps) < len(all_target_steps):
            return False
        raise ValueError(f'Cannot compare {src_step} and {target_step}')

    def _parse_entities(self, data: dict):
        question = data['question']
        targets = data['target']
        entities = re.findall(r'List: (.+)', question)[0].split()
        assert(sorted(entities) == targets)

        return entities
    
    def _parse_segment(self, segment: str):
        assert(segment[-1] == '.')
        segment = segment[:-1]  # remove the last dot

        step_number = re.findall(r'^\((.+)\) \w.+', segment)[0]
        if step_number == '1':
            lhs, rhs = segment.split(':')
            actions = {
                (self.ACT_LETTER_TO_SORT, 0): rhs.strip(),
                (self.ACT_SELF_CONSISTENT, ): (lhs[lhs.index(')')+1:].strip(), rhs.strip()),
                (self.ACT_TO_NUM, 0): rhs.strip(),
            }
        elif "sort this subpart" in segment.lower():
            # by looking at their x letters
            idx_n_text = re.findall(r'by looking at their (\w+) letters:', segment)[0]
            idx = ORDER_TO_NUM[idx_n_text.lower()]

            lhs, rhs = segment.split(':')
            actions = {
                (self.ACT_TO_SUBPART, ): lhs[lhs.index(')')+1:].strip(),
                (self.ACT_LETTER_TO_SORT, idx): lhs[lhs.index(')')+1:].strip(),
                (self.ACT_SELF_CONSISTENT, ): (lhs[lhs.index(')')+1:].strip(), rhs.strip()),
                (self.ACT_TO_NUM, idx): rhs.strip(),
            }
        elif "again by looking at" in segment:
            # by looking at their x letters
            idx_n_text = re.findall(r'by looking at their (\w+) letters:', segment)[0]
            idx = ORDER_TO_NUM[idx_n_text.lower()]

            lhs, rhs = segment.split(':')
            actions = {
                (self.ACT_TO_SUBPART, ): lhs[lhs.index(')')+1:].strip(),
                (self.ACT_LETTER_TO_SORT, idx): lhs[lhs.index(')')+1:].strip(),
                (self.ACT_SELF_CONSISTENT, ): (lhs[lhs.index(')')+1:].strip(), rhs.strip()),
                (self.ACT_TO_NUM, idx): rhs.strip(),
            }
        elif "We now have" in segment:
            actions = {
                (self.ACT_SORT,): segment.split(':')[1].strip()
            }
        elif "Hence, we have" in segment:
            actions = {
                (self.ACT_COMBINE,): re.findall(r'Hence, we have (.+)', segment)[0]
            }
        elif "So the answer is" in segment:
            actions = {
                (self.ACT_FINAL,): segment.split(':')[1].strip()
            }
        else:
            raise ValueError(f'Unknown segment: {segment}')
        return actions

    def to_tree_rationale(self, all_steps: list):
        final_step = all_steps[-1]
        if 'Final response' not in final_step:
            raise ValueError('Final response is not the last step.')
        
        root_node = Node(
            final_step,
            step_num='Final response',
            prev_step=None,  # added later
            actions=self._parse_segment(final_step)
        )
        tree_mapping = {
            'root': root_node,
        }
        prev_step = ''
        for i, step in enumerate(all_steps[:-1]):
            extracted_step = re.findall(r'^\((.+)\) \w.+', step)[0]
            
            parent_step = '.'.join(extracted_step.split('.')[:-1])
            if parent_step == '':
                parent_node = root_node
            else:
                parent_node = tree_mapping[parent_step]

            # previous node step
            if prev_step == '':
                prev_node = None
            else:
                prev_node = tree_mapping[prev_step]
            
            # extract other info to be used leter, such as task and attempt
            new_node = Node(
                step, 
                step_num=extracted_step, 
                parent=parent_node, 
                prev_step=prev_node,
                actions=self._parse_segment(step)
            )
            tree_mapping[extracted_step] = new_node

            prev_step = extracted_step
        root_node.prev_step = new_node
        return root_node
    
    def _verify_to_num(self, task, result, parsed_entities, node):
        step_number = node.step_num
        idx_n = task[1]
        orders = result.split(',')
        for order in orders:
            order = order.strip()
            lhs, rhs = order.split()
            word = lhs.split('=')[0][1:-1]
            nth_letter = lhs.split('=')[1][1:-1]
            num = int(rhs[1:-1])

            if nth_letter.isupper() or word[0].isupper():
                raise ValueError(f'first letter of word is capitalized in {lhs}')

            if word not in parsed_entities:
                return {
                    'err_step': step_number,
                    'incorrect_segment': order,
                    'reason': f'"{word}" is not in the list of words provided by the question statement: {" ".join(parsed_entities)}',
                    'generic_reason': f'it is inconsistent with the question statement',
                    'node': node
                }
            if nth_letter != word[idx_n]:
                return {
                    'err_step': step_number,
                    'incorrect_segment': order,
                    'reason': f'the {NUM_TO_ORDER[idx_n]} letter of "{word}" should be "{word[idx_n]}"',
                    'generic_reason': f'you made a mistake in finding the k-th letter of some words',
                    'node': node
                }
            real_num = ord(nth_letter) - ord('a') + 1
            if num != real_num:
                return {
                    'err_step': step_number,
                    'incorrect_segment': order,
                    'reason': f'the alphabetical order for the character "{nth_letter}" should be {real_num}',
                    'generic_reason': f'you made a mistake in finding the alphabetical order of some characters',
                    'node': node
                }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def _verify_sort(self, task, result, parsed_entities, node):
        result = result.strip()
        single_word_subpart_found = re.findall(r'(\["[^"]+"\])', result)
        if len(single_word_subpart_found) > 0:
            return {
                'err_step': node.step_num,
                'incorrect_segment': result,
                'reason': f'"{single_word_subpart_found[0]}" has only a single word, so it is not a subpart that needs further sorting',
                'generic_reason': f'subparts to sort need to include more than one word',
                'node': node
            }

        step_number = node.step_num
        prev_step: Node = node.prev_step
        for prev_t, prev_r in prev_step.actions.items():
            if prev_t[0] == self.ACT_TO_NUM:
                prev_result = prev_r
                break
        
        # convert the previous result to a sort list of [(num, list/word), ...]
        _map = {}
        for entity in prev_result.split(','):
            entity = entity.strip()
            lhs, rhs = entity.split()
            word = lhs.split('=')[0][1:-1]
            num = int(rhs[1:-1])
            if num not in _map:
                _map[num] = []
            _map[num].append(word)
        sort_list = sorted(_map.items(), key=lambda x: x[0])

        # verify if result is the same as this sort list
        if '<' not in result and '?' not in result:
            return {
                'err_step': step_number,
                'incorrect_segment': result.strip(),
                'reason': f'it is not a valid sorted list',
                'generic_reason': f'it is not a valid sorted list',
                'node': node
            }
        proposed_sort_list = result.split('<')
        if len(proposed_sort_list) != len(sort_list):
            sort_list_keys = set([x[0] for x in sort_list])
            proposed_sort_list_keys = re.findall(r'\((\d+)\)', result)
            proposed_sort_list_keys = set([int(x) for x in proposed_sort_list_keys])
            diff_1 = sort_list_keys - proposed_sort_list_keys
            diff_2 = proposed_sort_list_keys - sort_list_keys
            if len(diff_1) > 0:
                missing_nums = [f'({i})' for i in diff_1]
                return {
                    'err_step': step_number,
                    'incorrect_segment': result.strip(),
                    'reason': f'according to step ({prev_step.step_num}), this ordering does not have words with the numbers {" ".join(missing_nums)} sorted correctly',
                    'generic_reason': f'it is inconsistent with the result in step ({prev_step.step_num})',
                    'node': node
                }
            if len(diff_2) > 0:
                extra_nums = [f'({i})' for i in diff_2]
                return {
                    'err_step': step_number,
                    'incorrect_segment': result.strip(),
                    'reason': f'words with the numbers {" ".join(extra_nums)} does not exist in the result from previous step ({prev_step.step_num})',
                    'generic_reason': f'it is inconsistent with the result in step ({prev_step.step_num})',
                    'node': node
                }
            raise ValueError(f'cannot find the difference between {sort_list_keys} and {proposed_sort_list_keys} despite the lengths being the different')
        for i, p in enumerate(proposed_sort_list):
            p = p.strip()
            tokens = p.split(' ')
            if '[' in p:
                # has a list
                num = int(tokens[0][1:-1])
                words = set()
                for t in tokens[1:]:
                    t = t.replace('[', '')
                    t = t.replace(']', '')
                    t = t.replace('"', '')
                    if t == '?':
                        continue
                    t = t.replace('?','')
                    if t == '':
                        continue
                    if t not in parsed_entities:
                        return {
                            'err_step': step_number,
                            'incorrect_segment': result,
                            'reason': f'the word "{t}" is not in the list of words provided by the question statement: {" ".join(parsed_entities)}',
                            'generic_reason': f'it is inconsistent with the question statement',
                            'node': node
                        }
                    if t in words:
                        return {
                            'err_step': step_number,
                            'incorrect_segment': result,
                            'reason': f'the word "{t}" inside is duplicated',
                            'generic_reason': f'some words are duplicated',
                            'node': node
                        }
                    words.add(t)
            else:
                # only has a word
                num = int(tokens[0][1:-1])
                word = tokens[1][1:-1]
                words = set([word])
                if word not in parsed_entities:
                    return {
                        'err_step': step_number,
                        'incorrect_segment': result.strip(),
                        'reason': f'the word "{word}" is not in the list of words provided by the question statement: {" ".join(parsed_entities)}',
                        'generic_reason': f'it is inconsistent with the question statement',
                        'node': node
                    }
            # verify if the num and words are the same as the sort list
            if num != sort_list[i][0]:
                if i == 0:
                    return {
                        'err_step': step_number,
                        'incorrect_segment': result.strip(),
                        'reason': f'words are not sorted in ascending order. Using results from step ({prev_step.step_num}), the first and smallest number should be {sort_list[i][0]}',
                        'generic_reason': f'it is inconsistent with the result in step ({prev_step.step_num})',
                        'node': node
                    }
                else:
                    return {
                        'err_step': step_number,
                        'incorrect_segment': result.strip(),
                        'reason': f'words are not sorted in ascending order. Using results from step ({prev_step.step_num}), those words with ({sort_list[i][0]}) should come after words with ({sort_list[i-1][0]})',
                        'generic_reason': f'it is inconsistent with the result in step ({prev_step.step_num})',
                        'node': node
                    }
            if words != set(sort_list[i][1]):
                return {
                    'err_step': step_number,
                    'incorrect_segment': result.strip(),
                    'reason': f'according to step ({prev_step.step_num}), words with ({sort_list[i][0]}) should be: {" ".join(sort_list[i][1])}',
                    'generic_reason': f'it is inconsistent with the result in step ({prev_step.step_num})',
                    'node': node
                }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}
    
    def _verify_combine(self, task, result, parsed_entities, node):
        result = result.strip()
        if "?" in result:
            return {
                'err_step': node.step_num,
                'incorrect_segment': result,
                'reason': f'this subpart has not yet been sorted completely (contains "?"), so it should not be a conclusion step',
                'generic_reason': f'this subpart has not yet been sorted completely',
                'node': node
            }
        
        # check all the subnodes
        # works as-is even for the last step of Hence, because its parent node is the root
        all_constraints = []
        for n in PreOrderIter(node.parent):
            if 'We now have' in n.name and "<" in n.name:
                all_constraints.append((n.step_num, n.actions[(self.ACT_SORT,)]))
        assert(len(all_constraints) > 0)

        # parse current result into a list of [word1, word2, ...]
        current_result = result.split(' ')
        current_order = []
        for t in current_result:
            t = t.strip()
            if t == '':
                continue
            if t == '<':
                continue
            t = t.replace('[', '')
            t = t.replace(']', '')
            t = t.replace('"', '')
            if t == '?':
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': result,
                    'reason': f'{result} is not a conclusive ordering of the words (contains "?")',
                    'generic_reason': f'this is not a conclusive ordering of the words',
                    'node': node
                }
            t = t.replace('?','')
            if t == '>':
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': result,
                    'reason': f'you should always sort in ascending order, from smallest to largest',
                    'generic_reason': f'you should always sort from smallest to largest',
                    'node': node
                }
            if t not in parsed_entities:
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': result,
                    'reason': f'"{t}" is not in the list of words provided by the question statement: {" ".join(parsed_entities)}',
                    'generic_reason': f'it is inconsistent with the question statement',
                    'node': node
                }
            current_order.append(t)
        
        # check if all the constraints are satisfied       
        for c_step_num, c in all_constraints:
            # parse the constraint as [{word_1, word_2, ...}, word_5, ...]
            proposed_sort_list = c.split('<')
            proposed_sorting = []
            for i, p in enumerate(proposed_sort_list):
                p = p.strip()
                tokens = p.split(' ')
                if '[' in p:
                    # has a list
                    words = set()
                    for t in tokens[1:]:
                        t = t.replace('[', '')
                        t = t.replace(']', '')
                        t = t.replace('"', '')
                        if t == '?':
                            continue
                        t = t.replace('?','')
                        if t == '':
                            continue
                        assert(t in parsed_entities)  # should be checked already
                        words.add(t)
                else:
                    # only has a word
                    word = tokens[1][1:-1]
                    words = set([word])
                    assert(word in parsed_entities)  # should be checked already
                proposed_sorting.append(words)
            # verify
            for i, ws in enumerate(proposed_sorting[:-1]):
                next_ws = proposed_sorting[i+1]
                for w in ws:
                    for nw in next_ws:
                        assert(w != nw)  # should be checked already
                        # w should be < nw
                        if w not in current_order:
                            return {
                                'err_step': node.step_num,
                                'incorrect_segment': result,
                                'reason': f'the order of the word "{w}" is already established in step ({c_step_num}) that: {c}',
                                'generic_reason': f'it is inconsistent with the result in step ({c_step_num})',
                                'node': node
                            }
                        if nw not in current_order:
                            return {
                                'err_step': node.step_num,
                                'incorrect_segment': result,
                                'reason': f'the order of the word "{nw}" is already established in step ({c_step_num}) that: {c}',
                                'generic_reason': f'it is inconsistent with the result in step ({c_step_num})',
                                'node': node
                            }
                        w_idx = current_order.index(w)
                        nw_idx = current_order.index(nw)
                        if w_idx > nw_idx:
                            return {
                                'err_step': node.step_num,
                                'incorrect_segment': result,
                                'reason': f'the order of "{w}" and "{nw}" is inconsistent with the result in step ({c_step_num}) that: {c}',
                                'generic_reason': f'it is inconsistent with the result in step ({c_step_num})',
                                'node': node
                            }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def _verify_letter_to_sort(self, task, result, parsed_entities, node):
        # check how many times PREVIOUS had the unknown list appeared
        task, letter_idx = task
        if node.step_num == '1':
            if 'first letter' not in node.name:
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': result.lower(),
                    'reason': f'you should start sorting from the first letter of the words',
                    'generic_reason': f'you should start sorting from the first letter of the words',
                    'node': node
                }
        else:
            subpart_to_sort = re.findall(r'(\[.*\]) (?:again )?by looking at', result)[0].strip()
            # check if it is the correct letter to sort
            # first convert the one to sort into a list of words
            subpart_to_sort_words = set()
            for t in subpart_to_sort[1:-1].split(' '):
                t = t.strip()
                t = t.replace('[', '')
                t = t.replace(']', '')
                t = t.replace('"', '')
                if t == '?':
                    continue
                t = t.replace('?','')
                if t == '':
                    continue
                assert(t in parsed_entities)
                subpart_to_sort_words.add(t)

            prev_subpart_looked_at = {}
            for n in PreOrderIter(node.root):
                if 'by looking at' in n.name and self._is_step_larger(node.step_num, n.step_num):
                    subpart_sorted = re.findall(r'(\[.*\]) (?:again )?by looking at', n.name)[0].strip()
                    subpart_sorted_words = set()
                    for t in subpart_sorted[1:-1].split(' '):
                        t = t.strip()
                        t = t.replace('[', '')
                        t = t.replace(']', '')
                        t = t.replace('"', '')
                        if t == '?':
                            continue
                        t = t.replace('?','')
                        if t == '':
                            continue
                        assert(t in parsed_entities)
                        subpart_sorted_words.add(t)

                    idx_n_text = re.findall(r'by looking at their (\w+) letters:', n.name)[0]
                    letters_looked_at = ORDER_TO_NUM[idx_n_text.lower()]
                    
                    if subpart_to_sort_words.issubset(subpart_sorted_words):
                        if subpart_to_sort not in prev_subpart_looked_at:
                            prev_subpart_looked_at[subpart_to_sort] = []
                        prev_subpart_looked_at[subpart_to_sort].append((letters_looked_at, n.step_num))
            
            if subpart_to_sort not in prev_subpart_looked_at:
                # first time
                if letter_idx != 1:
                    return {
                        'err_step': node.step_num,
                        'incorrect_segment': result.lower(),
                        'reason': f'the second letter of {subpart_to_sort} has not been sorted yet',
                        'generic_reason': f'you should make sure each letter is sorted before moving on to the next letter',
                        'node': node
                    }
            else:
                subpart_num_sorted = prev_subpart_looked_at[subpart_to_sort]
                subpart_num_sorted = sorted(subpart_num_sorted, key=lambda x: x[0])
                if letter_idx <= subpart_num_sorted[-1][0]:
                    seen_step = [step for idx, step in subpart_num_sorted if idx == letter_idx][0]
                    return {
                        'err_step': node.step_num,
                        'incorrect_segment': result.lower(),
                        'reason': f"""the {NUM_TO_ORDER[letter_idx]} letter of {subpart_to_sort} has already been checked in step ({seen_step})""",
                        'generic_reason': f'it is inconsistent with the result in step ({seen_step})',
                        'node': node
                    }
                elif letter_idx - 1 != subpart_num_sorted[-1][0]:
                    should_sort = subpart_num_sorted[-1][0] + 1
                    return {
                        'err_step': node.step_num,
                        'incorrect_segment': result.lower(),
                        'reason': f'the {NUM_TO_ORDER[should_sort]} letter of {subpart_to_sort} has not been sorted yet',
                        'generic_reason': f'you should make sure each letter is sorted before moving on to the next letter',
                        'node': node
                    }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def _verify_self_consistent(self, task, result, parsed_entities, node):
        res_lhs, res_rhs = result
        if node.step_num == '1':
            # check if it has the same number of words as the question
            orders = res_rhs.split(',')
            found_words = set()
            for order in orders:
                order = order.strip()
                lhs, rhs = order.split()
                word = lhs.split('=')[0][1:-1]

                if word[0].isupper():
                    raise ValueError(f'first letter of word is capitalized in {lhs}')

                if word not in parsed_entities:
                    return {
                        'err_step': node.step_num,
                        'incorrect_segment': order,
                        'reason': f'"{word}" is not in the list of words provided by the question statement: {" ".join(parsed_entities)}',
                        'generic_reason': f'it is inconsistent with the question statement',
                        'node': node
                    }
                if word in found_words:
                    return {
                        'err_step': node.step_num,
                        'incorrect_segment': order,
                        'reason': f'"{word}" is duplicated',
                        'generic_reason': f'some words are duplicated',
                        'node': node
                    }
                found_words.add(word)
            diff = set(parsed_entities) - found_words
            if len(diff) > 0:
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': res_rhs,
                    'reason': f'you missed the words "{" ".join(diff)}", which are in the list of words provided by the question statement: {" ".join(parsed_entities)}',
                    'generic_reason': f'it is inconsistent with the question statement',
                    'node': node
                }
        else:
            # check lhs and rhs if we are sorting the same words
            subpart_to_sort = re.findall(r'(\[.*\]) (?:again )?by looking at', res_lhs)[0].strip()
            subpart_to_sort_words = set()
            for t in subpart_to_sort[1:-1].split(' '):
                t = t.strip()
                t = t.replace('[', '')
                t = t.replace(']', '')
                t = t.replace('"', '')
                if t == '?':
                    continue
                t = t.replace('?','')
                if t == '':
                    continue
                assert(t in parsed_entities)
                subpart_to_sort_words.add(t)
            sorting_words = set()
            for w in res_rhs.split(','):
                w = w.strip()
                lhs, rhs = w.split()
                word = lhs.split('=')[0][1:-1]
                if word not in subpart_to_sort_words:
                    return {
                        'err_step': node.step_num,
                        'incorrect_segment': rhs,
                        'reason': f'"{word}" is not in the list of words you wanted to sort: {subpart_to_sort}',
                        'generic_reason': f'it is inconsistent with the list of words you wanted to sort',
                        'node': node
                    }
                sorting_words.add(word)
            diff = subpart_to_sort_words - sorting_words
            if len(diff) > 0:
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': res_rhs,
                    'reason': f'you miss the words "{" ".join(diff)}", which were in the subpart you wanted to sort: {subpart_to_sort}',
                    'generic_reason': f'it is inconsistent with the list of words you wanted to sort',
                    'node': node
                }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def _verify_to_subpart(self, task, result, parsed_entities, node):
        # check if it is the correct thing to sort
        subpart_to_sort = re.findall(r'(\[.*\]) (?:again )?by looking at', result)[0].strip()
        prev_sibling_we_now_have = None
        prev_sibling_we_now_have_step = '0'
        for n in node.siblings:
            if 'We now have' in n.name and self._is_step_larger(node.step_num, n.step_num):
                if self._is_step_larger(n.step_num, prev_sibling_we_now_have_step):
                    prev_sibling_we_now_have = n
                    prev_sibling_we_now_have_step = n.step_num
        assert(prev_sibling_we_now_have is not None)
        if subpart_to_sort.replace(' ', '') not in prev_sibling_we_now_have.name.replace(' ', ''):
            return {
                'err_step': node.step_num,
                'incorrect_segment': subpart_to_sort,
                'reason': f'this subpart is not amongst the unknown subparts in step ({prev_sibling_we_now_have.step_num})',
                'generic_reason': f'it is inconsistent with the result in step ({prev_sibling_we_now_have.step_num})',
                'node': node
            }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def _verify_final(self, task, result, parsed_entities, node):
        # check if the final result is correct
        final_sorting = result.split()
        last_sorted_node: Node = node.prev_step
        last_sorted_step = last_sorted_node.name[:-1]
        if 'Hence' in last_sorted_node.name:
            sorted_subpart = re.findall(r'Hence, we have (.+)', last_sorted_step)[0]
        elif 'We now have' in last_sorted_node.name:
            sorted_subpart = last_sorted_step.split(":")[1]
        else:
            raise ValueError(f'last sorted node is not a "Hence" nor a "We now have" node: {last_sorted_step}')
        
        # check if the final result is correct, assuming the all prev nodes are checked
        last_sorted_words = []
        for t in sorted_subpart.split(' '):
            t = t.strip()
            if t == '<':
                continue
            if t == '':
                continue
            if '(' in t:
                continue
            t = t.replace('[', '')
            t = t.replace(']', '')
            t = t.replace('"', '')
            assert(t in parsed_entities)
            last_sorted_words.append(t)
        if set(last_sorted_words) != set(parsed_entities):
            return {
                'err_step': node.step_num,
                'incorrect_segment': '',
                'reason': f'you have not yet sorted all the words in the question statement in step ({last_sorted_node.step_num})',
                'generic_reason': f'you have not yet sorted all the words in the question statement',
                'node': node
            }
        elif final_sorting != last_sorted_words:
            return {
                'err_step': node.step_num,
                'incorrect_segment': result,
                'reason': f'in step ({last_sorted_node.step_num}) it is established that: {sorted_subpart}',
                'generic_reason': f'it is inconsistent with the result in step ({last_sorted_node.step_num})',
                'node': node
            }
        # if reached here, the result should be correct
        assert(final_sorting == sorted(parsed_entities))
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}
    
    def verify_action(self, task, result, parsed_entities, node):
        if task[0] == self.ACT_TO_NUM:
            return self._verify_to_num(task, result, parsed_entities, node)
        elif task[0] == self.ACT_SORT:
            return self._verify_sort(task, result, parsed_entities, node)
        elif task[0] == self.ACT_COMBINE:
            return self._verify_combine(task, result, parsed_entities, node)
        elif task[0] == self.ACT_LETTER_TO_SORT:
            return self._verify_letter_to_sort(task, result, parsed_entities, node)
        elif task[0] ==  self.ACT_SELF_CONSISTENT:
            return self._verify_self_consistent(task, result, parsed_entities, node)
        elif task[0] == self.ACT_TO_SUBPART:
            return self._verify_to_subpart(task, result, parsed_entities, node)
        elif task[0] == self.ACT_FINAL:
            return self._verify_final(task, result, parsed_entities, node)
        raise NotImplementedError(f'unknown task: {task}')
    
    def __verify_n_feedback(self, node, parsed_entities):
        assert(node.name[-1] == '.')
        
        actions = node.actions
        for task, r in actions.items():
            feedback_data = self.verify_action(task, r, parsed_entities, node)
            feedback_data['error_task'] = ''

            if feedback_data['err_step'] != -1:
                feedback_data['error_task'] = task
                return feedback_data
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root, 'task': ''}

    def _verify_tree(self, root, parsed_entities):
        # in order
        for n in PreOrderIter(root):
            if n == root:
                continue
            feedback_data = self.__verify_n_feedback(n, parsed_entities)
            if feedback_data['err_step'] != -1:
                return feedback_data
        return self.__verify_n_feedback(root, parsed_entities)
    
    def format_feedback(self, feedback_data):
        output_sents = []
        err_node = feedback_data['node']
        if feedback_data['err_step'] == -1:
            correct_steps = f"Step (1) to ({err_node.prev_step.step_num}) are correct. The final response is also correct."
            return correct_steps
        
        # correct steps
        if err_node.step_num == '1':
            pass
        elif err_node.prev_step.step_num == '1':
            correct_steps = f"Step (1) is correct."
            output_sents.append(correct_steps)
        else:
            correct_steps = f"Step (1) to ({err_node.prev_step.step_num}) are correct."
            output_sents.append(correct_steps)
        
        if self.fb_wrong_segment:
            where_error = f"""In step ({feedback_data["err_step"]}) the part " {feedback_data["incorrect_segment"]} " is incorrect."""
        else:
            where_error = f"""There is a mistake in ({feedback_data["err_step"]})."""
        output_sents.append(where_error)

        if not self.fb_reason:
            return " ".join(output_sents)
        
        if self.fb_specific_reason:
            why_error = f"""This is because {feedback_data["reason"]}."""
        else:
            why_error = f"""This is because {feedback_data["generic_reason"]}."""
        output_sents.append(why_error)
        return " ".join(output_sents)

    def _update_error_stats(self, feedback_data):
        if feedback_data['err_step'] == -1:
            return
        error_task_name = feedback_data['error_task'][0]
        self.error_stats[error_task_name] += 1
        self.error_stats['total'] += 1
        return

    def verify_rationale(self, data: dict):
        all_steps = data['rationale'].split('\n')
        parsed_entities = self._parse_entities(data)

        tree_rationale: Node = self.to_tree_rationale(all_steps)
        
        feedback_data = self._verify_tree(tree_rationale, parsed_entities)
        self._update_error_stats(feedback_data)
        feedback = self.format_feedback(feedback_data)
        return feedback


class Scripted_WordSort_Feedback(GenerativeModel):
    def __init__(self, *args, **kwargs):
        self.verifier = WS_Verifier(*args, **kwargs)

    def prepare_input(self, data: dict):
        question = data['question'].strip()
        attempted_answer = data['attempted_answer']
        target = data['correct_answer']
        init_feedback_data = {
            "question": question,
            "rationale": attempted_answer,
            "target": target
        }
        return init_feedback_data
    
    def generate(self, input_data, **gen_kwargs):
        init_feedback_data = self.prepare_input(input_data)
        try:
            feedback = self.verifier.verify_rationale(init_feedback_data)
        except Exception as e:
            print(e)
            feedback = "[ERROR]"
        out = deepcopy(input_data)
        out['feedback'] = feedback
        if gen_kwargs.get('out_dict', False):
            return out
        return out['feedback']


if __name__ == '__main__':
    example_q_1 = """
    Sort the following words alphabetically:
    List: sioux fortescue purloin percept helmsman purcell forest
    """.replace("    ","").strip()

    example_t_1 = ['forest', 'fortescue', 'helmsman', 'percept', 'purcell', 'purloin', 'sioux']

    example_ans_1 = """
    (1) The first letter: "sioux"="s" (19), "fortescue"="f" (6), "purloin"="p" (16), "percept"="p" (16), "helmsman"="h" (8), "purcell"="p" (16), "forest"="f" (6).
    (2) We now have: (6) ["fortescue" ? "forest"] < (8) "helmsman" < (16) ["purloin" ? "percept" ? "purcell" ] < (19) "sioux".
    (3) Now let's sort this subpart ["fortescue" ? "forest"] by looking at their second letters: "fortescue"="o" (15), "forest"="o" (15).
    (3.1) We now have: (15) ["fortescue" ? "forest"].
    (3.2) Sort ["fortescue" ? "forest"] again by looking at their third letters: "fortescue"="r" (18), "forest"="r" (18).
    (3.3) We now have: (18) ["fortescue" ? "forest"].
    (3.4) Sort ["fortescue" ? "forest"] again by looking at their fourth letters: "fortescue"="t" (20), "forest"="e" (5).
    (3.5) We now have: (5) "forest" < (20) "fortescue".
    (3.6) Hence, we have "forest" < "fortescue".
    (4) Now let's sort this subpart ["purloin" ? "percept" ? "purcell"] by looking at their second letters: "purloin"="u" (21), "percept"="e" (5), "purcell"="u" (21).
    (4.1) We now have: (5) "percept" < (21) ["purloin" ? "purcell"].
    (4.2) Now let's sort this subpart ["purloin" ? "purcell"] by looking at their third letters: "purloin"="r" (18), "purcell"="r" (18).
    (4.2.1) We now have: (18) ["purloin" ? "purcell"].
    (4.2.2) Sort ["purloin" ? "purcell"] again by looking at their fourth letters: "purloin"="l" (12), "purcell"="c" (3).
    (4.2.3) We now have: (3) "purcell" < (12) "purloin".
    (4.2.4) Hence, we have "purcell" < "purloin".
    (4.3) Hence, we have "percept" < "purcell" < "purloin".
    (5) Hence, we have ["forest" < "fortescue"] < "helmsman" < ["percept" < "purcell" < "purloin"] < "sioux".
    (Final response) So the answer is: forest fortescue helmsman percept purcell purloin sioux.
    """.replace("    ","").strip()

    verifier = WS_Verifier(
        fb_wrong_segment=False,
        fb_reason=False,
        fb_specific_reason=False
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    verifier = WS_Verifier(
        fb_wrong_segment=True,
        fb_reason=False,
        fb_specific_reason=False
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    verifier = WS_Verifier(
        fb_wrong_segment=True,
        fb_reason=True,
        fb_specific_reason=False
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    verifier = WS_Verifier(
        fb_wrong_segment=True,
        fb_reason=True,
        fb_specific_reason=True
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    # wrong answer
    example_ans_1 = """
    (1) The first letter: "sioux"="s" (19), "fortescue"="f" (6), "purloin"="p" (16), "percept"="p" (16), "helmsman"="h" (8), "purcell"="p" (16), "forest"="f" (6).
    (2) We now have: (6) ["fortescue" ? "forest"] < (8) "helmsman" < (16) ["purloin" ? "percept" ? "purcell" ] < (19) "sioux".
    (3) Now let's sort this subpart ["fortescue" ? "forest"] by looking at their second letters: "fortescue"="o" (15), "forest"="o" (15).
    (3.1) We now have: (15) ["fortescue" ? "forest"].
    (3.2) Sort ["fortescue" ? "forest"] again by looking at their third letters: "fortescue"="r" (18), "forest"="r" (18).
    (3.3) We now have: (18) ["fortescue" ? "forest"].
    (3.4) Sort ["fortescue" ? "forest"] again by looking at their fourth letters: "fortescue"="t" (20), "forest"="e" (5).
    (3.5) We now have: (5) "forest" < (20) "fortescue".
    (3.6) Hence, we have "forest" < "fortescue".
    (4) Now let's sort this subpart ["purloin" ? "percept" ? "purcell"] by looking at their second letters: "purloin"="u" (21), "percept"="e" (5), "purcell"="u" (21).
    (4.1) We now have: (5) "percept" < (21) ["purloin" ? "purcell"].
    (4.2) Now let's sort this subpart ["purloin" ? "purcell"] by looking at their third letters: "purloin"="r" (18), "purcell"="r" (18).
    (4.2.1) We now have: (18) ["purloin" ? "purcell"].
    (4.2.2) Sort ["purloin" ? "purcell"] again by looking at their fifth letters: "purloin"="l" (12), "purcell"="c" (3).
    (4.2.3) We now have: (3) "purcell" < (12) "purloin".
    (4.2.4) Hence, we have "purcell" < "purloin".
    (4.3) Hence, we have "percept" < "purcell" < "purloin".
    (5) Hence, we have ["forest" < "fortescue"] < "helmsman" < ["percept" < "purcell" < "purloin"] < "sioux".
    (Final response) So the answer is: forest fortescue helmsman percept purcell purloin sioux.
    """.replace("    ","").strip()

    verifier = WS_Verifier(
        fb_wrong_segment=False,
        fb_reason=False,
        fb_specific_reason=False
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    verifier = WS_Verifier(
        fb_wrong_segment=True,
        fb_reason=False,
        fb_specific_reason=False
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    verifier = WS_Verifier(
        fb_wrong_segment=True,
        fb_reason=True,
        fb_specific_reason=False
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)

    verifier = WS_Verifier(
        fb_wrong_segment=True,
        fb_reason=True,
        fb_specific_reason=True
    )
    feedback = verifier.verify_rationale({
        'rationale': example_ans_1,
        'question': example_q_1,
        'target': example_t_1
    })
    print(feedback)