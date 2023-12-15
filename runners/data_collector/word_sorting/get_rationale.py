from datasets import load_dataset


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


def _sort_by_nth_letter(words, n):
	words_w_idx = []
	for word in words:
		char_idx = ord(word[n]) - ord('a') + 1
		words_w_idx.append(f'"{word}"="{word[n]}" ({char_idx})')
	return ", ".join(words_w_idx)

def _sort_and_group_by_nth_letter(words, n):
	group_by_idx = {}
	for word in words:
		char_idx = ord(word[n]) - ord('a') + 1
		if char_idx not in group_by_idx:
			group_by_idx[char_idx] = []
		group_by_idx[char_idx].append(word)
	sorted_keys = sorted(group_by_idx.keys())
	
	subpart_w_idx = []
	subparts_to_sort = []
	for idx in sorted_keys:
		word_list = group_by_idx[idx]
		if len(word_list) > 1:
			s = " ? ".join([f'"{w}"' for w in word_list])
			s = f'({idx}) [{s}]'
			subparts_to_sort.append(word_list)
		else:
			w = word_list[0]
			s = f'({idx}) "{w}"'
		subpart_w_idx.append(s)
	return " < ".join(subpart_w_idx), subparts_to_sort

def _final_sort_step(words):
	# the order is correct regardless which letter we check
	sorted_by_nth_letter = sorted(words)
	return " < ".join([f'"{w}"' for w in sorted_by_nth_letter])

def sort_by_first_letter(question: list):
	# first letter: "sioux"="s" (19), "fortescue"="f" (6)
	first_step = "(1) The first letter: "
	first_step += _sort_by_nth_letter(question, 0) + "."

	# genreate subparts to solve
	we_now_have_step = "(2) We now have: "
	sorted_parts_text, subparts_to_sort = _sort_and_group_by_nth_letter(question, 0)
	we_now_have_step += sorted_parts_text + "."
	return [first_step, we_now_have_step], subparts_to_sort

def solve_subpart(subpart: list, letter_idx_to_sort: int, substep_prefix: str, substep_idx: int):
	"""
	example: ["purloin" ? "percept" ? "purcell"]
	(4) Now sort this subpart ["purloin" ? "percept" ? "purcell"] by looking at their second letters: "purloin"="u" (21), "percept"="e" (5), "purcell"="u" (21).
	(4.1) We now have: (5) "percept" < (21) ["purloin" ? "purcell"].
	(4.2) Now sort this subpart ["purloin" ? "purcell"] by looking at their third letters: "purloin"="r" (18), "purcell"="r" (18).
	(4.2.1) We now have: (18) ["purloin" ? "purcell"].
	(4.2.2) Now sort this subpart ["purloin" ? "purcell"] again by looking at their fourth letters: "purloin"="l" (12), "purcell"="c" (3).
	(4.2.3) We now have: (3) "purcell" < (12) "purloin".
	(4.2.4) Hence, we have "purcell" < "purloin".
	(4.3) Hence, we have "percept" < "purcell" < "purloin".
	"""
	steps = []

	sort_words = " ? ".join([f'"{w}"' for w in subpart])
	sort_words = f'[{sort_words}]'
	now_lets_sort_step = f"({substep_prefix}) Now sort this subpart {sort_words} by looking at their {NUM_TO_ORDER[letter_idx_to_sort]} letters: "
	now_lets_sort_step += _sort_by_nth_letter(subpart, letter_idx_to_sort) + "."
	steps.append(now_lets_sort_step)
	
	# indent
	sorted_parts_text, subparts_to_sort = _sort_and_group_by_nth_letter(subpart, letter_idx_to_sort)
	we_now_have_step = f"({substep_prefix}.{substep_idx}) We now have: "
	we_now_have_step += sorted_parts_text + "."
	steps.append(we_now_have_step)

	local_letter_idx_to_sort = letter_idx_to_sort + 1
	local_substep_idx = substep_idx
	while len(subparts_to_sort) == 1 and (subparts_to_sort[0] == subpart):
		local_substep_idx += 1
		# don't indent if we are solving the same subpart
		# sort without indenting
		sort_words = " ? ".join([f'"{w}"' for w in subpart])
		sort_words = f'[{sort_words}]'
		now_lets_sort_step = f"({substep_prefix}.{local_substep_idx}) Now sort this subpart {sort_words} by looking at their {NUM_TO_ORDER[local_letter_idx_to_sort]} letters: "
		now_lets_sort_step += _sort_by_nth_letter(subpart, local_letter_idx_to_sort) + "."
		steps.append(now_lets_sort_step)
		
		local_substep_idx += 1

		sorted_parts_text, subparts_to_sort = _sort_and_group_by_nth_letter(subpart, local_letter_idx_to_sort)
		we_now_have_step = f"({substep_prefix}.{local_substep_idx}) We now have: "
		we_now_have_step += sorted_parts_text + "."
		steps.append(we_now_have_step)

		local_letter_idx_to_sort += 1
	
	final_local_substep_idx = local_substep_idx
	for i, subsubpart in enumerate(subparts_to_sort):
		substeps = solve_subpart(subsubpart, local_letter_idx_to_sort, f'{substep_prefix}.{local_substep_idx+1}', 1)
		steps.extend(substeps)
		final_local_substep_idx += 1

	# hence step
	hence_step = _final_sort_step(subpart)
	hence_step = f'({substep_prefix}.{final_local_substep_idx+1}) Hence, we have {hence_step}.'
	steps.append(hence_step)
	return steps

def solve(question: list):
	# return a rationale given a list of words to sort
	solution = []
	# first step sort first letter and generate subparts to sort
	steps, subparts = sort_by_first_letter(question)
	solution.extend(steps)

	for i, subpart in enumerate(subparts):
		steps = solve_subpart(
			subpart, 
			letter_idx_to_sort=1,
			substep_prefix=f'{i+3}',
			substep_idx=1
		)
		solution.extend(steps)

	# assume all the previous are correct
	last_step = _final_sort_step(question)
	last_step = f'({len(subparts)+3})' + " Hence, we have " + last_step + "."
	solution.append(last_step)

	# add final response
	final_response = f"(Final response) So the answer is: {' '.join(sorted(question))}."
	solution.append(final_response)
	return "\n".join(solution)


if __name__ == '__main__':
    question = "sioux fortescue purloin percept helmsman purcell forest"
    solved = solve(question.split())
    print(question)
    print(solved)

    # all_data = load_dataset("bigbench", "word_sorting")