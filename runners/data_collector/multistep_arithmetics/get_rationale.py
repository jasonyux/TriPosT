from bigbench.benchmark_tasks.multistep_arithmetic.task import MultistepArithmeticTask
from utils.utils import CHARACTERS
import re
import itertools
import ast


binary_operator_classes = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**',

    ast.FloorDiv: '//',
    ast.MatMult: '@',

    ast.BitAnd: '&',
    ast.BitOr: '|',
    ast.BitXor: '^',

    ast.LShift: '<<',
    ast.RShift: '>>',
}


unary_operator_classes = {
    ast.UAdd: '+',
    ast.USub: '-',
}


unary_operator_parentheses = {
    ast.Not: 'not ',
}


comparison_operator_classes = {
    ast.Lt: '<',
    ast.Gt: '>',
    ast.LtE: '<=',
    ast.GtE: '>=',

    ast.Eq: '==',
    ast.NotEq: '!=',
}


NamedExpr = ast.NamedExpr

def _add_parentheses_recursive(branch: ast.AST, parentheses: bool) -> str:
    if isinstance(branch, ast.Constant):
        return str(branch.value)
    elif isinstance(branch, ast.Num): # In Python 3.7 and earlier, numbers are ast.Num instead of ast.Constant
        return str(branch.n)
    elif isinstance(branch, ast.Name):
        return branch.id
    elif isinstance(branch, ast.BinOp):
        result = f'{_add_parentheses_recursive(branch.left, parentheses)} {binary_operator_classes[type(branch.op)]} {_add_parentheses_recursive(branch.right, parentheses)}'
        if parentheses:
            result = f'({result})'
        return result
    elif isinstance(branch, ast.UnaryOp):
        opclass = type(branch.op)
        opvalue = unary_operator_classes.get(opclass, unary_operator_parentheses.get(opclass))
        result = f'{opvalue}{_add_parentheses_recursive(branch.operand, parentheses)}'
        if parentheses and opclass in unary_operator_parentheses:
            result = f'({result})'
        return result
    elif isinstance(branch, ast.Compare):
        result = _add_parentheses_recursive(branch.left, parentheses)
        for (op, comparator) in zip(branch.ops, branch.comparators):
            result += f' {comparison_operator_classes[type(op)]} {_add_parentheses_recursive(comparator, parentheses)}'
        if parentheses:
            result = f'({result})'
        return result
    elif isinstance(branch, ast.Call):
        result = _add_parentheses_recursive(branch.func, parentheses)
        args = ', '.join(_add_parentheses_recursive(arg, parentheses) for arg in branch.args)
        kwargs = ', '.join(f'{keyword.arg}={_add_parentheses_recursive(keyword.value, parentheses)}' for keyword in branch.keywords)
        arg_data = args
        if kwargs:
            arg_data += ', ' + kwargs
        return f'{result}({arg_data})'
    elif isinstance(branch, NamedExpr):
        result = f'{branch.target.id} := {_add_parentheses_recursive(branch.value, parentheses)}'
        if parentheses:
            result = f'({result})'
        return result
    else:
        raise ValueError(f'unsupported branch type: {branch.__class__.__name__!r}')


def add_parentheses(equ: str, parentheses: bool = True) -> str:
    """Parentheses set to False is not guaranteed to keep order of operations"""
    equation_ast = ast.parse(equ, '<expression>', 'eval')
    return _add_parentheses_recursive(equation_ast.body, parentheses)


def substitute_first_step(input_question):
	all_subparts = re.findall(r'\(([^\(\)]+)\)', input_question)
	substituted_question = input_question
	mapping = {}
	for i, subpart in enumerate(all_subparts):
		subpart = f'({subpart})'
		substituted_question = substituted_question.replace(subpart, CHARACTERS[i])
		mapping[CHARACTERS[i]] = subpart
	
	step = f'This equation can be written as "{substituted_question}", where'
	for i in range(len(all_subparts)):
		step += f' {CHARACTERS[i]} = {mapping[CHARACTERS[i]]}'
		if i == len(all_subparts) - 2:
			step += ' and'
		elif i < len(all_subparts) - 2:
			step += ','
	step += '.'
	return step, all_subparts, substituted_question

def __solve_parantheses(question_w_paranthesis: str):
	all_subparts = re.findall(r'\(([^\(\)]+)\)', question_w_paranthesis)
	if len(all_subparts) == 0:
		return []
	all_substeps = []
	new_progress = question_w_paranthesis
	for subpart in all_subparts:
		subpart = f'({subpart})'
		sol = eval(subpart)
		new_progress = new_progress.replace(subpart, str(sol))
		all_substeps.append(new_progress)
	new_progress_w_p = add_parentheses(new_progress, parentheses=True)
	return all_substeps + __solve_parantheses(new_progress_w_p)

def calculate_subpart(subpart, placeholder):
	# bracket the ones that have multiplication and division
	subpart_added_p = add_parentheses(subpart, parentheses=True)
	step = f"Let's calculate {placeholder} = ({subpart}) = {subpart_added_p}"
	solving_steps = __solve_parantheses(subpart_added_p)
	for s in solving_steps:
		step += f' = {s}'
	step += '.'
	return step, eval(subpart_added_p)

def calculate_final_response(substituted_question, all_subparts):
	# calculate the final response
	step = f'Then, the final equation is {substituted_question} = '
	substituted_question_w_value = substituted_question
	for k, sol in all_subparts.items():
		substituted_question_w_value = substituted_question_w_value.replace(k, str(sol))
	step += substituted_question_w_value
	solving_steps = __solve_parantheses(substituted_question_w_value)
	for s in solving_steps:
		step += f' = {s}'
	step += '.'
	return step

def solve(input_question: str):
	""" return a rationale given a list of words to sort.
	For example
	```txt
	(1) This equation can be written as "A * B", where A = (-5 + 9 * -4 - 0) and B = (4 + -7 + 0 * -5).
	(2) Let's calculate A = (-5 + 9 * -4 - 0) = (-5 + (9 * -4) - 0) = (-5 + (-36) - 0) = (-5 + -36 - 0) = -5 - 36 = -41.
	(3) Let's calculate B = (4 + -7 + 0 * -5) = (4 + -7 + (0 * -5)) = (4 + -7 + 0) = (4 + -7) = (4 - 7) = -3.
	(4) Then, the final equation is A * B = -41 * -3 = (-41 * -3) = 123.
	(Final response) So the answer is 123.
	```
	"""
	solution = []
	# first step sort first letter and generate subparts to sort
	step, subparts_to_calculate, substituted_question = substitute_first_step(input_question)
	solution.append(f'(1) {step}')

	# calculate subparts
	subpart_solutions = {}
	for i, subpart in enumerate(subparts_to_calculate):
		placeholder = CHARACTERS[i]
		step, subpart_sol = calculate_subpart(subpart, placeholder)
		solution.append(f'({i+2}) {step}')
		subpart_solutions[placeholder] = subpart_sol

	step = calculate_final_response(substituted_question, subpart_solutions)
	solution.append(f'({len(subparts_to_calculate) + 2}) {step}')

	# add final response
	final_ans = eval(input_question)
	final_response = f"(Final response) So the answer is {final_ans}."
	solution.append(final_response)

	assert(f'= {final_ans}.' in solution[-2])
	return "\n".join(solution)

if __name__ == '__main__':
    task = MultistepArithmeticTask(
        num_trials = 300,
        operations =['+', '-', '*'],
        lengths = [3, 4, 5, 6],
        depth_level_list = ([2], [2, 2], [2, 2, 2])
    )
    print(f'{task.lengths=}, {task.depth_level_list=}')

    all_questions = []

    for depth_levels, length in itertools.product(
        task.depth_level_list, task.lengths
    ):
        print(f'{depth_levels=}, {length=}')
        for _ in range(task.num_trials):
            correct = False
            input = task.generate_string(depth_levels=depth_levels, length=length)
            problem = input + " = "

            all_questions.append({
                "question": problem,
                "input": input,
                "target": str(eval(input)),
                "depth_levels": depth_levels,
                "length": length,
            })

    print(len(all_questions), all_questions[0]['question'], all_questions[0]['target'])
    print(len(all_questions), all_questions[-1]['question'], all_questions[-1]['target'])

    print('\n\n\n')
    solved = solve(all_questions[-1]['input'])
    print(all_questions[-1]['input'])
    print(solved)