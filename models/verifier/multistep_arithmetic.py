import re

from copy import deepcopy
from collections import OrderedDict
from anytree import Node, PreOrderIter, RenderTree
from models.base import GenerativeModel
from models.verifier.base import ScriptedVerifier


class MSA_Verifier(ScriptedVerifier):
    ACT_PARTITION = 'partition'  # partition the question
    ACT_CALCULATE = 'calculate'  # check if the calculation is correct
    ACT_COMBINE = 'combine'  # combine the results
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
            self.ACT_PARTITION: 0,
            self.ACT_CALCULATE: 0,
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
        entities = re.findall(r'\((.*)\) =', question)[0]
        assert(float(eval(entities)) == float(targets))

        return entities
    
    def _parse_segment(self, segment: str):
        if segment[-1] != '.':
            segment += '.'

        if "this equation can be written as" in segment.lower():
            actions = {
                self.ACT_PARTITION: segment
            }
        elif "let's calculate" in segment.lower():
            actions = {
                self.ACT_CALCULATE: segment
            }
        elif "the final equation is" in segment:
            actions = OrderedDict([
                (self.ACT_COMBINE, segment),  # check this first
                (self.ACT_CALCULATE, segment),
            ])
        elif "So the answer is" in segment:
            actions = {
                self.ACT_FINAL: segment
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
            extracted_step = re.findall(r'^\((\d+)\) \w.+', step)[0]
            
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

    def _verify_partition(self, task, result, parsed_entities, node):
        partitioned_entities = re.findall(r'([A-Z]) = (\([^a-zA-Z]+\))', result)
        if len(partitioned_entities) < 2:
            raise ValueError(f'Cannot parse partitioned entities: {result} into more than 2 parts.')
        proposed_new_eq = re.search(r'can be written as "(.*)",', result)
        if proposed_new_eq is None:
            raise ValueError(f'Cannot parse the proposed new equation: {result}')
        
        new_eq = proposed_new_eq.group(1)
        for var, eq in partitioned_entities:
            if var not in new_eq:
                raise ValueError(f'Cannot find {var} in the new equation: {new_eq}')
            new_eq = new_eq.replace(var, eq)
        if eval(new_eq) != eval(parsed_entities):
            return {
                'err_step': node.step_num,
                'incorrect_segment': result[4:].strip(),
                'reason': f'it is not consistent with the original equation: {parsed_entities}',
                'generic_reason': f'it is not consistent with the original equation',
                'node': node
            }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def __get_equation_part_to_verify(self, result):
        if "let's calculate" in result.lower():
            equation_part = re.search(r'calculate [A-Z] = (.+)\.', result)
            if equation_part is None:
                raise ValueError(f'Cannot parse the equations inside: {result}')
            return equation_part.group(1)
        elif "final equation is" in result.lower():
            equation_part = re.search(r'final equation is (.+)\.', result)
            if equation_part is None:
                raise ValueError(f'Cannot parse the equations inside: {result}')
            return equation_part.group(1)
        else:
            raise ValueError(f'Unknown format. Cannot parse the equations inside: {result}')

    def __check_valid_equation(self, equation, node):
        try:
            int(eval(equation))
        except:
            num_lbracket = equation.count('(')
            num_rbracket = equation.count(')')
            extra_reason = ''
            if num_lbracket != num_rbracket:
                extra_reason = ' as the number of left and right brackets do not match'
            return {
                'err_step': node.step_num,
                'incorrect_segment': f'{equation}',
                'reason': f'it is an invalid equation' + extra_reason,
                'generic_reason': f'it is an invalid equation',
                'node': node
            }
        return None
    
    def _verify_calculate(self, task, result, parsed_entities, node):
        equation_part = self.__get_equation_part_to_verify(result)
        _equations_to_check = equation_part.split("=")
        equations_to_check = [
            eq.strip() for eq in _equations_to_check if eq.strip() != '' and re.search(r'[A-Z]', eq) is None
        ]
        for i in range(len(equations_to_check) - 1):
            lhs = equations_to_check[i].strip()
            rhs = equations_to_check[i+1].strip()

            __check_valid_equation_lhs = self.__check_valid_equation(lhs, node)
            if __check_valid_equation_lhs is not None:
                return __check_valid_equation_lhs
            lhs_num = int(eval(lhs))

            __check_valid_equation_rhs = self.__check_valid_equation(rhs, node)
            if __check_valid_equation_rhs is not None:
                return __check_valid_equation_rhs
            rhs_num = int(eval(rhs))

            if lhs_num != rhs_num:
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': f'{lhs} = {rhs}',
                    'reason': f'there is an calculation error, since {lhs} is not equal to {rhs}',
                    'generic_reason': f'there is an calculation error',
                    'node': node
                }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}
    
    def _verify_combine(self, task, result, parsed_entities, node):
        equation_part = self.__get_equation_part_to_verify(result)
        if len(equation_part.split("=")) < 2:
            raise ValueError(f'Cannot extract the equation to subsitiute in the result: {result}')
        eq_to_sub, subbed_eq = equation_part.split("=")[:2]
        eq_to_sub = eq_to_sub.strip()
        subbed_eq = subbed_eq.strip()
        if re.search(r'[A-Z]', eq_to_sub) is None:
            raise ValueError(f'Cannot find the variable to substitute in the result: {result}')
        all_variables = re.findall(r'[A-Z]', eq_to_sub)

        # 1. check if the subbed equations is correct
        all_found_mapping = {}
        prev_node = node.prev_step
        while prev_node is not None:
            prev_step = prev_node.name
            for var in all_variables:
                if f"calculate {var} = " in prev_step:
                    prev_equation_part = self.__get_equation_part_to_verify(prev_step)
                    _prev_equations_to_check = prev_equation_part.split("=")
                    prev_equations_to_check = [
                        eq.strip() for eq in _prev_equations_to_check if eq.strip() != '' and re.search(r'[A-Z]', eq) is None
                    ]
                    prev_final_result = prev_equations_to_check[-1]
                    all_found_mapping[var] = prev_final_result
            prev_node = prev_node.prev_step

        for var in all_variables:
            if var not in all_found_mapping:
                return {
                    'err_step': node.step_num,
                    'incorrect_segment': f'{eq_to_sub} = {subbed_eq}',
                    'reason': f'{var} has not been calculated in previous steps',
                    'generic_reason': f'{var} has not been calculated in previous steps',
                    'node': node
                }
        
        new_subbed_eq = eq_to_sub
        for var, val in all_found_mapping.items():
            new_subbed_eq = new_subbed_eq.replace(var, val)
        
        __check_new_subbed_eq = self.__check_valid_equation(new_subbed_eq, node)
        if __check_new_subbed_eq is not None:
            return __check_new_subbed_eq
        new_subbed_eq_num = int(eval(new_subbed_eq))

        __check_subbed_eq = self.__check_valid_equation(subbed_eq, node)
        if __check_subbed_eq is not None:
            return __check_subbed_eq
        subbed_eq_num = int(eval(subbed_eq))

        if new_subbed_eq_num != subbed_eq_num:
            return {
                'err_step': node.step_num,
                'incorrect_segment': f'{eq_to_sub} = {subbed_eq}',
                'reason': f'this substitution is incosistent with the results in the previous steps',
                'generic_reason': f'this substitution is incosistent with the results in the previous steps',
                'node': node
            }

        # 2. if the subbed one is correct, check if the eval answer is the same as solution
        # Because all prev steps must be correct, if incorrect it means @eq_to_sub is incorrect
        correct_solution = int(eval(parsed_entities))
        if subbed_eq_num != correct_solution:
            return {
                'err_step': node.step_num,
                'incorrect_segment': f'{eq_to_sub}',
                'reason': f'this equation is inconsistent with the question statement',
                'generic_reason': f'this equation is inconsistent with the question statement',
                'node': node
            }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}

    def _verify_final(self, task, result, parsed_entities, node):
        # check if the final result is correct
        extracted_answer = re.search(r'the answer is (-?\d+)\.', result)
        if extracted_answer is None:
            raise ValueError(f'Cannot extract the answer from the result: {result}')
        answer = int(extracted_answer.group(1))

        prev_step = node.prev_step.name
        prev_equation_part = self.__get_equation_part_to_verify(prev_step)
        _prev_equations_to_check = prev_equation_part.split("=")
        prev_equations_to_check = [
            eq.strip() for eq in _prev_equations_to_check if eq.strip() != '' and re.search(r'[A-Z]', eq) is None
        ]
        prev_final_result = prev_equations_to_check[-1]
        prev_final_result_num = int(prev_final_result)

        if answer != prev_final_result_num:
            return {
                'err_step': node.step_num,
                'incorrect_segment': f'the answer is {answer}.',
                'reason': f'this answer is inconsistent with the result in step ({node.prev_step.step_num})',
                'generic_reason': f'this answer is inconsistent with the result in step ({node.prev_step.step_num})',
                'node': node
            }
        return {'err_step': -1, 'incorrect_segment': '', 'reason': '', 'node': node.root}
    
    def verify_action(self, task, result, parsed_entities, node):
        if task == self.ACT_COMBINE:
            return self._verify_combine(task, result, parsed_entities, node)
        elif task == self.ACT_CALCULATE:
            return self._verify_calculate(task, result, parsed_entities, node)
        elif task == self.ACT_PARTITION:
            return self._verify_partition(task, result, parsed_entities, node)
        elif task == self.ACT_FINAL:
            return self._verify_final(task, result, parsed_entities, node)
        raise NotImplementedError(f'unknown task: {task}')
    
    def __verify_n_feedback(self, node, parsed_entities):
        assert(node.name[-1] == '.')
        
        actions = node.actions
        for task, segment in actions.items():
            feedback_data = self.verify_action(task, segment, parsed_entities, node)
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
        error_task_name = feedback_data['error_task']
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


class Scripted_MultistepArithmetic_Feedback(GenerativeModel):
    def __init__(self, *args, **kwargs):
        self.verifier = MSA_Verifier(*args, **kwargs)

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