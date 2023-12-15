# this should only be used for testing LLMs performance, as this task is SCRIPTED
EXAMPLES_MULTISTEP_ARITHMETIC_RATIONALE = """
Q: (((-9 * 3 + -6) * (-2 + 7 * -6)) - ((-4 - -6 + 6) * (4 * -4 + 8))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A * B) - (C * D))", where A = (-9 * 3 + -6), B = (-2 + 7 * -6), C = (-4 - -6 + 6) and D = (4 * -4 + 8).
(2) Let's calculate A = (-9 * 3 + -6) = ((-9 * 3) + -6) = (-27 + -6) = -33.
(3) Let's calculate B = (-2 + 7 * -6) = (-2 + (7 * -6)) = (-2 + -42) = -44.
(4) Let's calculate C = (-4 - -6 + 6) = ((-4 - -6) + 6) = (2 + 6) = 8.
(5) Let's calculate D = (4 * -4 + 8) = ((4 * -4) + 8) = (-16 + 8) = -8.
(6) Then, the final equation is ((A * B) - (C * D)) = ((-33 * -44) - (8 * -8)) = (1452 - (8 * -8)) = (1452 - -64) = 1516.
(Final response) So the answer is 1516. [END]
Q: (((-5 - 6 * 0) - (-8 * -6 + 6)) - ((-3 + 2 * 7) + (-2 * -4 - 5))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A - B) - (C + D))", where A = (-5 - 6 * 0), B = (-8 * -6 + 6), C = (-3 + 2 * 7) and D = (-2 * -4 - 5).
(2) Let's calculate A = (-5 - 6 * 0) = (-5 - (6 * 0)) = (-5 - 0) = -5.
(3) Let's calculate B = (-8 * -6 + 6) = ((-8 * -6) + 6) = (48 + 6) = 54.
(4) Let's calculate C = (-3 + 2 * 7) = (-3 + (2 * 7)) = (-3 + 14) = 11.
(5) Let's calculate D = (-2 * -4 - 5) = ((-2 * -4) - 5) = (8 - 5) = 3.
(6) Then, the final equation is ((A - B) - (C + D)) = ((-5 - 54) - (11 + 3)) = (-59 - (11 + 3)) = (-59 - 14) = -73.
(Final response) So the answer is -73. [END]
Q: ((-9 + -6 + 8 * 1) * (-9 - -1 * -6 - 6)) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "(A * B)", where A = (-9 + -6 + 8 * 1) and B = (-9 - -1 * -6 - 6).
(2) Let's calculate A = (-9 + -6 + 8 * 1) = ((-9 + -6) + (8 * 1)) = (-15 + (8 * 1)) = (-15 + 8) = -7.
(3) Let's calculate B = (-9 - -1 * -6 - 6) = ((-9 - (-1 * -6)) - 6) = ((-9 - 6) - 6) = (-15 - 6) = -21.
(4) Then, the final equation is (A * B) = (-7 * -21) = 147.
(Final response) So the answer is 147. [END]
""".strip()


EXAMPLES_MULTISTEP_ARITHMETIC_FEEDBACK_NOCORRECT_TABULAR = """
Q: (((-9 * 3 + -6) * (-2 + 7 * -6)) - ((-4 - -6 + 6) * (4 * -4 + 8))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A * B) - (C * D))", where A = (-9 * 3 + -6), B = (-2 + 7 * -6), C = (-4 - -6 + 6) and D = (4 * -4 + 8).
(2) Let's calculate A = (-9 * 3 + -6) = ((-9 * 3) + -6) = (-27 + -6) = -33.
(3) Let's calculate B = (-2 + 7 * -6) = (-2 + (7 * -6)) = (-2 + -42) = -44.
(4) Let's calculate C = (-4 - -6 + 6) = ((-4 - -6) + 6) = (2 + 6) = 8.
(5) Let's calculate D = (4 * -4 + 8) = ((4 * -4) + 8) = (-16 + 8) = -8.
(6) Then, the final equation is ((A * B) - (C * D)) = ((-33 * -44) - (8 * -8)) = (1424 - (8 * -8)) = (1424 - 64) = 1360.
(Final response) So the answer is 1360.
Earliest error step: (6)
Error segment: "((-33 * -44) - (8 * -8)) = (1424 - (8 * -8))"
Error reason: There is a calculation error, since (-33 * -44) should be 1452. [END]
Q: (((-5 - 6 * 0) - (-8 * -6 + 6)) - ((-3 + 2 * 7) + (-2 * -4 - 5))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A - B) - (C + D))", where A = (-5 - 6 * 0), B = (-8 * -6 + 6), C = (-3 + 2 * 7) and D = (-2 * -4 - 5).
(2) Let's calculate A = (-5 - 6 * 0) = (-5 - (6 * 0)) = (-5 - 0) = -5.
(3) Let's calculate B = (-8 * -6 + 6) = ((-8 * -6) + 6) = (48 + 6) = 54.
(4) Let's calculate C = (-3 + 2 * 7) = (-3 + (2 * 7)) = (-3 + 14) = 11.
(5) Let's calculate D = (-2 * -4 - 5) = ((-2 * -4) - 5) = (8 - 5) = 3.
(6) Then, the final equation is ((A - B) - (C + D)) = ((-5 - 54) - (11 + 3)) = (-69 - (11 + 3)) = (-69 - 14) = -83.
(Final response) So the answer is -83.
Earliest error step: (6)
Error segment: "((-5 - 54) - (11 + 3)) = (-69 - (11 + 3))"
Error reason: There is a calculation error, since (-5 - 54) should be -59. [END]
Q: ((-9 + -6 + 8 * 1) * (-9 - -1 * -6 - 6)) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "(A * B)", where A = (-9 + -6 + 8 * 1) and B = (-9 - -1 * -6 - 6).
(2) Let's calculate A = (-9 + -6 + 8 * 1) = ((-9 + -6) + 8 * 1) = (-15 + 8 * 1) = -3.
(3) Let's calculate B = (-9 - -1 * -6 - 6) = ((-9 - (-1 * -6) - 6) = (-9 - 6) = -15.
(4) Then, the final equation is (A * B) = (-3 * -15) = 45.
(Final response) So the answer is 45.
Earliest error step: (2)
Error segment: "(-15 + 8 * 1) = -3"
Error reason: Multiplication and division should be done first, so (-15 + 8 * 1) = (-15 + (8 * 1)) = (-15 + 8) = -7. [END]
""".strip()


EXAMPLES_MULTISTEP_ARITHMETIC_FEEDBACK_HASCORRECT_TABULAR = """
Q: (((-9 * 3 + -6) * (-2 + 7 * -6)) - ((-4 - -6 + 6) * (4 * -4 + 8))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A * B) - (C * D))", where A = (-9 * 3 + -6), B = (-2 + 7 * -6), C = (-4 - -6 + 6) and D = (4 * -4 + 8).
(2) Let's calculate A = (-9 * 3 + -6) = ((-9 * 3) + -6) = (-27 + -6) = -33.
(3) Let's calculate B = (-2 + 7 * -6) = (-2 + (7 * -6)) = (-2 + -42) = -44.
(4) Let's calculate C = (-4 - -6 + 6) = ((-4 - -6) + 6) = (2 + 6) = 8.
(5) Let's calculate D = (4 * -4 + 8) = ((4 * -4) + 8) = (-16 + 8) = -8.
(6) Then, the final equation is ((A * B) - (C * D)) = ((-33 * -44) - (8 * -8)) = (1424 - (8 * -8)) = (1424 - 64) = 1360.
(Final response) So the answer is 1360.
Earliest error step: (6)
Error segment: "((-33 * -44) - (8 * -8)) = (1424 - (8 * -8))"
Error reason: There is a calculation error, since (-33 * -44) should be 1452. [END]
Q: (((-5 - 6 * 0) - (-8 * -6 + 6)) - ((-3 + 2 * 7) + (-2 * -4 - 5))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A - B) - (C + D))", where A = (-5 - 6 * 0), B = (-8 * -6 + 6), C = (-3 + 2 * 7) and D = (-2 * -4 - 5).
(2) Let's calculate A = (-5 - 6 * 0) = (-5 - (6 * 0)) = (-5 - 0) = -5.
(3) Let's calculate B = (-8 * -6 + 6) = ((-8 * -6) + 6) = (48 + 6) = 54.
(4) Let's calculate C = (-3 + 2 * 7) = (-3 + (2 * 7)) = (-3 + 14) = 11.
(5) Let's calculate D = (-2 * -4 - 5) = ((-2 * -4) - 5) = (8 - 5) = 3.
(6) Then, the final equation is ((A - B) - (C + D)) = ((-5 - 54) - (11 + 3)) = (-59 - (11 + 3)) = (-59 - 14) = -73.
(Final response) So the answer is -73.
Earliest error step: None
Error segment: None
Error reason: The final response is correct. [END]
Q: ((-9 + -6 + 8 * 1) * (-9 - -1 * -6 - 6)) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "(A * B)", where A = (-9 + -6 + 8 * 1) and B = (-9 - -1 * -6 - 6).
(2) Let's calculate A = (-9 + -6 + 8 * 1) = ((-9 + -6) + 8 * 1) = (-15 + 8 * 1) = -3.
(3) Let's calculate B = (-9 - -1 * -6 - 6) = ((-9 - (-1 * -6) - 6) = (-9 - 6) = -15.
(4) Then, the final equation is (A * B) = (-3 * -15) = 45.
(Final response) So the answer is 45.
Earliest error step: (2)
Error segment: "(-15 + 8 * 1) = -3"
Error reason: Multiplication and division should be done first, so (-15 + 8 * 1) = (-15 + (8 * 1)) = (-15 + 8) = -7. [END]
""".strip()

EXAMPLES_MULTISTEP_ARITHMETIC_UPDATE = """
Q: (((-9 * 3 + -6) * (-2 + 7 * -6)) - ((-4 - -6 + 6) * (4 * -4 + 8))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A * B) - (C * D))", where A = (-9 * 3 + -6), B = (-2 + 7 * -6), C = (-4 - -6 + 6) and D = (4 * -4 + 8).
(2) Let's calculate A = (-9 * 3 + -6) = ((-9 * 3) + -6) = (-27 + -6) = -33.
(3) Let's calculate B = (-2 + 7 * -6) = (-2 + (7 * -6)) = (-2 + -42) = -44.
(4) Let's calculate C = (-4 - -6 + 6) = ((-4 - -6) + 6) = (2 + 6) = 8.
(5) Let's calculate D = (4 * -4 + 8) = ((4 * -4) + 8) = (-16 + 8) = -8.
(6) Then, the final equation is ((A * B) - (C * D)) = ((-33 * -44) - (8 * -8)) = (1424 - (8 * -8)) = (1424 - 64) = 1360.
(Final response) So the answer is 1360.
Feedback: Step (1) to step (5) are correct. In step (6) the part "((-33 * -44) - (8 * -8)) = (1424 - (8 * -8))" is incorrect. This is because there is a calculation error, since (-33 * -44) should be 1452.
Updated Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A * B) - (C * D))", where A = (-9 * 3 + -6), B = (-2 + 7 * -6), C = (-4 - -6 + 6) and D = (4 * -4 + 8).
(2) Let's calculate A = (-9 * 3 + -6) = ((-9 * 3) + -6) = (-27 + -6) = -33.
(3) Let's calculate B = (-2 + 7 * -6) = (-2 + (7 * -6)) = (-2 + -42) = -44.
(4) Let's calculate C = (-4 - -6 + 6) = ((-4 - -6) + 6) = (2 + 6) = 8.
(5) Let's calculate D = (4 * -4 + 8) = ((4 * -4) + 8) = (-16 + 8) = -8.
(6) Then, the final equation is ((A * B) - (C * D)) = ((-33 * -44) - (8 * -8)) = (1452 - (8 * -8)) = (1452 - -64) = 1516.
(Final response) So the answer is 1516. [END]
Q: (((-5 - 6 * 0) - (-8 * -6 + 6)) - ((-3 + 2 * 7) + (-2 * -4 - 5))) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A - B) - (C + D))", where A = (-5 - 6 * 0), B = (-8 * -6 + 6), C = (-3 + 2 * 7) and D = (-2 * -4 - 5).
(2) Let's calculate A = (-5 - 6 * 0) = (-5 - (6 * 0)) = (-5 - 0) = -5.
(3) Let's calculate B = (-8 * -6 + 6) = ((-8 * -6) + 6) = (48 + 6) = 54.
(4) Let's calculate C = (-3 + 2 * 7) = (-3 + (2 * 7)) = (-3 + 14) = 11.
(5) Let's calculate D = (-2 * -4 - 5) = ((-2 * -4) - 5) = (8 - 5) = 3.
(6) Then, the final equation is ((A - B) - (C + D)) = ((-5 - 54) - (11 + 3)) = (-69 - (11 + 3)) = (-69 - 14) = -83.
(Final response) So the answer is -83.
Feedback: Step (1) to step (5) are correct. In step (6) the part "((-5 - 54) - (11 + 3)) = (-69 - (11 + 3))" is incorrect. This is because there is a calculation error, since (-5 - 54) should be -59.
Updated Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "((A - B) - (C + D))", where A = (-5 - 6 * 0), B = (-8 * -6 + 6), C = (-3 + 2 * 7) and D = (-2 * -4 - 5).
(2) Let's calculate A = (-5 - 6 * 0) = (-5 - (6 * 0)) = (-5 - 0) = -5.
(3) Let's calculate B = (-8 * -6 + 6) = ((-8 * -6) + 6) = (48 + 6) = 54.
(4) Let's calculate C = (-3 + 2 * 7) = (-3 + (2 * 7)) = (-3 + 14) = 11.
(5) Let's calculate D = (-2 * -4 - 5) = ((-2 * -4) - 5) = (8 - 5) = 3.
(6) Then, the final equation is ((A - B) - (C + D)) = ((-5 - 54) - (11 + 3)) = (-59 - (11 + 3)) = (-59 - 14) = -73.
(Final response) So the answer is -73. [END]
Q: ((-9 + -6 + 8 * 1) * (-9 - -1 * -6 - 6)) =
Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "(A * B)", where A = (-9 + -6 + 8 * 1) and B = (-9 - -1 * -6 - 6).
(2) Let's calculate A = (-9 + -6 + 8 * 1) = ((-9 + -6) + 8 * 1) = (-15 + 8 * 1) = -3.
(3) Let's calculate B = (-9 - -1 * -6 - 6) = ((-9 - (-1 * -6) - 6) = (-9 - 6) = -15.
(4) Then, the final equation is (A * B) = (-3 * -15) = 45.
(Final response) So the answer is 45.
Feedback: Step (1) is correct. In step (2) the part "(-15 + 8 * 1) = -3" is incorrect. This is because multiplication and division should be done first, so (-15 + 8 * 1) = (-15 + (8 * 1)) = (-15 + 8) = -7.
Updated Answer: Let's think step by step. Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first.
(1) This equation can be written as "(A * B)", where A = (-9 + -6 + 8 * 1) and B = (-9 - -1 * -6 - 6).
(2) Let's calculate A = (-9 + -6 + 8 * 1) = ((-9 + -6) + (8 * 1)) = (-15 + (8 * 1)) = (-15 + 8) = -7.
(3) Let's calculate B = (-9 - -1 * -6 - 6) = ((-9 - (-1 * -6)) - 6) = ((-9 - 6) - 6) = (-15 - 6) = -21.
(4) Then, the final equation is (A * B) = (-7 * -21) = 147.
(Final response) So the answer is 147. [END]
""".strip()