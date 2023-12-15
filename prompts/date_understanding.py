EXAMPLES_DATE_UNDERSTANDING_RATIONALE = """
Q: Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
Answer: Let's think step by step.
(1) If today is Christmas Eve of 1937, then today's date is December 24, 1937.
(2) 10 days before today is December 14, 1937, that is 12/14/1937.
(Final response) So the answer is (D). [END]
Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
Answer: Let's think step by step.
(1) If tomorrow is 11/12/2019, then today is 11/11/2019.
(2) The date one year ago from today is 11/11/2018.
(Final response) So the answer is (B). [END]
Q: Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 01/11/1961
(B) 01/03/1963
(C) 01/18/1961
(D) 10/14/1960
(E) 01/03/1982
(F) 12/03/1960
Answer: Let's think step by step.
(1) If Jane and John married on Jan 2, 1958 and today is their 5-year anniversary, then today's date is Jan 2, 1963.
(2) The date tomorrow is Jan 3, 1963, that is 01/03/1963.
(Final response) So the answer is (B). [END]
""".strip()

EXAMPLES_DATE_UNDERSTANDING_UPDATE = """
Q: Yesterday was 12/31/1929. Today could not be 12/32/1929 because December has only 31 days. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 12/12/1929
(B) 01/01/1930
(C) 01/02/1998
(D) 01/02/1885
(E) 01/02/1930
(F) 12/23/1929
Answer: Let's think step by step.
(1) If yesterday was 12/31/1929, then today is 01/01/1930.
(2) The date tomorrow is 01/02/1930.
(Final response) So the answer is (F).
Feedback: Step (1) to step (2) are correct. In step (Final response) the part "the answer is (F)" is incorrect. This is because (F) 12/23/1929 is inconsistent with the result "01/02/1930" in step (2).
Updated Answer: Let's think step by step.
(1) If yesterday was 12/31/1929, then today is 01/01/1930.
(2) The date tomorrow is 01/02/1930.
(Final response) So the answer is (B). [END]
Q: Jane got her job in 2016. Today is her 3-year work anniversary. She still remember that on Dec 2, her second day at work, she spilled coffee on her laptop. What is the date tomorrow in MM/DD/YYYY?
Answer: Let's think step by step.
(A) 01/02/2020
(B) 08/02/2020
(C) 12/01/2019
(D) 12/03/2019
(E) 11/18/2019
(F) 12/02/2019
Answer: Let's think step by step.
(1) If Jane got her job in 2016 and today is her 3-year work anniversary, then today's date is Dec 2, 2019.
(2) The date tomorrow is Dec 3, 2019, that is 12/03/2019.
(Final response) So the answer is (D).
Feedback: In step (1) the part "today's date is Dec 2, 2019" is incorrect. This is because it is inconsistent with the question statement that Dec 2 is the second day at work.
Updated Answer: Let's think step by step.
(1) If Dec 2 is the second day at work, then Jane's first day at work is Dec 1, 2016.
(2) If today is her 3-year work anniversary, then today's date is Dec 1, 2019.
(3) The date tomorrow is Dec 2, 2019, that is 12/02/2019.
(Final response) So the answer is (F). [END]
Q: Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date one week ago from today in MM/DD/YYYY?
Options:
(A) 03/15/2011
(B) 12/24/2010
(C) 01/17/2011
(D) 01/15/2011
(E) 01/14/2011
Answer: Let's think step by step.
(1) If yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings, then yesterday is Jan 21, 2011.
(2) The date one week ago from today is Jan 14, 2011, that is 01/14/2011.
(Final response) So the answer is (E).
Feedback: Step (1) is correct. In step (2) the part "one week ago from today is Jan 14, 2011" is incorrect. This is because Jan 14, 2011 is one week ago from Jan 21, 2011, which is the date of yesterday, not today.
Updated Answer: Let's think step by step.
(1) If yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings, then today is Jan 22, 2011.
(2) The date one week ago from today is Jan 15, 2011, that is 01/15/2011.
(Final response) So the answer is (D). [END]
""".strip()

EXAMPLES_DATE_UNDERSTANDING_FEEDBACK_NOCORRECT_TABULAR = """
Q: Yesterday was 12/31/1929. Today could not be 12/32/1929 because December has only 31 days. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 12/12/1929
(B) 01/01/1930
(C) 01/02/1998
(D) 01/02/1885
(E) 01/02/1930
(F) 12/23/1929
Answer: Let's think step by step.
(1) If yesterday was 12/31/1929, then today is 01/01/1930.
(2) The date tomorrow is 01/02/1930.
(Final response) So the answer is (F).
Earliest error step: (Final response)
Error segment: "the answer is (F)"
Error reason: (F) 12/23/1929 is inconsistent with the result "01/02/1930" in step (2). [END]
Q: Jane got her job in 2016. Today is her 3-year work anniversary. She still remember that on Dec 2, her second day at work, she spilled coffee on her laptop. What is the date tomorrow in MM/DD/YYYY?
Answer: Let's think step by step.
(A) 01/02/2020
(B) 08/02/2020
(C) 12/01/2019
(D) 12/02/1950
(E) 11/18/2019
(F) 12/02/2019
Answer: Let's think step by step.
(1) If Jane got her job in 2016 and today is her 3-year work anniversary, then today's date is Dec 2, 2019.
(2) The date tomorrow is Dec 3, 2019, that is 12/03/2019.
(Final response) So the answer is (F).
Earliest error step: (1)
Error segment: "today's date is Dec 2, 2019"
Error reason: It is inconsistent with the question statement that Dec 2 is the second day at work. [END]
Q: Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date one week ago from today in MM/DD/YYYY?
Options:
(A) 03/15/2011
(B) 12/24/2010
(C) 01/17/2011
(D) 01/15/2011
(E) 12/06/2010
Answer: Let's think step by step.
(1) If yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings, then yesterday is Jan 21, 2011.
(2) The date one week ago from today is Jan 14, 2011, that is 01/14/2011.
(Final response) So the answer is (D).
Earliest error step: (2)
Error segment: "one week ago from today is Jan 14, 2011"
Error reason: Jan 14, 2011 is one week ago from Jan 21, 2011, which is the date of yesterday, not today. [END]
Q: Yesterday was April 30, 2021. What is the date a month ago in MM/DD/YYYY?
(A) 03/01/2021
(B) 03/22/2021
(C) 04/01/2021
(D) 03/18/2021
(E) 03/31/2021
(F) 02/21/2021
Answer: Let's think step by step.
(1) If yesterday was April 30, 2021, then today's date is April 1, 2021.
(2) The date a month ago is March 1, 2021, that is 03/01/2021.
(Final response) So the answer is (A).
Earliest error step: (1)
Error segment: "today's date is April 1, 2021"
Error reason: One day after April 30, 2021 is May 1, 2021. [END]
""".strip()


EXAMPLES_DATE_UNDERSTANDING_FEEDBACK_HASCORRECT_TABULAR = """
Q: Yesterday was 12/31/1929. Today could not be 12/32/1929 because December has only 31 days. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 12/12/1929
(B) 01/01/1930
(C) 01/02/1998
(D) 01/02/1885
(E) 01/02/1930
(F) 12/23/1929
Answer: Let's think step by step.
(1) If yesterday was 12/31/1929, then today is 01/01/1930.
(2) The date tomorrow is 01/02/1930.
(Final response) So the answer is (F).
Earliest error step: (Final response)
Error segment: "the answer is (F)"
Error reason: (F) 12/23/1929 is inconsistent with the result "01/02/1930" in step (2). [END]
Q: Jane got her job in 2016. Today is her 3-year work anniversary. She still remember that on Dec 2, her second day at work, she spilled coffee on her laptop. What is the date tomorrow in MM/DD/YYYY?
Answer: Let's think step by step.
(A) 01/02/2020
(B) 08/02/2020
(C) 12/01/2019
(D) 12/03/2019
(E) 11/18/2019
(F) 12/02/2019
Answer: Let's think step by step.
(1) If Dec 2 is the second day at work, then Jane's first day at work is Dec 1, 2016.
(2) If today is her 3-year work anniversary, then today's date is Dec 1, 2019.
(3) The date tomorrow is Dec 2, 2019, that is 12/02/2019.
(Final response) So the answer is (F).
Earliest error step: None
Error segment: None
Error reason: The final response is correct. [END]
Q: Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date one week ago from today in MM/DD/YYYY?
Options:
(A) 03/15/2011
(B) 12/24/2010
(C) 01/17/2011
(D) 01/15/2011
(E) 12/06/2010
Answer: Let's think step by step.
(1) If yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings, then yesterday is Jan 21, 2011.
(2) The date one week ago from today is Jan 14, 2011, that is 01/14/2011.
(Final response) So the answer is (D).
Earliest error step: (2)
Error segment: "one week ago from today is Jan 14, 2011"
Error reason: Jan 14, 2011 is one week ago from Jan 21, 2011, which is the date of yesterday, not today. [END]
Q: Yesterday was April 30, 2021. What is the date a month ago in MM/DD/YYYY?
(A) 03/01/2021
(B) 03/22/2021
(C) 04/01/2021
(D) 03/18/2021
(E) 03/31/2021
(F) 02/21/2021
Answer: Let's think step by step.
(1) If yesterday was April 30, 2021, then today's date is April 1, 2021.
(2) The date a month ago is March 1, 2021, that is 03/01/2021.
(Final response) So the answer is (A).
Earliest error step: (1)
Error segment: "today's date is April 1, 2021"
Error reason: One day after April 30, 2021 is May 1, 2021. [END]
""".strip()